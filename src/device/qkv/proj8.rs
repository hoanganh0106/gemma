//! f8 x f8 contractions: no decode table, ~3.5x faster on the device than table-fused f8 -> bf16.
//! Q/K/V projections with WHOLE weight rows per slice (no inter-slice reduce), the rows strided
//! over the slices for a faster DMA (see `super::QueryRowSlices`), sorted back with one
//! `InterTranspose` pass and gathered per KV head on the Switch network. The heads stay on the
//! cluster that projected them: no DMA gather, no HBM round trip.
use furiosa_opt_std::prelude::*;

use super::pool::{Vrf1, Vrf2};
use super::{HeadCopy4, HeadCopyClusters, HeadCopySlices, KeyValueRowSlices, KvRowsByHead, Pool, QueryRowSlices, QueryRowsByHead, Term};
use crate::Chip;
use super::HeadClusters;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
type QueryClusters = HeadClusters;
type KeyValueClusters = HeadClusters;


pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale_vrf: &Vrf2,
) -> DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Pool = 2, Ds]> {
    let x_trf = query_operand(ctx, x);

    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Ns, Gs, Ds, H]> = unsafe { weight.view().reshape() };
    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Ds / 8 % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction = contract_query(ctx, &weight_f8, &x_trf);
    // Relabel only: Qs = (Ns, Gs, Ds) row-major, so cluster Qs/2048 is Ns/4, slice Qs/8%256 is
    // (Ns%4, Gs, Ds/8) and the in-slice Qs%8 is Ds%8. Same physical order.
    let contraction: DmTensor<bf16, Chip, HeadCopyClusters, m![HeadCopy4, Pool = 2, Ds / 8], m![Ds % 8]> =
        unsafe { contraction.reshape() };

    // One pass gathers a head's 64 row slices into the head's slice and applies the scale.
    ctx.main
        .begin(contraction.view())
        .fetch::<m![1], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .switch::<HeadCopySlices, m![Pool = 2, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Pool = 2, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Pool = 2, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 1.0 / super::X_PRESCALE)
        .vector_widen_concat::<m![Pool = 2, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale_vrf: &Vrf1,
) -> DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]> {
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Ns, Ds, H]> = unsafe { weight.view().reshape() };
    let weight_f8: DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ds / 4 % 4, H]> =
        weight.to_dm(&mut ctx.tdma);

    let contraction = contract_key_value(ctx, &weight_f8, x_trf);
    // Relabel only: Ps = (Ns, Ds) row-major: cluster Ns/4, slice (Ns%4, Ds/4), in-slice Ds%4.
    let contraction: DmTensor<bf16, Chip, HeadCopyClusters, m![HeadCopy4, Ds / 4], m![Ds % 4]> =
        unsafe { contraction.reshape() };

    ctx.main
        .begin(contraction.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<HeadCopySlices, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 1.0 / super::X_PRESCALE)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &Vrf1,
    v_weight_scale: &Vrf1,
) -> (
    DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]>,
    DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]>,
) {
    let x_trf = key_value_operand(ctx, x);

    let k = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (k, v)
}

/// The two f8 terms of the hidden state, one per TRF lane: the contraction multiplies the weight
/// stream against both lanes at once, so the second term costs no extra pass.
type QueryOperand = TrfTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term], m![H]>;
type KvOperand = TrfTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Term], m![H]>;

fn query_operand(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
) -> QueryOperand {
    ctx.sub
        .begin(x.view())
        .fetch::<m![Term], m![H]>()
        .collect::<m![Term, H / 32], m![H % 32]>()
        .to_trf()
}

fn key_value_operand(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
) -> KvOperand {
    let x: DmTensorView<'_, f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Term, H]> =
        unsafe { x.view().reshape() };
    ctx.sub.begin(x).fetch::<m![Term], m![H]>().collect::<m![Term, H / 32], m![H % 32]>().to_trf()
}

/// The Q projection, f8 x f8: 8 whole rows per slice against both terms of H (one per lane); the
/// vector engine adds the two lane results while they are still f32. The rows of a slice are 8
/// apart (`Ds / 8 % 8` with `Ds % 8` in the slice axis), so a second small pass swaps that slice
/// axis with the time axis and every slice ends with 8 consecutive rows.
fn contract_query(
    ctx: &mut Context,
    weight_f8: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Ds / 8 % 8, H]>,
    x_trf: &QueryOperand,
) -> DmTensor<bf16, Chip, QueryClusters, QueryRowsByHead, m![Ds % 8]> {
    let sums: DmTensor<f32, Chip, QueryClusters, QueryRowSlices, m![Ds / 8 % 8, 1 # 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ds / 8 % 8, H / 32], m![H % 32]>()
        .collect::<m![Ds / 8 % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 8 % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 8 % 8]>()
        .contract_lane::<m![Ds / 8 % 8, Term], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Term, m![Ds / 8 % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let sorted: DmTensor<bf16, Chip, QueryClusters, m![Ns % 4, Gs, Ds / 64, Ds / 8 % 8], m![Ds % 8]> = ctx
        .main
        .begin(sums.view())
        .fetch::<m![Ds / 8 % 8], m![1 # 8]>()
        .switch::<m![Ns % 4, Gs, Ds / 64, Ds / 8 % 8], m![Ds % 8]>(SwitchConfig::InterTranspose {
            slice1: 8,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Ds % 8], m![1 # 8]>()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ds / 4 % 2], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    unsafe { sorted.reshape() }
}

fn contract_key_value(
    ctx: &mut Context,
    weight_f8: &DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ds / 4 % 4, H]>,
    x_trf: &KvOperand,
) -> DmTensor<bf16, Chip, KeyValueClusters, KvRowsByHead, m![Ds % 4]> {
    let sums: DmTensor<f32, Chip, KeyValueClusters, KeyValueRowSlices, m![Ds / 4 % 4, 1 # 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ds / 4 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ds / 4 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 4 % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 4 % 4]>()
        .contract_lane::<m![Ds / 4 % 4, Term], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Term, m![Ds / 4 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let sorted: DmTensor<bf16, Chip, KeyValueClusters, m![Ns % 4, Ds / 16, Ds / 4 % 4], m![Ds % 4]> = ctx
        .main
        .begin(sums.view())
        .fetch::<m![Ds / 4 % 4], m![1 # 8]>()
        .switch::<m![Ns % 4, Ds / 16, Ds / 4 % 4], m![Ds % 4]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Ds % 4], m![1 # 8]>()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    unsafe { sorted.reshape() }
}
