//! f8 x f8 contractions: no decode table, ~3.5x faster on the device than table-fused f8 -> bf16.
//! Q/K/V projections with WHOLE weight rows per slice (one contiguous HBM run per slice, no
//! inter-slice reduce), gathered per KV head on the Switch network. The heads stay on the
//! cluster that projected them: no DMA gather, no HBM round trip.
use furiosa_opt_std::prelude::*;

use super::{HeadClusters, HeadSlices, KeyValueRowSlices, KvRowsByHead, QueryRowSlices, QueryRowsByHead, Term};
use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{KeyValueClusters, QueryClusters};


pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Gs, Ds]> {
    let x_trf = query_operand(ctx, x);

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction = contract_query(ctx, &weight_f8, &x_trf);
    // Relabel only: Qs = (Ns, Gs, Ds) row-major, so cluster Qs/2048 is Ns/4, slice Qs/8%256 is
    // (Ns%4, Gs, Ds/8) and the in-slice Qs%8 is Ds%8. Same physical order.
    let contraction: DmTensor<bf16, Chip, HeadClusters, QueryRowsByHead, m![Ds % 8]> =
        unsafe { contraction.reshape() };

    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Gs, Ds]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, HeadClusters, HeadSlices, m![Gs, Ds]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    // One pass gathers a head's 64 row slices into the head's slice and applies the scale.
    ctx.main
        .begin(contraction.view())
        .fetch::<m![1], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .switch::<HeadSlices, m![Gs, Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 1.0 / super::X_PRESCALE)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4, H]> =
        weight.to_dm(&mut ctx.tdma);

    let contraction = contract_key_value(ctx, &weight_f8, x_trf);
    // Relabel only: Ps = (Ns, Ds) row-major: cluster Ns/4, slice (Ns%4, Ds/4), in-slice Ds%4.
    let contraction: DmTensor<bf16, Chip, HeadClusters, KvRowsByHead, m![Ds % 4]> = unsafe { contraction.reshape() };

    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> = unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, HeadClusters, HeadSlices, m![Ds]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    ctx.main
        .begin(contraction.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<HeadSlices, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
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
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
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
/// vector engine adds the two lane results while they are still f32.
fn contract_query(
    ctx: &mut Context,
    weight_f8: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Qs % 8, H]>,
    x_trf: &QueryOperand,
) -> DmTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![Qs % 8]> {
    ctx.main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .collect::<m![Qs % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8, Term], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Term, m![Qs % 8], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit()
}

fn contract_key_value(
    ctx: &mut Context,
    weight_f8: &DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4, H]>,
    x_trf: &KvOperand,
) -> DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4]> {
    ctx.main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .collect::<m![Ps % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4, Term], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Term, m![Ps % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit()
}
