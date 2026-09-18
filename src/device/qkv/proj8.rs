//! f8 x f8 contractions: no decode table, ~3.5x faster on the device than table-fused f8 -> bf16.
//! Q/K/V projections with WHOLE weight rows per slice (no inter-slice reduce), the rows strided
//! over the slices for a faster DMA (see `super::QueryRowSlices`), sorted back with one
//! `InterTranspose` pass and gathered per KV head on the Switch network. The heads stay on the
//! cluster that projected them: no DMA gather, no HBM round trip.
use furiosa_opt_std::prelude::*;

use super::pool::{SmallPool, Vrf1, Vrf2};
use super::{HeadCopy4, HeadCopyClusters, HeadCopySlices, KeyValueRowSlices, Kv, KvRowsByHead, Pool, QueryRowSlices, QueryRowsByHead, Term};
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

/// The contraction of a weight tensor whose DM address is an ODD multiple of 256 bytes takes
/// 4.9K device cycles instead of 2.1K (measured across 37 traces: every build whose V weight sits
/// at a 512-aligned DM offset contracts in 2.1K, the two whose V weight sits at a 256-odd offset
/// take 4.8-4.9K). The allocator puts the V weight at such an offset in this build, so a K/V weight
/// is loaded into ONE TILE of a tensor twice its size: tile 1 starts 15,360 = 30 x 512 bytes into
/// the tensor, so it is 512-aligned whenever the tensor's own 256-aligned base is, and the pair of
/// tiles shifts everything the allocator packs after it by a multiple of 512. Tile 0 of V's tensor
/// (and tile 1 of K's) is never written or read: 15 KB of the 512 KB of a slice.
type KvPad = DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Kv, Ds / 8 % 4, H]>;
type KvTile<'l> = DmTensorView<'l, f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Kv = 1 # 2, Ds / 8 % 4, H]>;

fn contract_kv_matrix(
    ctx: &mut Context,
    x_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    tile: usize,
    store: &mut KvPad,
) -> DmTensor<bf16, Chip, HeadCopyClusters, m![HeadCopy4, Ds / 4], m![Ds % 4]> {
    let weight: HbmTensorView<'_, f8e4m3, Chip, m![Kv = 1, Ns, Ds, H]> = unsafe { weight.view().reshape() };
    weight.to_dm_view(
        &mut ctx.tdma,
        store.view_mut().tile::<m![Kv], 1, m![Kv = 1 #{!} 2, Ds / 8 % 4, H]>(tile),
    );

    let contraction = contract_key_value(ctx, store.view().tile::<m![Kv], 1, m![Kv = 1 # 2, Ds / 8 % 4, H]>(tile), x_trf);
    // Relabel only: Ps = (Ns, Ds) row-major: cluster Ns/4, slice (Ns%4, Ds/4), in-slice Ds%4.
    unsafe { contraction.reshape() }
}

/// One pass gathers a head's 64 row slices into the head's slice and applies the row scale.
fn project_value_matrix(
    ctx: &mut Context,
    x_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale_vrf: &Vrf1,
    store: &mut KvPad,
) -> DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]> {
    let contraction = contract_kv_matrix(ctx, x_trf, weight, 1, store);
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

/// The same pass for K, committing the head vector into tile `pool::K_HEAD` of the pool: a tile
/// write, chained in program order with the RoPE rows' reload that follows it.
fn project_key_matrix(
    ctx: &mut Context,
    x_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale_vrf: &Vrf1,
    pool: &mut SmallPool,
    store: &mut KvPad,
) {
    let contraction = contract_kv_matrix(ctx, x_trf, weight, 0, store);
    let contraction: DmTensorView<'_, bf16, Chip, HeadCopyClusters, m![HeadCopy4, Ds / 4], m![Pool = 1, Ds % 4]> =
        unsafe { contraction.view().reshape() };
    ctx.main
        .begin(contraction)
        .fetch::<m![Pool = 1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<HeadCopySlices, m![Pool = 1, Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Pool = 1, Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), weight_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 1.0 / super::X_PRESCALE)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit_view(pool.view_mut().tile::<m![Pool], 1, m![Pool = 1 #{!} 10, Ds]>(super::pool::K_HEAD));
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Term, H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &Vrf1,
    v_weight_scale: &Vrf1,
    pool: &mut SmallPool,
) -> DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]> {
    let x_trf = key_value_operand(ctx, x);

    let mut k_store: KvPad = DmTensor::new();
    let mut v_store: KvPad = DmTensor::new();
    project_key_matrix(ctx, &x_trf, k_weight, k_weight_scale, pool, &mut k_store);
    project_value_matrix(ctx, &x_trf, v_weight, v_weight_scale, &mut v_store)
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
    weight_f8: KvTile<'_>,
    x_trf: &KvOperand,
) -> DmTensor<bf16, Chip, KeyValueClusters, KvRowsByHead, m![Ds % 4]> {
    let sums: DmTensor<f32, Chip, KeyValueClusters, KeyValueRowSlices, m![Ds / 8 % 4, 1 # 8]> = ctx
        .main
        .begin(weight_f8)
        .fetch::<m![Kv = 1, Ds / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ds / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 8 % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 8 % 4]>()
        .contract_lane::<m![Ds / 8 % 4, Term], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Term, m![Ds / 8 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let sorted: DmTensor<bf16, Chip, KeyValueClusters, m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2], m![Ds % 4]> = ctx
        .main
        .begin(sums.view())
        .fetch::<m![Ds / 8 % 4], m![1 # 8]>()
        .switch::<m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2], m![Ds % 4]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 2,
            time0: 1,
        })
        .collect::<m![Ds % 4], m![1 # 8]>()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    unsafe { sorted.reshape() }
}
