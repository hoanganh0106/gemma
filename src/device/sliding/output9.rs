//! `ops::sliding_attention_output`: the projection as in output5 (two symmetric weight tiles, f8 x f8,
//! the level lanes summed inside the contraction), but the cross-cluster hop carries the projected
//! VALUES, not partial sums, and the whole tail runs in sixteen ADJACENT slices:
//! every slice holds 240 consecutive rows of y, so the global mean square is one intra-slice reduce
//! and one ring of sixteen, and the final pass, the residual add and the one store need nothing else.
//! The hop's wait is meant to hide under the operand loads issued between the store and the reload.

use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, H, Ns, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Lq = 2, Vc = 2, Vr = 16];

const H_F32: f32 = H::SIZE as f32;
const X_SCALE: f32 = 16.0;
/// z stays 16 times too large to the end; the RMSNorm divides that out, given a matching epsilon.
const EPS_SCALED: f32 = EPS * X_SCALE * X_SCALE;

type Residual = DmTensor<f32, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>;
type Levels = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq, Qs % 256]>;

/// Rounds the residual to f8 into lane `level`.
fn quantise_level(ctx: &mut Context, residual: &Residual, levels: &mut Levels, level: usize) {
    ctx.main
        .begin(residual.view())
        .fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(levels.view_mut().tile::<m![Lq], 1, m![Lq = 1 #{!} 2, Qs % 256]>(level));
}

/// What lane `level` failed to represent: `residual - levels[level]`, exact in f32.
fn residual_after(ctx: &mut Context, residual: &Residual, levels: &Levels, level: usize) -> Residual {
    let residual_vrf: VrfTensor<f32, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .to_vrf();
    let next: DmTensor<f32, Chip, OutputClusters, SlidingOutputColumns, m![Lq = 1, Qs % 256]> = ctx
        .main
        .begin(levels.view().tile::<m![Lq], 1, m![Lq = 1 # 2, Qs % 256]>(level))
        .fetch::<m![Lq = 1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Lq = 1, Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Lq = 1, Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &residual_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -1.0f32)
        .vector_widen_concat::<m![Lq = 1, Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    unsafe { next.reshape() }
}

pub(crate) type Rows = SlidingOutputRows;
/// The tail's home: slices 0..16 of both clusters (replicated), slice g holding rows 240g..240g+240.
pub(crate) type Tail = m![1 # 16, H / 240];
pub(crate) type TailDm<D> = DmTensor<D, Chip, m![Vc], Tail, m![H % 240]>;
pub(crate) type TailVrf = VrfTensor<f32, Chip, m![Vc], Tail, m![H % 240]>;

pub(crate) fn tail_operand(ctx: &mut Context, v: &HbmTensor<bf16, Chip, m![H]>) -> TailVrf {
    let dm: TailDm<bf16> = v.to_dm(&mut ctx.tdma);
    ctx.sub
        .begin(dm.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf()
}

pub(crate) type XTrf = TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>;
pub(crate) type Z = DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]>;

pub(crate) type WeightTileA = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 104, Qs % 256]>;
pub(crate) type WeightTileB = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 16, Qs % 256]>;

/// Rows `offset..offset + 104` of every row slice: the two levels summed inside the pass.
pub(crate) fn contract_tile_a(ctx: &mut Context, weight: &WeightTileA, x_trf: &XTrf, z: &mut Z, offset: usize) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 104, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 104, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 104]>()
        .contract_lane::<m![H % 120 = 104, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 104], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 104]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 104 / 4], m![H % 120 = 104 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 104 % 4]>()
        .commit_view(z.view_mut().tile::<m![H % 120], 104, m![H % 120 = 104 #{!} 120]>(offset));
}

/// Rows `offset..offset + 16` of every row slice.
pub(crate) fn contract_tile_b(ctx: &mut Context, weight: &WeightTileB, x_trf: &XTrf, z: &mut Z, offset: usize) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 16, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 16, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 16]>()
        .contract_lane::<m![H % 120 = 16, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 16], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 16 / 4], m![H % 120 = 16 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 16 % 4]>()
        .commit_view(z.view_mut().tile::<m![H % 120], 16, m![H % 120 = 16 #{!} 120]>(offset));
}

/// x as two exact f8 levels in the TRF: q0 = f8(16x), q1 = f8(16x - q0), exact for a bf16 x
/// (see output5::project_quantised for the argument).
pub(crate) fn quantise_x(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
) -> XTrf {
    let mut levels: Levels = DmTensor::new();
    let r0: Residual = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), X_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    quantise_level(ctx, &r0, &mut levels, 0);
    let r1 = residual_after(ctx, &r0, &levels, 0);
    quantise_level(ctx, &r1, &mut levels, 1);
    ctx.sub
        .begin(levels.view())
        .fetch::<m![Lq], m![Qs % 256]>()
        .collect::<m![Lq, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf()
}

pub(crate) fn project_normalize_add(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The weight in two tiles, 104 and 16 rows per slice: the big tile is contracted while the
    // small one is still on its way.
    let weight_a: WeightTileA = weight
        .view()
        .tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let weight_b: WeightTileB = weight
        .view()
        .tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104)
        .to_dm(&mut ctx.tdma);

    // One DMA load hands every slice the 256 columns it contracts, once per row group.
    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };
    let x_trf = quantise_x(ctx, &x);

    // 16 * (weight @ x), bf16, 120 rows per row slice.
    let mut z: Z = DmTensor::new();
    contract_tile_a(ctx, &weight_a, &x_trf, &mut z, 0);
    contract_tile_b(ctx, &weight_b, &x_trf, &mut z, 104);

    // The residual, loaded here so that the DMA has work while the small tile is contracted.
    let residual = tail_operand(ctx, residual_hbm);

    // The hop: all 3,840 projected values go to HBM in row order (the store is the same shape as
    // the kernel's output store) and come back, 240 consecutive rows per slice, into slices 0..16.
    let mut hop: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    z.view().to_hbm_view(&mut ctx.tdma, hop.view_mut());

    let scale = tail_operand(ctx, weight_scale);
    let rms_weight = tail_operand(ctx, rms_weight);

    let z_tail: TailDm<bf16> = hop.to_dm(&mut ctx.tdma);

    // Sum of squares of the 240 scaled values of each slice, over H.
    let partial: DmTensor<f32, Chip, m![Vc], Tail, m![1 # 8]> = ctx
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // Ring of sixteen: the mean square in every slice, epsilon, root.
    let rms: DmTensor<f32, Chip, m![Vc], m![1 # 16, Vr], m![1 # 8]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, Vr], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS_SCALED)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, m![Vc], Tail, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, m![Vc], Tail, m![1 # 8]> =
        ctx.sub.begin(rms.view()).fetch::<m![1], m![1 # 8]>().collect::<m![1], m![1 # 8]>().to_vrf();

    let out: TailDm<bf16> = ctx
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &rms_weight)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let out: DmTensor<bf16, Chip, m![1 # 2], Tail, m![H % 240]> = unsafe { out.reshape() };
    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}
