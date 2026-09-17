//! `ops::sliding_attention_output`, end to end. The projection result never leaves the sixteen
//! slices per cluster that reduced it: the channel scale, the RMSNorm and the residual add all run
//! 120 elements wide there, and the only thing that crosses clusters is the sum of squares.

use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, H, Ns, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Lq = 4, Oc = 2, Ok = 2, Or = 16, Og = 32, Ow = 8];

const H_F32: f32 = H::SIZE as f32;
const X_SCALE: f32 = 16.0;
/// y stays 16 times too large to the end; the RMSNorm divides that out, given a matching epsilon.
const EPS_SCALED: f32 = EPS * X_SCALE * X_SCALE;

type Rows = SlidingOutputRows;
/// The sixteen row slices of a cluster, named as a replication axis.
type RowsReplicated = m![Or, 1 # 16];

fn to_vrf_120(
    ctx: &mut Context,
    v: &HbmTensor<bf16, Chip, m![H]>,
) -> VrfTensor<f32, Chip, OutputClusters, Rows, m![H % 120]> {
    let dm: DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]> = v.to_dm(&mut ctx.tdma);
    ctx.sub
        .begin(dm.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf()
}

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
        .commit_view(levels.view_mut().tile::<m![Lq], 1, m![Lq = 1 #{!} 4, Qs % 256]>(level));
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
        .begin(levels.view().tile::<m![Lq], 1, m![Lq = 1 # 4, Qs % 256]>(level))
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

type WeightTile = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 60, Qs % 256]>;

/// Sixty rows per slice against every level of x, then the levels added up: rows `offset..` of `y`.
fn contract_tile(
    ctx: &mut Context,
    weight: &WeightTile,
    x_trf: &TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>,
    y: &mut DmTensor<f32, Chip, OutputClusters, Rows, m![H % 120]>,
    offset: usize,
) {
    // The sixteen column partials of a row group meet in the inter-slice reduce, still f32. The
    // transpose lays the result out as (4 rows) x (levels) x (row in 4) so that the next pass can
    // add the levels up along time and keep four rows per packet.
    let y_levels: DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120 = 60 / 4, Lq, H % 120 = 60 % 4]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 60, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 60, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 60]>()
        .contract_lane::<m![H % 120 = 60], m![Lq # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 60]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Lq # 16]>()
        .transpose::<m![H % 120 = 60 / 4, Lq], m![H % 120 = 60 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 60 % 4]>()
        .commit();

    ctx.main
        .begin(y_levels.view())
        .fetch::<m![H % 120 = 60 / 4, Lq / 2], m![Lq % 2, H % 120 = 60 % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120 = 60 / 4, Lq / 2], m![Lq % 2, H % 120 = 60 % 4]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 120 = 60 / 4, Lq], m![H % 120 = 60 % 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 60 / 4], m![H % 120 = 60 % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![H % 120 = 60 % 4 # 8]>()
        .vector_final()
        .commit_trim::<m![H % 120 = 60 % 4]>()
        .commit_view(y.view_mut().tile::<m![H % 120], 60, m![H % 120 = 60 #{!} 120]>(offset));
}

pub(crate) fn project_normalize_add(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The three per-row operands of the tail: loaded and parked in the VRF up front.
    let scale_vrf = to_vrf_120(ctx, weight_scale);
    let rms_weight_vrf = to_vrf_120(ctx, rms_weight);
    let residual_vrf = to_vrf_120(ctx, residual_hbm);

    // The weight in two halves of 60 rows per slice, so the first half is contracted while the
    // second is still on its way. (No decode table any more, so a tile costs only a DMA op.)
    let weight_a: WeightTile = weight
        .view()
        .tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let weight_b: WeightTile = weight
        .view()
        .tile::<m![H % 120], 60, m![H / 120, H % 120 = 60 # 120, Qs]>(60)
        .to_dm(&mut ctx.tdma);

    // One DMA load hands every slice the 256 columns it contracts, once per row group.
    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };

    // f8 x f8 contraction: no decode table, a third of the device time of the table-fused one.
    // x is not an f8 tensor, so it is quantised by successive residuals, one TRF lane per level:
    // 16x = q0 + q1 + .. exactly up to the last level's rounding, 1/16 of the level before it. The
    // factor 16 keeps the residuals of ordinary activations (|x| <= 16 here: a convex mix of
    // RMS-normalized 256-vectors) inside f8e4m3's normal range. All of this runs on the otherwise
    // idle tensor unit while the weight is still loading.
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
    // No loop here: a `for` whose body rebinds the residual compiles, and every iteration then
    // reads the FIRST residual (seen in the schedule's tensor ids).
    quantise_level(ctx, &r0, &mut levels, 0);
    let r1 = residual_after(ctx, &r0, &levels, 0);
    quantise_level(ctx, &r1, &mut levels, 1);
    let r2 = residual_after(ctx, &r1, &levels, 1);
    quantise_level(ctx, &r2, &mut levels, 2);
    let r3 = residual_after(ctx, &r2, &levels, 2);
    quantise_level(ctx, &r3, &mut levels, 3);
    let x_trf: TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]> = ctx
        .sub
        .begin(levels.view())
        .fetch::<m![Lq], m![Qs % 256]>()
        .collect::<m![Lq, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();


    let mut y: DmTensor<f32, Chip, OutputClusters, Rows, m![H % 120]> = DmTensor::new();
    contract_tile(ctx, &weight_a, &x_trf, &mut y, 0);
    contract_tile(ctx, &weight_b, &x_trf, &mut y, 60);

    // Sum of squares of the 120 scaled rows each row slice holds.
    let sum_squares: DmTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]> = ctx
        .main
        .begin(y.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // The 32 partial sums are the one thing that has to cross clusters, and HBM is the only place
    // the clusters meet. All 32 come back to every row slice.
    // A scattered store of 32-byte pieces is very slow, so the DMA first collects a cluster's
    // sixteen into its first slice: 512 contiguous, block-aligned bytes per cluster.
    let partials: DmTensor<f32, Chip, m![Oc], RowsReplicated, m![1 # 8]> = unsafe { sum_squares.reshape() };
    let partials: DmTensor<f32, Chip, m![Oc], m![1 # 256], m![Or, 1 # 8]> = partials.to_dm(&mut ctx.tdma);
    // (An HBM tensor cannot end in padding, so the seven pad words ride along as a real axis.)
    let partials: DmTensor<f32, Chip, m![Oc], m![1 # 256], m![Or, Ow]> = unsafe { partials.reshape() };
    let partials: HbmTensor<f32, Chip, m![Oc, Or, Ow]> = partials.to_hbm(&mut ctx.tdma);
    let partials: HbmTensor<f32, Chip, m![Og, Ow]> = unsafe { partials.reshape() };
    let partials: DmTensor<f32, Chip, m![Ok], RowsReplicated, m![Og, Ow]> = partials.to_dm(&mut ctx.tdma);
    let partials: DmTensor<f32, Chip, m![Ok], RowsReplicated, m![Og, 1 # 8]> = unsafe { partials.reshape() };

    let mean_square: DmTensor<f32, Chip, m![Ok], RowsReplicated, m![1 # 8]> = ctx
        .main
        .begin(partials.view())
        .fetch::<m![Og], m![1 # 8]>()
        .collect::<m![Og], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Og, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS_SCALED)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, m![Ok], RowsReplicated, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]> = unsafe { rms.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]> = ctx
        .main
        .begin(y.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &rms_weight_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}
