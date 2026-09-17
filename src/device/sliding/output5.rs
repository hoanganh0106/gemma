//! `ops::sliding_attention_output`, end to end. The projection result never leaves the sixteen
//! slices per cluster that reduced it: the channel scale, the RMSNorm and the residual add all run
//! 120 elements wide there, and the only thing that crosses clusters is the sum of squares.

use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, H, Ns, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Lq = 2, Oc = 2, Ok = 2, Or = 16, Og = 32, Ow = 8, Ob = 8, Orep = 8, Oring = 32];

pub(crate) const H_F32: f32 = H::SIZE as f32;
pub(crate) const X_SCALE: f32 = 16.0;
/// y stays 16 times too large to the end; the RMSNorm divides that out, given a matching epsilon.
pub(crate) const EPS_SCALED: f32 = EPS * X_SCALE * X_SCALE;

pub(crate) type Rows = SlidingOutputRows;
/// The sixteen row slices of a cluster, named as a replication axis.
pub(crate) type RowsReplicated = m![Or, 1 # 16];

pub(crate) fn to_vrf_120(
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
pub(crate) type Residual = DmTensor<f32, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>;
pub(crate) type Levels = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq, Qs % 256]>;

/// Rounds the residual to f8 into lane `level`.
pub(crate) fn quantise_level(ctx: &mut Context, residual: &Residual, levels: &mut Levels, level: usize) {
    ctx.main
        .begin(residual.view())
        .fetch::<m![Qs / 8 % 32], m![Qs % 8]>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(levels.view_mut().tile::<m![Lq], 1, m![Lq = 1 #{!} 2, Qs % 256]>(level));
}

/// What lane `level` failed to represent: `residual - levels[level]`, exact in f32.
pub(crate) fn residual_after(ctx: &mut Context, residual: &Residual, levels: &Levels, level: usize) -> Residual {
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

pub(crate) type WeightTileA = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 96, Qs % 256]>;

/// Ninety-six rows per slice against every level of x, then the levels added up: rows `offset..` of `y`.
pub(crate) fn contract_tile_a(
    ctx: &mut Context,
    weight: &WeightTileA,
    x_trf: &TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>,
    y: &mut DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]>,
    offset: usize,
) {
    // The sixteen column partials of a row group meet in the inter-slice reduce, still f32. The
    // transpose lays the result out as (4 rows) x (levels) x (row in 4) so that the next pass can
    // add the levels up along time and keep four rows per packet.
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 96, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 96, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 96, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 96]>()
        .contract_lane::<m![H % 120 = 96, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 96], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 96]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 96 / 4], m![H % 120 = 96 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 96 % 4]>()
        .commit_view(y.view_mut().tile::<m![H % 120], 96, m![H % 120 = 96 #{!} 120]>(offset));
}

pub(crate) type WeightTileB = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 24, Qs % 256]>;

/// Twenty-four rows per slice against every level of x, then the levels added up: rows `offset..` of `y`.
pub(crate) fn contract_tile_b(
    ctx: &mut Context,
    weight: &WeightTileB,
    x_trf: &TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>,
    y: &mut DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]>,
    offset: usize,
) {
    // The sixteen column partials of a row group meet in the inter-slice reduce, still f32. The
    // transpose lays the result out as (4 rows) x (levels) x (row in 4) so that the next pass can
    // add the levels up along time and keep four rows per packet.
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 24, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 24, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 24, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 24]>()
        .contract_lane::<m![H % 120 = 24, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 24], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 24]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 24 / 4], m![H % 120 = 24 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 24 % 4]>()
        .commit_view(y.view_mut().tile::<m![H % 120], 24, m![H % 120 = 24 #{!} 120]>(offset));
}

/// `16 * scale * (weight @ x)` for the 120 rows of every row slice, by an f8 x f8 contraction.
pub(crate) fn project_quantised(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
    weight_a: &WeightTileA,
    weight_b: &WeightTileB,
) -> DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]> {
    // f8 x f8 contraction: no decode table, and under a third of the device time of the
    // table-fused one. x is not an f8 tensor, so it goes in as two f8 levels, one TRF lane each:
    // q0 = f8(16x) and q1 = f8(16x - q0). That is EXACT for a bf16 x: q0 rounds to 4 significant
    // bits, so the residual is at most 16 units of x's last (9th) bit, and every integer up to 16
    // has 4 significant bits. (Measured on the device with x of varied magnitudes: reconstruction
    // error 6.3e-4 relative with q0 alone, below 1e-12 with q0 + q1.) The factor 16 keeps q1 out of
    // f8e4m3's subnormals down to |x| = 1/32 -- below that the error is at most 6e-5 absolute --
    // and q0 saturates only beyond |x| = 28; x here is a convex mix of RMS-normalized 256-vectors,
    // so |x| <= 16. All of this runs on the otherwise idle tensor unit while the weight loads.
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
    let x_trf: TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]> = ctx
        .sub
        .begin(levels.view())
        .fetch::<m![Lq], m![Qs % 256]>()
        .collect::<m![Lq, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();


    let mut y: DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]> = DmTensor::new();
    contract_tile_a(ctx, weight_a, &x_trf, &mut y, 0);
    contract_tile_b(ctx, weight_b, &x_trf, &mut y, 96);

    y
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
    // The channel scale is applied where the levels are added up, tile by tile. That is also what
    // makes the scheduler load it BEFORE the second weight tile instead of into the tail.
    let scale_vrf = to_vrf_120(ctx, weight_scale);
    let rms_weight_vrf = to_vrf_120(ctx, rms_weight);
    let residual_vrf = to_vrf_120(ctx, residual_hbm);

    // The weight in two tiles, 96 and 24 rows per slice: the big tile is contracted while the small
    // one is still on its way, and only the small one's contraction is left after the transfers.
    // (No decode table any more, so a tile costs only a DMA op.)
    let weight_a: WeightTileA = weight
        .view()
        .tile::<m![H % 120], 96, m![H / 120, H % 120 = 96 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let weight_b: WeightTileB = weight
        .view()
        .tile::<m![H % 120], 24, m![H / 120, H % 120 = 24 # 120, Qs]>(96)
        .to_dm(&mut ctx.tdma);

    // One DMA load hands every slice the 256 columns it contracts, once per row group.
    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };

    let y = project_quantised(ctx, &x, &weight_a, &weight_b);

    // Sum of squares of the 120 scaled rows each row slice holds.
    let mut sum_squares: DmTensor<f32, Chip, OutputClusters, Rows, m![Ob, 1 # 8]> = DmTensor::new();
    ctx.main
        .begin(y.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit_view(sum_squares.view_mut().tile::<m![Ob], 1, m![Ob = 1 #{!} 8, 1 # 8]>(0));

    // The 32 partial sums are the one thing that has to cross clusters, and HBM is the only place
    // the clusters meet. All 32 come back to every row slice.
    // A scattered store of 32-byte pieces is very slow (unaligned tails), so each partial sum sits
    // at the head of a 256-byte element and the store is block-aligned; only the heads come back.
    // (An HBM tensor cannot end in padding, so the pad words ride along as a real axis.)
    let partials: DmTensor<f32, Chip, m![Oc], RowsReplicated, m![Ob, Ow]> = unsafe { sum_squares.reshape() };
    let partials: HbmTensor<f32, Chip, m![Oc, Or, Ob, Ow]> = partials.to_hbm(&mut ctx.tdma);
    let partials: HbmTensor<f32, Chip, m![Og, Ob, Ow]> = unsafe { partials.reshape() };
    // One partial per slice, every ring of 32 adjacent slices holds all 32 of them (eight rings per
    // cluster), so ONE pass all-reduces them, adds the epsilon and takes the root -- in every slice,
    // the row slices included.
    let partials: DmTensor<f32, Chip, m![Ok], m![Orep, Og], m![Ob = 1, Ow]> = partials
        .view()
        .tile::<m![Ob], 1, m![Og, Ob = 1 # 8, Ow]>(0)
        .to_dm(&mut ctx.tdma);
    let partials: DmTensor<f32, Chip, m![Ok], m![Orep, Og], m![1 # 8]> = unsafe { partials.reshape() };
    let rms: DmTensor<f32, Chip, m![Ok], m![Orep, Oring], m![1 # 8]> = ctx
        .main
        .begin(partials.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Orep, Oring], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS_SCALED)
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
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
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
