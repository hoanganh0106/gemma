//! `ops::sliding_attention_output`, end to end. The projection result never leaves the sixteen
//! slices per cluster that reduced it: the channel scale, the RMSNorm and the residual add all run
//! 120 elements wide there, and the only thing that crosses clusters is the sum of squares.

use furiosa_opt_std::prelude::*;

use crate::axes::{H, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Oc = 2, Ok = 2, Or = 16, Og = 32, Ow = 8];

const H_F32: f32 = H::SIZE as f32;

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

pub(crate) fn project_normalize_add(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    let x_f8: DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    let x_trf: TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x_f8.view())
        .fetch::<m![1], m![Qs % 256]>()
        .collect::<m![Qs / 32 % 8], m![Qs % 32]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]> =
        weight.to_dm(&mut ctx.tdma);

    let scale_vrf = to_vrf_120(ctx, weight_scale);
    let rms_weight_vrf = to_vrf_120(ctx, rms_weight);
    let residual_vrf = to_vrf_120(ctx, residual_hbm);

    let y: DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120, Qs / 64 % 4], m![Qs % 64], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<Rows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    // Sum of squares of the 120 scaled rows each row slice holds.
    let sum_squares: DmTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]> = ctx
        .main
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
        .vector_clip(ClipBinaryOpF32::Add, EPS)
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
