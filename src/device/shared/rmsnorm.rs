
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy2, Dummy8, Dummy256, Gs, H};
use crate::device::layout::Cluster;

use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn normalize_replicated<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![Dummy256], m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // EXP-010A:
    // Fuse the final RMSNorm redistribution with the FFN replication.
    //
    // Instead of:
    //   ReducingSlices -> Slice -> TDMA -> Replicated
    //
    // try:
    //   ReducingSlices -> Replicated
    //
    // Arithmetic and FP32->BF16 rounding remain unchanged.
    // EXP-020A:
    // Narrow normalized FP32 -> BF16 in the Fetch Adapter before the
    // ring-256 replication. The baseline already rounds to BF16 after
    // the switch, so arithmetic and final storage precision are unchanged.
    //
    // Baseline:
    //   FP32 fetch -> ring256 -> collect -> BF16 cast -> DM
    //
    // Candidate:
    //   FP32 fetch -> BF16 fetch_cast -> ring256 -> collect -> DM
    //
    // For BF16, one 32-byte flit carries 16 elements instead of 8 FP32
    // elements, so the switch payload should require half as many flits.
    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<m![Dummy256], m![H / 480]>(
            SwitchConfig::CustomBroadcast { ring_size: 256 },
        )
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

pub(crate) fn normalize<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn normalize_and_add_residual_distributed(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![H / 120, 1 # 8], m![H % 120]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let residual: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        residual_hbm.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn scale_normalize_and_add_residual_distributed(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]>,
    output_scale_dm: &DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]>,
    rms_weight_dm: &DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]>,
    residual: &DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]>,
) -> DmTensor<bf16, Chip, Cluster, m![1 # 32, H / 480], m![H % 480]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let output_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(output_scale_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    // M27: fold the output-channel scale into both RMS passes. This removes
    // the full scaled HiddenRows temporary. It intentionally keeps the
    // projection result itself in BF16, but no longer inserts the extra
    // BF16 rounding after channel scaling before the RMS statistic.
    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &output_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let inv_rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { inv_rms.reshape() };

    let rms_weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(rms_weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &output_scale_vrf)
        .vector_fp_ternary(FpTernaryOp::FmaF, (&inv_rms_vrf, 0.0f32))
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &rms_weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn scale_normalize_and_add_residual_cluster_split(
    ctx: &mut Context,
    x: &DmTensor<
        bf16,
        Chip,
        m![Gs],
        m![1 # 16, H / 120 % 16],
        m![H % 120],
    >,
    output_weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<
    bf16,
    Chip,
    m![Gs],
    m![1 # 16, H / 120 % 16],
    m![H % 120],
> {
    type SplitClusters = m![Gs];
    type ExchangeClusters = m![Dummy2];
    type SplitSlices = m![1 # 16, H / 120 % 16];
    type PartialSlices = m![1 # 16, Dummy8, Gs];

    let output_scale_dm: DmTensor<bf16, Chip, SplitClusters, SplitSlices, m![H % 120]> =
        output_weight_scale.to_dm(&mut ctx.tdma);
    let output_scale_vrf: VrfTensor<f32, Chip, SplitClusters, SplitSlices, m![H % 120]> = ctx
        .sub
        .begin(output_scale_dm.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, SplitClusters, SplitSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &output_scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let local_mean_square: DmTensor<f32, Chip, SplitClusters, PartialSlices, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<PartialSlices, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mut remote_mean_square_exchange: DmTensor<
        f32,
        Chip,
        ExchangeClusters,
        PartialSlices,
        m![1 # 8],
    > = DmTensor::new();
    unsafe {
        local_mean_square
            .view()
            .reshape::<Chip, ExchangeClusters, PartialSlices, m![1 # 8]>()
    }
        .cluster_tile::<ExchangeClusters, 1, Padding<Identity, 2>>(1)
        .to_dm_view(
            &mut ctx.tdma,
            remote_mean_square_exchange
                .view_mut()
                .cluster_tile::<
                    ExchangeClusters,
                    1,
                    Padding<Identity, 2, { PaddingKind::Bottom }>,
                >(0),
        );
    unsafe {
        local_mean_square
            .view()
            .reshape::<Chip, ExchangeClusters, PartialSlices, m![1 # 8]>()
    }
        .cluster_tile::<ExchangeClusters, 1, Padding<Identity, 2>>(0)
        .to_dm_view(
            &mut ctx.tdma,
            remote_mean_square_exchange
                .view_mut()
                .cluster_tile::<
                    ExchangeClusters,
                    1,
                    Padding<Identity, 2, { PaddingKind::Bottom }>,
                >(1),
        );
    let remote_mean_square: DmTensor<f32, Chip, SplitClusters, PartialSlices, m![1 # 8]> =
        unsafe { remote_mean_square_exchange.reshape() };
    let remote_mean_square_vrf: VrfTensor<f32, Chip, SplitClusters, PartialSlices, m![1 # 8]> = ctx
        .sub
        .begin(remote_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let global_mean_square: DmTensor<f32, Chip, SplitClusters, PartialSlices, m![1 # 8]> = ctx
        .main
        .begin(local_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &remote_mean_square_vrf)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let inv_rms: DmTensor<f32, Chip, SplitClusters, PartialSlices, m![1 # 8]> = ctx
        .main
        .begin(global_mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0f32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, SplitClusters, SplitSlices, m![1 # 8]> =
        unsafe { inv_rms.reshape() };

    let rms_weight_dm: DmTensor<bf16, Chip, SplitClusters, SplitSlices, m![H % 120]> =
        rms_weight.to_dm(&mut ctx.tdma);
    let rms_weight_vrf: VrfTensor<f32, Chip, SplitClusters, SplitSlices, m![H % 120]> = ctx
        .sub
        .begin(rms_weight_dm.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let inv_rms_vrf: VrfTensor<f32, Chip, SplitClusters, SplitSlices, m![1 # 8]> = ctx
        .sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<bf16, Chip, SplitClusters, SplitSlices, m![H % 120]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &output_scale_vrf)
        .vector_fp_ternary(FpTernaryOp::FmaF, (&inv_rms_vrf, 0.0f32))
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &rms_weight_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let residual_dm: DmTensor<bf16, Chip, SplitClusters, SplitSlices, m![H % 120]> =
        residual_hbm.to_dm(&mut ctx.tdma);
    let residual_vrf: VrfTensor<f32, Chip, SplitClusters, SplitSlices, m![H % 120]> = ctx
        .sub
        .begin(residual_dm.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &residual_vrf)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

pub(crate) fn normalize_replicated_k1<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, m![Dummy2], m![Dummy256], m![H]> {
    type ReducingSlices = m![1 # 32, H / 480];

    let x: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();

    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let normalized: DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Duplicate the compact normalized vector through HBM (only 2 x H f32 values),
    // then run the same 32->256 slice broadcast independently on both clusters.
    let mut dup_hbm: HbmTensor<f32, Chip, m![Dummy2, H]> = HbmTensor::new();
    normalized.view().to_hbm_view(
        &mut ctx.tdma,
        dup_hbm.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, H]>(0),
    );
    normalized.view().to_hbm_view(
        &mut ctx.tdma,
        dup_hbm.view_mut().tile::<m![Dummy2], 1, m![1 #{!} 2, H]>(1),
    );
    let normalized2: DmTensor<f32, Chip, m![Dummy2], ReducingSlices, m![H % 480]> =
        dup_hbm.to_dm(&mut ctx.tdma);

    let replicated2: DmTensor<bf16, Chip, m![Dummy2], m![Dummy256], m![H]> = ctx.main
        .begin(normalized2.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<m![Dummy256], m![H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    replicated2
}
