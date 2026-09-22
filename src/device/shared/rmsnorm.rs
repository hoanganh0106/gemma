
use furiosa_opt_std::prelude::*;

use crate::axes::{Dummy8, H};
use crate::device::layout::QkvColumns;

use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

/// The hidden state divided over eight slices, 480 columns each, which is how the sum of
/// squares is taken. Both `normalize` and `normalize_columns` are this plus a final layout.
pub(crate) type ReducingSlices = m![1 # 32, H / 480];

/// `x / rms(x) * rms_weight`, in f32 and still split eight ways across slices.
fn normalized_f32<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> {

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

    normalized
}

/// The whole normalized vector in one slice, as the rest of the model expects it.
pub(crate) fn normalize<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let normalized = normalized_f32(ctx, x, rms_weight);

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<Slice, m![H / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// The same normalization, left divided over the eight slices that took the sum of squares.
/// Whatever runs next then runs 480 elements wide on eight slices instead of 3840 on one.
pub(crate) fn normalize_spread<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let normalized = normalized_f32(ctx, x, rms_weight);

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

/// Each of the eight 480-column pieces replicated across the 32 row groups that contract it.
/// Eight times less copying than gathering the pieces and broadcasting the whole vector.
pub(crate) fn normalize_columns<Cluster: M, Slice: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, QkvColumns, m![H % 480]> {
    let normalized = normalized_f32(ctx, x, rms_weight);

    ctx.main
        .begin(normalized.view())
        .fetch::<m![1], m![H % 480]>()
        .switch::<QkvColumns, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
