//! The input RMSNorm on replicated pieces (see `xnorm`), all-gathered into every slice as
//! f8e4m3: an f8 x f8 contraction needs no decode table and runs ~3.5x faster on the device.
use furiosa_opt_std::prelude::*;

use super::{Rep16, Ring16};
use crate::axes::H;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn normalize_everywhere_f8<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (
    DmTensor<f8e4m3, Chip, Cluster, m![Rep16, Ring16], m![H]>,
    DmTensor<f8e4m3, Chip, Cluster, m![Rep16, Ring16], m![H]>,
) {
    let x: DmTensor<bf16, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced: DmTensor<f32, Chip, Cluster, m![Rep16, Ring16], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Rep16, Ring16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, m![Rep16, Ring16], m![1 # 8]> = ctx
        .main
        .begin(reduced.view())
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
    let rms: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), super::X_PRESCALE)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Exact two-term f8 expansion: q1 = f8(x), q2 = f8(x - q1). Both residual steps are exact
    // in f32 and |x - q1 - q2| <= |x| / 256, the order of bf16's own rounding. The projection
    // is linear, so W*x = W*q1 + W*q2.
    let q1_pieces: DmTensor<f8e4m3, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let q1_vrf: VrfTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .sub
        .begin(q1_pieces.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    let residual: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 30], m![H % 8]>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .commit_trim::<m![H % 8]>()
        .commit();

    let q1 = gather_f8(ctx, &normalized);
    let q2 = gather_f8(ctx, &residual);
    (q1, q2)
}

/// All-gathers f32 pieces along each 16-slice ring and leaves them as f8e4m3 in every slice.
fn gather_f8<Cluster: M>(
    ctx: &mut Context,
    pieces: &DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]>,
) -> DmTensor<f8e4m3, Chip, Cluster, m![Rep16, Ring16], m![H]> {
    ctx.main
        .begin(pieces.view())
        .fetch::<m![1], m![H % 240]>()
        .switch::<m![Rep16, Ring16], m![H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
