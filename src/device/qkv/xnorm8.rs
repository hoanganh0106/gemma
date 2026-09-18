//! The input RMSNorm on genuinely replicated 240-element pieces, all-gathered into every slice.
//!
//! The hidden state is loaded as 16 pieces of 240 elements, 16 real copies of each (`Rep16` is a
//! real axis, not `1 # 16`), so all 256 slices of a cluster hold live data from the first load on.
//! Each group of 16 slices normalizes the whole vector on its own. The result is split into two
//! exact f8e4m3 terms and each is gathered along the ring with `Broadcast1`: an f8 x f8
//! contraction needs no decode table and runs ~3.5x faster on the device than f8 -> bf16.
use furiosa_opt_std::prelude::*;

use super::{Rep16, Ring16, Slot, Term};
use crate::axes::H;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn normalize_everywhere_f8<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> XTerms<Cluster> {
    // The input weight and the hidden state are two TILES of ONE DM tensor, the weight written
    // FIRST: tile writes chain in program order, so the x load is placed behind the weight load
    // instead of in front of it. Both loads sit on the DMA queue either way, but the scheduler
    // lists the whole RMSNorm chain one DMA slot later, and the issuing thread -- which walks
    // that list and stalls on every instruction whose inputs are not ready (BRIEF2 UPDATE 12) --
    // then reaches the first big weight load BEFORE the chain instead of after it.
    // x takes tile 0 so that it keeps the tensor's own alignment.
    let mut xpool: DmTensor<bf16, Chip, Cluster, m![Rep16, H / 240], m![Slot, H % 240]> = DmTensor::new();
    {
        let w = unsafe { rms_weight.view().reshape::<Chip, m![Slot = 1, H]>() };
        w.to_dm_view(&mut ctx.tdma, xpool.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, H % 240]>(1));
        let xv = unsafe { x.view().reshape::<Chip, m![Slot = 1, H]>() };
        xv.to_dm_view(&mut ctx.tdma, xpool.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, H % 240]>(0));
    }
    let mean_square: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![1 # 8]> = ctx
        .main
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 240]>(0))
        .fetch::<m![Slot = 1, H / 16 % 15], m![H % 16]>()
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

    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .sub
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 240]>(1))
        .fetch::<m![Slot = 1, H / 16 % 15], m![H % 16]>()
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
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 240]>(0))
        .fetch::<m![Slot = 1, H / 16 % 15], m![H % 16]>()
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

    let mut terms: XTerms<Cluster> = DmTensor::new();
    gather_f8(ctx, &normalized, &mut terms, 0);
    gather_f8(ctx, &residual, &mut terms, 1);
    terms
}

/// Both f8 terms of the normalized hidden state, in every slice.
pub(crate) type XTerms<Cluster> = DmTensor<f8e4m3, Chip, Cluster, m![Rep16, Ring16], m![Term, H]>;

/// All-gathers f32 pieces along each 16-slice ring and leaves them as f8e4m3 in every slice, as
/// term `index` of `terms`. (An interleaved pair of DM tensors does NOT make a two-lane TRF: on
/// the device only the first tensor arrived. One DM tensor with a real `Term` axis does.)
fn gather_f8<Cluster: M>(
    ctx: &mut Context,
    pieces: &DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]>,
    terms: &mut XTerms<Cluster>,
    index: usize,
) {
    let pieces: DmTensorView<'_, f32, Chip, Cluster, m![Rep16, H / 240], m![Term = 1, H % 240]> =
        unsafe { pieces.view().reshape() };
    ctx.main
        .begin(pieces)
        .fetch::<m![Term = 1], m![H % 240]>()
        .switch::<m![Rep16, Ring16], m![Term = 1, H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![Term = 1, H / 8], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit_view(terms.view_mut().tile::<m![Term], 1, m![Term = 1 #{!} 2, H]>(index));
}
