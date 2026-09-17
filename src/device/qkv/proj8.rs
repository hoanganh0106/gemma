//! f8 x f8 variant of `proj` (no decode table).
//! Q/K/V projections with WHOLE weight rows per slice (one contiguous HBM run per slice, no
//! inter-slice reduce), gathered per KV head on the Switch network. The heads stay on the
//! cluster that projected them: no DMA gather, no HBM round trip.
use furiosa_opt_std::prelude::*;

use super::{HeadClusters, HeadSlices, KvRowsByHead, QueryRowsByHead};
use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{KeyValueClusters, QueryClusters};

use super::proj::{KeyValueRowSlices, QueryRowSlices};

pub(crate) fn project_query(
    ctx: &mut Context,
    x1: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
    x2: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Gs, Ds]> {
    let x1_trf = query_operand(ctx, x1);
    let x2_trf = query_operand(ctx, x2);

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let term1 = contract_query(ctx, &weight_f8, &x1_trf);
    let term2 = contract_query(ctx, &weight_f8, &x2_trf);

    // Relabel only: Qs = (Ns, Gs, Ds) row-major, so cluster Qs/2048 is Ns/4, slice Qs/8%256 is
    // (Ns%4, Gs, Ds/8) and the in-slice Qs%8 is Ds%8. Same physical order.
    // term1 + term2 on the row slices, term2 riding in the VRF. (Feeding both terms through
    // begin_interleaved + Switch gather + unzip/zip in ONE pass lowers but gave wrong Q on the device.)
    let term2_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRowSlices, m![Qs % 8]> = ctx
        .sub
        .begin(term2.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .to_vrf();
    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![Qs % 8]> = ctx
        .main
        .begin(term1.view())
        .fetch::<m![1], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 2], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &term2_vrf)
        .vector_widen_concat::<m![1], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
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
    x1_trf: &KvOperand,
    x2_trf: &KvOperand,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4, H]> =
        weight.to_dm(&mut ctx.tdma);

    let term1 = contract_key_value(ctx, &weight_f8, x1_trf);
    let term2 = contract_key_value(ctx, &weight_f8, x2_trf);

    // Relabel only: Ps = (Ns, Ds) row-major: cluster Ns/4, slice (Ns%4, Ds/4), in-slice Ds%4.
    // term1 + term2 on the row slices. Four values per slice cannot take the interleave/zip pair
    // path (it has no narrow_trim), so term2 rides in the VRF.
    let term2_vrf: VrfTensor<f32, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4 # 8]> = ctx
        .sub
        .begin(term2.view())
        .fetch::<m![1], m![Ps % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .to_vrf();
    let contraction: DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4]> = ctx
        .main
        .begin(term1.view())
        .fetch::<m![1], m![Ps % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ps % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &term2_vrf)
        .vector_widen_pad::<m![Ps % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 4 # 8 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();
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
    x1: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
    x2: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
) {
    let x1_trf = key_value_operand(ctx, x1);
    let x2_trf = key_value_operand(ctx, x2);

    let k = project_one_kv_matrix(ctx, &x1_trf, &x2_trf, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(ctx, &x1_trf, &x2_trf, v_weight, v_weight_scale);

    (k, v)
}

type QueryOperand = TrfTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![1], m![H]>;
type KvOperand = TrfTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![1], m![H]>;

fn query_operand(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
) -> QueryOperand {
    ctx.sub
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 32], m![H % 32]>()
        .to_trf()
}

fn key_value_operand(
    ctx: &mut Context,
    x: &DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![H]>,
) -> KvOperand {
    let x: DmTensorView<'_, f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![H]> = unsafe { x.view().reshape() };
    ctx.sub.begin(x).fetch::<m![1], m![H]>().collect::<m![H / 32], m![H % 32]>().to_trf()
}

/// One f8 x f8 term of the Q projection: 8 whole rows per slice against the whole of H.
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
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
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
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit()
}
