//! Q/K/V projections with WHOLE weight rows per slice (one contiguous HBM run per slice, no
//! inter-slice reduce), gathered per KV head on the Switch network. The heads stay on the
//! cluster that projected them: no DMA gather, no HBM round trip.
use furiosa_opt_std::prelude::*;

use super::{HeadClusters, HeadSlices, KvRowsByHead, QueryRowsByHead};
use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{KeyValueClusters, QueryClusters};

pub(crate) type QueryRowSlices = m![Qs / 8 % 256];
pub(crate) type KeyValueRowSlices = m![Ps / 4 % 256];

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Gs, Ds]> {
    let x_trf: TrfTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![1], m![H]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QueryRowSlices, m![Qs % 8, H]> = weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![Qs % 8]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 8, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 8, H / 16], m![H % 16]>()
        .contract_outer::<m![Qs % 8, H / 32], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 8]>()
        .contract_lane::<m![Qs % 8], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 2], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    // Relabel only: Qs = (Ns, Gs, Ds) row-major, so cluster Qs/2048 is Ns/4, slice Qs/8%256 is
    // (Ns%4, Gs, Ds/8) and the in-slice Qs%8 is Ds%8. Same physical order.
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
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![1], m![H]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4, H]> =
        weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![Ps % 4]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 4, H / 32], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 4, H / 16], m![H % 16]>()
        .contract_outer::<m![Ps % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 4]>()
        .contract_lane::<m![Ps % 4], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    // Relabel only: Ps = (Ns, Ds) row-major: cluster Ns/4, slice (Ns%4, Ds/4), in-slice Ds%4.
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
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QueryClusters, QueryRowSlices, m![H]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
    DmTensor<bf16, Chip, HeadClusters, HeadSlices, m![Ds]>,
) {
    let x: DmTensorView<'_, bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![H]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KeyValueClusters, KeyValueRowSlices, m![1], m![H]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H]>()
        .collect::<m![H / 16], m![H % 16]>()
        .to_trf();

    let k = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (k, v)
}
