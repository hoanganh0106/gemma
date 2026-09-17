
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{Ds, Gs, H, Ns, Ps, Qs};
use crate::device::layout::{
    Cluster, KeyValueClusters, KeyValueRowsReduced, KvColumns, OutputClusters, QkvColumns, QueryClusters,
    QueryRowsReduced, Slice, SlidingOutputColumns, SlidingOutputRows,
};

pub(crate) fn project_query(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QueryClusters, QkvColumns, m![H % 960]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> HbmTensor<bf16, Chip, m![Qs]> {
    let x_trf: TrfTensor<bf16, Chip, QueryClusters, QkvColumns, m![1], m![H % 960]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H % 960]>()
        .collect::<m![H / 16 % 60], m![H % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, QueryClusters, QkvColumns, m![Qs % 32, H % 960]> =
        weight.to_dm(&mut ctx.tdma);

    // 32 output rows by 960 contracted columns per slice; the four column groups of a row
    // group add up in the inter-slice reduce, still in f32.
    let contraction: DmTensor<bf16, Chip, QueryClusters, QueryRowsReduced, m![Qs % 32]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Qs % 32, H / 32 % 30], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Qs % 32, H / 16 % 60], m![H % 16]>()
        .contract_outer::<m![Qs % 32, H / 32 % 30], m![H % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Qs % 32]>()
        .contract_lane::<m![Qs % 32], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<QueryRowsReduced, m![Qs % 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Qs / 4 % 8], m![Qs % 4 # 16]>()
        .commit_trim::<m![Qs % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, QueryClusters, QueryRowsReduced, m![Qs % 32]> =
        weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, QueryClusters, QueryRowsReduced, m![Qs % 32]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 4], m![Qs % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, QueryClusters, QueryRowsReduced, m![Qs % 32]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Qs % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 4], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 8], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Qs / 8 % 4], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();

    // Gather inside the cluster first: collecting 64 slices straight into HBM is a scatter the
    // DMA engine runs at almost no utilization. HBM is chip-wide, so the two clusters' row
    // halves become one vector there.
    let gathered: DmTensor<bf16, Chip, QueryClusters, Slice, m![Qs % 2048]> = scaled.to_dm(&mut ctx.tdma);
    gathered.to_hbm(&mut ctx.tdma)
}

fn project_one_kv_matrix(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, KeyValueClusters, KvColumns, m![1], m![H % 960]>,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> HbmTensor<bf16, Chip, m![Ps]> {
    let weight_f8: DmTensor<f8e4m3, Chip, KeyValueClusters, KvColumns, m![Ps % 16, H % 960]> =
        weight.to_dm(&mut ctx.tdma);

    let contraction: DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowsReduced, m![Ps % 16]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![Ps % 16, H / 32 % 30], m![H % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![Ps % 16, H / 16 % 60], m![H % 16]>()
        .contract_outer::<m![Ps % 16, H / 32 % 30], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ps % 16]>()
        .contract_lane::<m![Ps % 16], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<KeyValueRowsReduced, m![Ps % 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![Ps / 4 % 4], m![Ps % 4 # 16]>()
        .commit_trim::<m![Ps % 4]>()
        .commit();

    let weight_scale: DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowsReduced, m![Ps % 16]> =
        weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, KeyValueClusters, KeyValueRowsReduced, m![Ps % 16]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 2], m![Ps % 8]>()
        .to_vrf();

    let scaled: DmTensor<bf16, Chip, KeyValueClusters, KeyValueRowsReduced, m![Ps % 16]> = ctx
        .main
        .begin(contraction.view())
        .fetch::<m![1], m![Ps % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ps / 8 % 2], m![Ps % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ps / 4 % 4], m![Ps % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![Ps / 8 % 2], m![Ps % 8]>()
        .vector_final()
        .cast::<bf16, m![Ps % 8 # 16]>()
        .commit_trim::<m![Ps % 8]>()
        .commit();

    let gathered: DmTensor<bf16, Chip, KeyValueClusters, Slice, m![Ps % 1024]> = scaled.to_dm(&mut ctx.tdma);
    gathered.to_hbm(&mut ctx.tdma)
}

pub(crate) fn project_key_value(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, QueryClusters, QkvColumns, m![H % 960]>,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (HbmTensor<bf16, Chip, m![Ps]>, HbmTensor<bf16, Chip, m![Ps]>) {
    // Same 2 clusters and 256 slices, named by the K/V row count instead of Q's.
    let x: DmTensorView<'_, bf16, Chip, KeyValueClusters, KvColumns, m![H % 960]> = unsafe { x.view().reshape() };
    let x_trf: TrfTensor<bf16, Chip, KeyValueClusters, KvColumns, m![1], m![H % 960]> = ctx
        .sub
        .begin(x)
        .fetch::<m![1], m![H % 960]>()
        .collect::<m![H / 16 % 60], m![H % 16]>()
        .to_trf();

    let k = project_one_kv_matrix(ctx, &x_trf, k_weight, k_weight_scale);
    let v = project_one_kv_matrix(ctx, &x_trf, v_weight, v_weight_scale);

    (k, v)
}

pub(crate) fn project_output(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![1], m![Qs % 256]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf();

    let weight_f8: DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]> =
        weight.to_dm(&mut ctx.tdma);

    // Slice (row group, column group) holds 120 of this cluster's output rows by 256 of the
    // 4096 contracted columns. The f8 -> bf16 decode rides the fetch, and the sixteen column
    // partials meet in the inter-slice reduce while still f32.
    let contraction: DmTensor<bf16, Chip, OutputClusters, SlidingOutputRows, m![H % 120]> = ctx
        .main
        .begin(weight_f8.view())
        .fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()
        .fetch_table_lookup::<bf16>()
        .collect::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .contract_outer::<m![H % 120, Qs / 32 % 8], m![Qs % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<SlidingOutputRows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H / 4 % 30], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    let result = apply_output_channel_scale(ctx, &contraction, weight_scale);
    // HBM is chip-wide, so this is where the two clusters' halves become one vector again.
    result.to_hbm(&mut ctx.tdma)
}

fn apply_output_channel_scale<Cluster: M, Rows: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Rows, m![H % 120]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Rows, m![H % 120]> {
    let weight_scale: DmTensor<bf16, Chip, Cluster, Rows, m![H % 120]> = weight_scale.to_dm(&mut ctx.tdma);
    let weight_scale_vrf: VrfTensor<f32, Chip, Cluster, Rows, m![H % 120]> = ctx
        .sub
        .begin(weight_scale.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_scale_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
