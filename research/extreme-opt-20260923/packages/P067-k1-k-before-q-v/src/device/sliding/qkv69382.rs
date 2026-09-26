use furiosa_opt_std::prelude::*;

use crate::axes::Qs;
use crate::axes::{Ds, E, Gs, H, Ns, Ps};
axes![Dummy256 = 256];
use crate::device::layout::Replicated;
use crate::{Chip, EPS};
type QkvDualQueryCluster = m![Qs / 2048 % 2];

// Keep complete heads local through normalization, RoPE and the final cache write.
pub(crate) type HeadCluster = m![Ns / 4];
pub(crate) type QueryHeads = m![Ns % 4, Gs, 1 # 32];
pub(crate) type KvHeads = m![Ns % 4, 1 # 64];
pub(crate) type QueryHeadTensor = DmTensor<bf16, Chip, HeadCluster, QueryHeads, m![Ds]>;
pub(crate) type KvHeadTensor = DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]>;
type KvCluster = m![Ps / 1024];
type KvRows = m![Ps / 4 % 256];

type NativeSlices = m![Dummy256];
pub(crate) type NativeTrf =
    TrfTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![1], m![H]>;
pub(crate) type NativeScale = VrfTensor<f32, Chip, QkvDualQueryCluster, NativeSlices, m![1 # 8]>;

pub(crate) fn project_query(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    weight: &HbmTensor<f8e4m3, Chip, m![Qs, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Qs]>,
) -> QueryHeadTensor {
    type QueryStripe = m![Ns % 4, Gs, Ds / 64, Ds % 8];

    // Spread adjacent Q rows across the row ring so TDMA sees longer HBM runs.
    // The hidden-state operand is replicated on every native slice, so only
    // its logical slice labels change here.
    let weight_view: HbmTensorView<'_, f8e4m3, Chip, m![Ns, Gs, Ds, H]> =
        unsafe { weight.view().reshape() };
    let weight_dm: DmTensor<f8e4m3, Chip, HeadCluster, QueryStripe, m![Ds / 8 % 8, H]> =
        weight_view.to_dm(&mut ctx.tdma);
    let x_rows: &TrfTensor<f8e4m3, Chip, HeadCluster, QueryStripe, m![1], m![H]> =
        unsafe { std::mem::transmute(x_trf) };

    let sums: DmTensor<f32, Chip, HeadCluster, QueryStripe, m![Ds / 8 % 8, 1 # 8]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<m![Ds / 8 % 8, H / 32], m![H % 32]>()
        .collect::<m![Ds / 8 % 8, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 8 % 8, H / 32], m![H % 32], _, _, _>(x_rows)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 8 % 8]>()
        .contract_lane::<m![Ds / 8 % 8], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let sorted: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Gs, Ds / 64, Ds / 8 % 8], m![Ds % 8]> =
        ctx.main
            .begin(sums.view())
            .fetch::<m![Ds / 8 % 8], m![1 # 8]>()
            .switch::<m![Ns % 4, Gs, Ds / 64, Ds / 8 % 8], m![Ds % 8]>(
                SwitchConfig::InterTranspose {
                    slice1: 8,
                    slice0: 1,
                    time0: 1,
                },
            )
            .collect::<m![Ds % 8], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![Ds / 4 % 2], m![Ds % 4 # 16]>()
            .commit_trim::<m![Ds % 4]>()
            .commit();

    let rows: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Gs, Ds / 8], m![Ds % 8]> =
        unsafe { sorted.reshape() };

    let scale_view: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let scale_dm: QueryHeadTensor = scale_view.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, QueryHeads, m![Ds]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    ctx.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .switch::<QueryHeads, m![Ds / 8]>(SwitchConfig::Broadcast1 {
            slice1: 32,
            slice0: 1,
        })
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn copy_query(ctx: &mut Device, x: &QueryHeadTensor) -> QueryHeadTensor {
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .collect::<m![Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit()
}

fn project_one_kv_matrix_scaled(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvHeadTensor {
    // Stripe consecutive KV rows across the 8-way row ring. This preserves
    // head ownership but gives TDMA long contiguous HBM row runs instead of
    // four rows parked on one slice.
    type KvStripe = m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2];
    let weight_view: HbmTensorView<'_, f8e4m3, Chip, m![Ns, Ds, H]> =
        unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, HeadCluster, KvStripe, m![Ds / 8 % 4, H]> =
        weight_view.to_dm(&mut ctx.tdma);
    let x_trf: &TrfTensor<f8e4m3, Chip, HeadCluster, KvStripe, m![1], m![H]> =
        unsafe { std::mem::transmute(x_trf) };
    let sums: DmTensor<f32, Chip, HeadCluster, KvStripe, m![Ds / 8 % 4, 1 # 8]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![Ds / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ds / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 8 % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 8 % 4]>()
        .contract_lane::<m![Ds / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sorted: DmTensor<
        bf16,
        Chip,
        HeadCluster,
        m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2],
        m![Ds % 4],
    > = ctx
        .main
        .begin(sums.view())
        .fetch::<m![Ds / 8 % 4], m![1 # 8]>()
        .switch::<m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2], m![Ds % 4]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 2,
                time0: 1,
            },
        )
        .collect::<m![Ds % 4], m![1 # 8]>()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    let rows: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { sorted.reshape() };
    // Apply the row scale while the 64-slice gather assembles the complete head.
    // This removes the repeated projection-scale multiply from the later RMS passes.
    let scale_view: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let scale_dm: KvHeadTensor = scale_view.to_dm(&mut ctx.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![Ds]> = ctx
        .sub
        .begin(scale_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    ctx.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<KvHeads, m![Ds / 4]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit()
}

fn project_one_kv_matrix_raw(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> KvHeadTensor {
    // Stripe consecutive KV rows across the 8-way row ring. This preserves
    // head ownership but gives TDMA long contiguous HBM row runs instead of
    // four rows parked on one slice.
    type KvStripe = m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2];
    let weight_view: HbmTensorView<'_, f8e4m3, Chip, m![Ns, Ds, H]> =
        unsafe { weight.view().reshape() };
    let weight: DmTensor<f8e4m3, Chip, HeadCluster, KvStripe, m![Ds / 8 % 4, H]> =
        weight_view.to_dm(&mut ctx.tdma);
    let x_trf: &TrfTensor<f8e4m3, Chip, HeadCluster, KvStripe, m![1], m![H]> =
        unsafe { std::mem::transmute(x_trf) };
    let sums: DmTensor<f32, Chip, HeadCluster, KvStripe, m![Ds / 8 % 4, 1 # 8]> = ctx
        .main
        .begin(weight.view())
        .fetch::<m![Ds / 8 % 4, H / 32], m![H % 32]>()
        .collect::<m![Ds / 8 % 4, H / 32], m![H % 32]>()
        .contract_outer::<m![Ds / 8 % 4, H / 32], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Ds / 8 % 4]>()
        .contract_lane::<m![Ds / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let sorted: DmTensor<
        bf16,
        Chip,
        HeadCluster,
        m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2],
        m![Ds % 4],
    > = ctx
        .main
        .begin(sums.view())
        .fetch::<m![Ds / 8 % 4], m![1 # 8]>()
        .switch::<m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2], m![Ds % 4]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 2,
                time0: 1,
            },
        )
        .collect::<m![Ds % 4], m![1 # 8]>()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![1], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    let rows: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4]> =
        unsafe { sorted.reshape() };
    // Each 64-slice group owns one key/value head.
    let heads: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Dummy256 % 64], m![Ds]> = ctx
        .main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .switch::<m![Ns % 4, Dummy256 % 64], m![Ds / 4]>(SwitchConfig::Broadcast1 {
            slice1: 64,
            slice0: 1,
        })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    // Only hide already initialized replicas; padding never creates data.
    let heads: KvHeadTensor = unsafe { heads.reshape() };
    heads
}

fn apply_projection_scale_kv(
    ctx: &mut Device,
    x: &KvHeadTensor,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvHeadTensor {
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_scale: DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]> =
        weight_scale.to_dm(&mut ctx.tdma);
    scale_head(ctx, x, &weight_scale)
}

pub(crate) fn project_key_value(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    k_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> (KvHeadTensor, KvHeadTensor) {
    let k = project_one_kv_matrix_raw(ctx, x_trf, k_weight);
    let v_raw = project_one_kv_matrix_raw(ctx, x_trf, v_weight);
    let v = normalize_value_project_scaled(ctx, &v_raw, v_weight_scale);
    (k, v)
}

pub(crate) fn project_key(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    k_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
) -> KvHeadTensor {
    project_one_kv_matrix_raw(ctx, x_trf, k_weight)
}

pub(crate) fn project_value(
    ctx: &mut Device,
    x_trf: &NativeTrf,
    v_weight: &HbmTensor<f8e4m3, Chip, m![Ps, H]>,
    v_weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvHeadTensor {
    let v_raw = project_one_kv_matrix_raw(ctx, x_trf, v_weight);
    normalize_value_project_scaled(ctx, &v_raw, v_weight_scale)
}

fn root_mean_square<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> VrfTensor<f32, Chip, HeadCluster, S, m![1 # 8]> {
    let mean_square: DmTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    ctx
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
        .to_vrf(&mut ctx.sub)
}

fn normalize_weighted_project_scaled<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    proj_scale: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    norm_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let proj_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(proj_scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let norm_dm: DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> = norm_weight.to_dm(&mut ctx.tdma);
    let norm_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(norm_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &proj_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = ctx
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
        .to_vrf(&mut ctx.sub);
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &proj_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &norm_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_query_project_scaled(
    ctx: &mut Device,
    x: &QueryHeadTensor,
    proj_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    norm_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> QueryHeadTensor {
    let v: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> = unsafe { proj_scale.view().reshape() };
    let dm: DmTensor<bf16, Chip, HeadCluster, QueryHeads, m![Ds]> = v.to_dm(&mut ctx.tdma);
    normalize_weighted_project_scaled(ctx, x, &dm, norm_weight)
}

pub(crate) fn normalize_key_project_scaled(
    ctx: &mut Device,
    x: &KvHeadTensor,
    proj_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    norm_weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> KvHeadTensor {
    let v: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> = unsafe { proj_scale.view().reshape() };
    let dm: DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]> = v.to_dm(&mut ctx.tdma);
    normalize_weighted_project_scaled(ctx, x, &dm, norm_weight)
}

pub(crate) fn normalize_weighted<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    weight: &HbmTensor<bf16, Chip, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let rms_vrf = root_mean_square(ctx, x);
    let weight: DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> = weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn normalize_value_project_scaled(
    ctx: &mut Device,
    x: &KvHeadTensor,
    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,
) -> KvHeadTensor {
    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_dm: DmTensor<bf16, Chip, HeadCluster, KvHeads, m![Ds]> =
        weight_scale.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![Ds]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let mean_square: DmTensor<f32, Chip, HeadCluster, KvHeads, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![1 # 8]> = ctx
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
        .to_vrf(&mut ctx.sub);
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_value(ctx: &mut Device, x: &KvHeadTensor) -> KvHeadTensor {
    let rms_vrf = root_mean_square(ctx, x);
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn rope_one<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    cos: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    sin: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    type Half = m![Ds = 128];
    let cos_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(cos.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let s0_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(sin.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let s1_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(sin.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    // Preserve baseline half-swap and Mul1 sine path, but materialize the
    // sine product directly. This removes the padded bf16 `rotated` buffer.
    let mut sin_product: DmTensor<f32, Chip, HeadCluster, S, m![Ds]> = DmTensor::new();
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &s0_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(
            sin_product
                .view_mut()
                .tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0),
        );
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], m![Ds = 128]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &s1_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(
            sin_product
                .view_mut()
                .tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128),
        );

    let sin_product_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(sin_product.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &sin_product_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

fn rope_one_pair_fixed<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    pair: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    type Half = m![Ds = 128];

    let cos_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(pair.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let sin_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(pair.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let mut sin_product: DmTensor<f32, Chip, HeadCluster, S, m![Ds]> = DmTensor::new();
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(
            sin_product
                .view_mut()
                .tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0),
        );
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), -1.0f32)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_final()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(
            sin_product
                .view_mut()
                .tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128),
        );

    let sin0_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(
            sin_product
                .view()
                .tile::<m![Ds], 128, m![Ds = 128 # 256]>(0),
        )
        .fetch::<m![1], Half>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();
    let sin1_vrf: VrfTensor<f32, Chip, HeadCluster, S, Half> = ctx
        .sub
        .begin(
            sin_product
                .view()
                .tile::<m![Ds], 128, m![Ds = 128 # 256]>(128),
        )
        .fetch::<m![1], Half>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .to_vrf();

    let mut out: DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> = DmTensor::new();
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &sin0_vrf)
        .vector_final()
        .cast::<bf16, m![Ds = 128 % 8 # 16]>()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(out.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));
    ctx.main
        .begin(x.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128))
        .fetch::<m![1], Half>()
        .fetch_cast::<f32>()
        .collect::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds = 128 / 4], m![Ds = 128 % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds = 128 / 8], m![Ds = 128 % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &sin1_vrf)
        .vector_final()
        .cast::<bf16, m![Ds = 128 % 8 # 16]>()
        .commit_trim::<m![Ds = 128 % 8]>()
        .commit_view(
            out.view_mut()
                .tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128),
        );
    out
}

pub(crate) fn apply_rope(
    ctx: &mut Device,
    q: &QueryHeadTensor,
    k: &KvHeadTensor,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> (QueryHeadTensor, KvHeadTensor) {
    let cos_row: DmTensor<bf16, Chip, m![2], m![8, 1 # 32], m![Ds]> =
        cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, m![2], m![8, 1 # 32], m![Ds]> =
        sin.dma_gather_scaled(rope_offset);
    let q_cos: QueryHeadTensor = unsafe { cos_row.reshape() };
    let q_sin: QueryHeadTensor = unsafe { sin_row.reshape() };
    let q = rope_one(ctx, q, &q_cos, &q_sin);

    let k_cos: KvHeadTensor = unsafe { q_cos.reshape() };
    let k_sin: KvHeadTensor = unsafe { q_sin.reshape() };
    let k = rope_one(ctx, k, &k_cos, &k_sin);
    (q, k)
}

fn scale_head<S: M>(
    ctx: &mut Device,
    x: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
    weight: &DmTensor<bf16, Chip, HeadCluster, S, m![Ds]>,
) -> DmTensor<bf16, Chip, HeadCluster, S, m![Ds]> {
    let weight_vrf: VrfTensor<f32, Chip, HeadCluster, S, m![Ds]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}

pub(crate) fn normalize_native_input(
    ctx: &mut Device,
    x: &HbmTensor<bf16, Chip, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![H]>,
) -> NativeTrf {
    type Shards = m![Dummy256 / 32, H / 120];
    type Copies = m![Dummy256 / 32, Dummy256 % 32];

    // Compute the same weighted RMSNorm directly in the 8x32 shard layout used by
    // the fast QKV path. This keeps input_rms_weight semantically live and removes
    // the temporary normalized HBM store/load used by P051.
    let x: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> =
        x.to_dm(&mut ctx.tdma);
    let weight: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> =
        weight.to_dm(&mut ctx.tdma);

    let partial: DmTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { H::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let mean: DmTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Copies, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    let rms: VrfTensor<f32, Chip, QkvDualQueryCluster, Copies, m![1 # 8]> = ctx
        .main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut ctx.sub);
    let rms_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> =
        unsafe { rms.reshape() };
    let weight_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let normalized: DmTensor<bf16, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Preserve the proven QKV69382 one-term FP8 operand and all projection/layout code.
    let quantized: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, Shards, m![H % 120]> = ctx
        .main
        .begin(normalized.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, -0.5)
        .vector_final()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    let shared: DmTensor<
        f8e4m3,
        Chip,
        QkvDualQueryCluster,
        Copies,
        m![H / 8 % 15, H / 120, H % 8],
    > = ctx
        .main
        .begin(quantized.view())
        .fetch::<m![H / 8 % 15], m![H % 8]>()
        .switch::<Copies, m![H / 8 % 15, H / 120]>(SwitchConfig::Broadcast1 {
            slice1: 32,
            slice0: 1,
        })
        .collect::<m![H / 8 % 15, H / 120], m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    let shared: DmTensor<
        f8e4m3,
        Chip,
        QkvDualQueryCluster,
        NativeSlices,
        m![H / 8 % 15, H / 120, H % 8],
    > = unsafe { shared.reshape() };
    let dense: DmTensor<f8e4m3, Chip, QkvDualQueryCluster, NativeSlices, m![H]> = ctx
        .main
        .begin(shared.view())
        .fetch::<m![H / 8], m![H % 8]>()
        .collect::<m![H / 8], m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    ctx.sub
        .begin(dense.view())
        .fetch::<m![H / 32], m![H % 32]>()
        .collect::<m![H / 32], m![H % 32]>()
        .to_trf()
}
