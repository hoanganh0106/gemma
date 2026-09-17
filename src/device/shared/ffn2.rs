//! Feed-forward with the block dequantization folded into the contraction.
//!
//! The activation is split into exact f8 terms, x = q1 + q2 (q1 = f8(x), q2 = f8(x - q1)), which
//! ride two lanes of the TRF. A weight tile streams as f4 -> table -> f8 and is contracted against
//! both terms sixteen columns at a time, so what leaves the Contraction Engine is one partial sum
//! per (row, block, term). The block scale then multiplies 1/16th of the elements in the Vector
//! Engine, which also sums the blocks of a row and the column groups of a row group. No decoded
//! weight ever reaches DM.

use furiosa_opt_std::prelude::*;

use super::mlp::{
    DownClusters, DownRows, DownRowsByColumns, UpGateClusters, UpGateColumns, UpGateRows, geglu,
};
use crate::Chip;
use crate::axes::{H, L};

axes![Xc = 1920, T2 = 2, Z = 3840];
use crate::device::layout::{Cluster, Slice};

/// x -> a two-lane TRF of exact f8 terms: lane 0 = f8(g x), lane 1 = f8(g x - lane 0), g a power of two.
/// The lanes and the columns share the axis name Z, so one intra-slice reduce folds both.
macro_rules! quantize {
    ($ctx:ident, $x:ident, $cl:ty, $sl:ty, $gain:literal) => {{
        let x: DmTensor<bf16, Chip, $cl, $sl, m![T2 = 1, Xc]> = unsafe { $x.reshape() };
        let mut q: DmTensor<f8e4m3, Chip, $cl, $sl, m![T2, Xc]> = DmTensor::new();
        $ctx.main
            .begin(x.view())
            .fetch::<m![T2 = 1, Xc / 16], m![Xc % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![T2 = 1, Xc / 8], m![Xc % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![T2 = 1, Xc / 4], m![Xc % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $gain)
            .vector_widen_concat::<m![T2 = 1, Xc / 8], m![Xc % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Xc % 8 # 32]>()
            .commit_trim::<m![Xc % 8]>()
            .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xc]>(0));
        let q1_vrf: VrfTensor<f32, Chip, $cl, $sl, m![T2 = 1, Xc]> = $ctx
            .sub
            .begin(q.view().tile::<m![T2], 1, m![T2 = 1 # 2, Xc]>(0))
            .fetch::<m![T2 = 1, Xc / 32], m![Xc % 32]>()
            .fetch_cast::<f32>()
            .collect::<m![T2 = 1, Xc / 8], m![Xc % 8]>()
            .to_vrf();
        $ctx.main
            .begin(x.view())
            .fetch::<m![T2 = 1, Xc / 16], m![Xc % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![T2 = 1, Xc / 8], m![Xc % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![T2 = 1, Xc / 4], m![Xc % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), $gain)
            .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
            .vector_widen_concat::<m![T2 = 1, Xc / 8], m![Xc % 8]>()
            .vector_final()
            .cast::<f8e4m3, m![Xc % 8 # 32]>()
            .commit_trim::<m![Xc % 8]>()
            .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xc]>(1));
        let q: DmTensor<f8e4m3, Chip, $cl, $sl, m![Z]> = unsafe { q.reshape() };
        let trf: TrfTensor<f8e4m3, Chip, $cl, $sl, m![Z / 1920], m![Z % 1920]> = $ctx
            .sub
            .begin(q.view())
            .fetch::<m![Z / 32], m![Z % 32]>()
            .collect::<m![Z / 32], m![Z % 32]>()
            .to_trf();
        trf
    }};
}

/// One 12-row tile: scales to the VRF on the sub context, the fused pass on the main context.
macro_rules! fused_tile {
    ($ctx:ident, $packed:ident, $scale:ident, $x_trf:ident, $out:ident, $start:literal,
     $cl:ty, $sl:ty, $slr:ty, $row:ident, $gain:literal) => {{
        let scale_vrf: VrfTensor<f32, Chip, $cl, $sl, m![$row % 60 = 12, Z / 16 % 120]> = $ctx
            .sub
            .begin($scale.view().tile::<m![$row % 60], 12, m![$row % 60 = 12 # 60, Z / 16 % 120]>($start))
            .fetch::<m![$row % 60 = 12], m![Z / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![$row % 60 = 12, Z / 16 / 8 % 15], m![Z / 16 % 8]>()
            .to_vrf();
        $ctx.main
            .begin($packed.view())
            .fetch::<m![$row % 60 = 12], m![Z % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![$row % 60 = 12, Z / 32 % 60], m![Z % 32]>()
            .contract_outer::<m![$row % 60 = 12, Z / 64 % 30], m![Z % 64], _, _, _>(&$x_trf)
            .contract_packet::<m![Z / 16 % 4]>()
            .contract_time::<m![$row % 60 = 12, Z / 64 % 30]>()
            .contract_lane::<m![$row % 60 = 12, Z / 64 % 30, Z / 1920], m![Z / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Z / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Z, m![$row % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div($gain)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<$slr, m![$row % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![$row % 60 = 12 / 4], m![$row % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![$row % 60 = 12 % 4]>()
            .commit_view($out.view_mut().tile::<m![$row % 60], 12, m![$row % 60 = 12 #{!} 60]>($start));
    }};
}

macro_rules! load_tile {
    ($ctx:ident, $w:ident, $start:literal, $cl:ty, $sl:ty, $row:ident, $col:ident) => {{
        let t: DmTensor<f4e2m1, Chip, $cl, $sl, m![$row % 60 = 12, $col % 1920]> = $w
            .view()
            .tile::<m![$row % 60], 12, m![$row / 60, $row % 60 = 12 # 60, $col]>($start)
            .to_dm(&mut $ctx.tdma);
        let t: DmTensor<f4e2m1, Chip, $cl, $sl, m![$row % 60 = 12, Z % 1920]> = unsafe { t.reshape() };
        t
    }};
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![H % 1920]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_trf = quantize!(ctx, x, UpGateClusters, UpGateColumns, 128f32);

    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, Z / 16 % 120]> =
        unsafe { up_scale.reshape() };
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, Z / 16 % 120]> =
        unsafe { gate_scale.reshape() };

    let up0 = load_tile!(ctx, up_weight_packed, 0, UpGateClusters, UpGateColumns, L, H);
    let gate0 = load_tile!(ctx, gate_weight_packed, 0, UpGateClusters, UpGateColumns, L, H);
    let up1 = load_tile!(ctx, up_weight_packed, 12, UpGateClusters, UpGateColumns, L, H);
    let gate1 = load_tile!(ctx, gate_weight_packed, 12, UpGateClusters, UpGateColumns, L, H);
    let up2 = load_tile!(ctx, up_weight_packed, 24, UpGateClusters, UpGateColumns, L, H);
    let gate2 = load_tile!(ctx, gate_weight_packed, 24, UpGateClusters, UpGateColumns, L, H);
    let up3 = load_tile!(ctx, up_weight_packed, 36, UpGateClusters, UpGateColumns, L, H);
    let gate3 = load_tile!(ctx, gate_weight_packed, 36, UpGateClusters, UpGateColumns, L, H);
    let up4 = load_tile!(ctx, up_weight_packed, 48, UpGateClusters, UpGateColumns, L, H);
    let gate4 = load_tile!(ctx, gate_weight_packed, 48, UpGateClusters, UpGateColumns, L, H);

    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    fused_tile!(ctx, up0, up_scale, x_trf, up, 0, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, gate0, gate_scale, x_trf, gate, 0, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, up1, up_scale, x_trf, up, 12, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, gate1, gate_scale, x_trf, gate, 12, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, up2, up_scale, x_trf, up, 24, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, gate2, gate_scale, x_trf, gate, 24, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, up3, up_scale, x_trf, up, 36, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, gate3, gate_scale, x_trf, gate, 36, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, up4, up_scale, x_trf, up, 48, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);
    fused_tile!(ctx, gate4, gate_scale, x_trf, gate, 48, UpGateClusters, UpGateColumns, UpGateRows, L, 128f32);


    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    // Both clusters contract over all of L, so the activation goes through chip-wide HBM.
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);

    let x_trf = quantize!(ctx, x, DownClusters, DownRowsByColumns, 4096f32);

    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, Z / 16 % 120]> =
        unsafe { down_scale.reshape() };
    let down0 = load_tile!(ctx, down_weight_packed, 0, DownClusters, DownRowsByColumns, H, L);
    let down1 = load_tile!(ctx, down_weight_packed, 12, DownClusters, DownRowsByColumns, H, L);
    let down2 = load_tile!(ctx, down_weight_packed, 24, DownClusters, DownRowsByColumns, H, L);
    let down3 = load_tile!(ctx, down_weight_packed, 36, DownClusters, DownRowsByColumns, H, L);
    let down4 = load_tile!(ctx, down_weight_packed, 48, DownClusters, DownRowsByColumns, H, L);

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    fused_tile!(ctx, down0, down_scale, x_trf, down, 0, DownClusters, DownRowsByColumns, DownRows, H, 4096f32);
    fused_tile!(ctx, down1, down_scale, x_trf, down, 12, DownClusters, DownRowsByColumns, DownRows, H, 4096f32);
    fused_tile!(ctx, down2, down_scale, x_trf, down, 24, DownClusters, DownRowsByColumns, DownRows, H, 4096f32);
    fused_tile!(ctx, down3, down_scale, x_trf, down, 36, DownClusters, DownRowsByColumns, DownRows, H, 4096f32);
    fused_tile!(ctx, down4, down_scale, x_trf, down, 48, DownClusters, DownRowsByColumns, DownRows, H, 4096f32);

    let down: HbmTensor<bf16, Chip, m![H]> = down.to_hbm(&mut ctx.tdma);
    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = down.to_dm(&mut ctx.tdma);

    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(down.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_widen_concat::<m![H / 8], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
