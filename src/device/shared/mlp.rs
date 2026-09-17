
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateClusters = m![L / 7680];
/// One cluster's half of L: 128 row groups by 2 column groups, all 256 of its slices.
pub(crate) type UpGateColumns = m![L / 60 % 128, H / 1920];
/// Where the two column partials of a row group meet.
pub(crate) type UpGateRows = m![L / 60 % 128, 1 # 2];
/// Two row groups joined into one slice, which is the 120-value row the GeGLU works on.
pub(crate) type UpGateRowsPaired = m![L / 120 % 64, 1 # 4];

pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![1], m![H % 1920]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (
    DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
) {
    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 4;
    const TILES_PER_PASS: usize = 3;
    const PASSES: usize = ROWS_PER_SLICE / (ROWS_PER_PASS * TILES_PER_PASS);

    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let up0_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            up_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 0)
                .to_dm(&mut ctx.tdma);
        let gate0_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            gate_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 0)
                .to_dm(&mut ctx.tdma);
        let up1_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            up_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 4)
                .to_dm(&mut ctx.tdma);
        let gate1_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            gate_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 4)
                .to_dm(&mut ctx.tdma);
        let up2_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            up_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 8)
                .to_dm(&mut ctx.tdma);
        let gate2_packed: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> =
            gate_weight_packed
                .view()
                .tile::<m![L % 60], 4, m![L / 60, L % 60 = 4 # 60, H]>(12 * i + 8)
                .to_dm(&mut ctx.tdma);

        let up0_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 0),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let gate0_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 0),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let up1_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 4),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let gate1_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 4),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let up2_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                up_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 8),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let gate2_scale: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H / 16 % 120]> = ctx
            .sub
            .begin(
                gate_weight_scale
                    .view()
                    .tile::<m![L % 60], 4, m![L % 60 = 4 # 60, H / 16 % 120]>(12 * i + 8),
            )
            .fetch::<m![L % 60 = 4], m![H / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 16 / 8 % 15], m![H / 16 % 8]>()
            .to_vrf();

        let up0_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(up0_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up0_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        let gate0_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(gate0_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate0_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        let up1_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(up1_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up1_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        let gate1_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(gate1_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate1_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        let up2_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(up2_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up2_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        let gate2_weight: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 4, H % 1920]> = ctx
            .main
            .begin(gate2_packed.view())
            .fetch::<m![L % 60 = 4], m![H % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 60 = 4, H / 4 % 480], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate2_scale)
            .vector_widen_concat::<m![L % 60 = 4, H / 8 % 240], m![H % 8]>()
            .vector_final()
            .cast::<bf16, m![H % 8 # 16]>()
            .commit_trim::<m![H % 8]>()
            .commit();

        ctx.main
            .begin(up0_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(up.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 0));

        ctx.main
            .begin(gate0_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 0));

        ctx.main
            .begin(up1_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(up.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 4));

        ctx.main
            .begin(gate1_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 4));

        ctx.main
            .begin(up2_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(up.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 8));

        ctx.main
            .begin(gate2_weight.view())
            .fetch::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .collect::<m![L % 60 = 4, H / 16 % 120], m![H % 16]>()
            .contract_outer::<m![L % 60 = 4, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![L % 60 = 4]>()
            .contract_lane::<m![L % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![L % 60 = 4 # 16]>()
            .commit_trim::<m![L % 60 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L % 60], 4, m![L % 60 = 4 #{!} 60]>(12 * i + 8));
    }

    (up, gate)
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
    let x_trf: TrfTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![1], m![H % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![1], m![H % 1920]>()
        .collect::<m![H / 16 % 120], m![H % 16]>()
        .to_trf();

    let (up, gate) = project_up_and_gate(
        ctx,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    // Both clusters contract over all of L, so the activation goes through chip-wide HBM.
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);
    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> =
        project_down(ctx, &x, down_weight_packed, down_weight_scale).to_dm(&mut ctx.tdma);

    let down_global_scale: DmTensor<f32, Chip, Cluster, Slice, m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let down: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = ctx
        .main
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
        .commit();

    down
}

pub(crate) fn geglu(
    ctx: &mut Context,
    up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 2 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let up_global_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_global_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 2 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_global_scale_vrf)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![1], m![L % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), INVSQRT2)
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();

    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(2f32)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

/// The down projection's rows split over the chip's two clusters: cluster 0 owns rows
/// 0..1919, cluster 1 owns 1920..3839. The clusters read HBM on separate paths, so each
/// taking half the weight halves the transfer.
pub(crate) type DownClusters = m![H / 1920];
/// Where the eight column partials of a row group meet.
pub(crate) type DownRows = m![H / 60 % 32, 1 # 8];
/// One cluster's 32 row groups by 8 column groups: all 256 of its slices.
pub(crate) type DownRowsByColumns = m![H / 60 % 32, L / 1920];

pub(crate) fn project_down(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> HbmTensor<bf16, Chip, m![H]> {
    let x_trf: TrfTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![1], m![L % 1920]> = ctx
        .sub
        .begin(x.view())
        .fetch::<m![L / 16 % 120], m![L % 16]>()
        .collect::<m![L / 16 % 120], m![L % 16]>()
        .to_trf();

    const ROWS_PER_SLICE: usize = 60;
    const ROWS_PER_PASS: usize = 4;
    const TILES_PER_PASS: usize = 5;
    const PASSES: usize = ROWS_PER_SLICE / (ROWS_PER_PASS * TILES_PER_PASS);

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    for i in 0..PASSES {
        let down_packed_a: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> =
            down_weight_packed
                .view()
                .tile::<m![H % 60], 4, m![H / 60, H % 60 = 4 # 60, L]>(20 * i + 0)
                .to_dm(&mut ctx.tdma);
        let down_packed_b: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> =
            down_weight_packed
                .view()
                .tile::<m![H % 60], 4, m![H / 60, H % 60 = 4 # 60, L]>(20 * i + 4)
                .to_dm(&mut ctx.tdma);
        let down_packed_c: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> =
            down_weight_packed
                .view()
                .tile::<m![H % 60], 4, m![H / 60, H % 60 = 4 # 60, L]>(20 * i + 8)
                .to_dm(&mut ctx.tdma);
        let down_packed_d: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> =
            down_weight_packed
                .view()
                .tile::<m![H % 60], 4, m![H / 60, H % 60 = 4 # 60, L]>(20 * i + 12)
                .to_dm(&mut ctx.tdma);
        let down_packed_e: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> =
            down_weight_packed
                .view()
                .tile::<m![H % 60], 4, m![H / 60, H % 60 = 4 # 60, L]>(20 * i + 16)
                .to_dm(&mut ctx.tdma);

        let down_scale_a: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L / 16 % 120]> =
            ctx.sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 60], 4, m![H % 60 = 4 # 60, L / 16 % 120]>(20 * i + 0),
                )
                .fetch::<m![H % 60 = 4], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
                .to_vrf();

        let down_scale_b: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L / 16 % 120]> =
            ctx.sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 60], 4, m![H % 60 = 4 # 60, L / 16 % 120]>(20 * i + 4),
                )
                .fetch::<m![H % 60 = 4], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
                .to_vrf();

        let down_scale_c: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L / 16 % 120]> =
            ctx.sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 60], 4, m![H % 60 = 4 # 60, L / 16 % 120]>(20 * i + 8),
                )
                .fetch::<m![H % 60 = 4], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
                .to_vrf();

        let down_scale_d: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L / 16 % 120]> =
            ctx.sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 60], 4, m![H % 60 = 4 # 60, L / 16 % 120]>(20 * i + 12),
                )
                .fetch::<m![H % 60 = 4], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
                .to_vrf();

        let down_scale_e: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L / 16 % 120]> =
            ctx.sub
                .begin(
                    down_weight_scale
                        .view()
                        .tile::<m![H % 60], 4, m![H % 60 = 4 # 60, L / 16 % 120]>(20 * i + 16),
                )
                .fetch::<m![H % 60 = 4], m![L / 16 % 120]>()
                .fetch_cast::<f32>()
                .collect::<m![H % 60 = 4, L / 16 / 8 % 15], m![L / 16 % 8]>()
                .to_vrf();

        let down_weight_a: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> = ctx
            .main
            .begin(down_packed_a.view())
            .fetch::<m![H % 60 = 4], m![L % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 4, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale_a)
            .vector_widen_concat::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        let down_weight_b: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> = ctx
            .main
            .begin(down_packed_b.view())
            .fetch::<m![H % 60 = 4], m![L % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 4, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale_b)
            .vector_widen_concat::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        let down_weight_c: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> = ctx
            .main
            .begin(down_packed_c.view())
            .fetch::<m![H % 60 = 4], m![L % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 4, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale_c)
            .vector_widen_concat::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        let down_weight_d: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> = ctx
            .main
            .begin(down_packed_d.view())
            .fetch::<m![H % 60 = 4], m![L % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 4, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale_d)
            .vector_widen_concat::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        let down_weight_e: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 4, L % 1920]> = ctx
            .main
            .begin(down_packed_e.view())
            .fetch::<m![H % 60 = 4], m![L % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 4, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale_e)
            .vector_widen_concat::<m![H % 60 = 4, L / 8 % 240], m![L % 8]>()
            .vector_final()
            .cast::<bf16, m![L % 8 # 16]>()
            .commit_trim::<m![L % 8]>()
            .commit();

        ctx.main
            .begin(down_weight_a.view())
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 60 = 4]>()
            .contract_lane::<m![H % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 60 = 4 # 16]>()
            .commit_trim::<m![H % 60 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(20 * i + 0));

        ctx.main
            .begin(down_weight_b.view())
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 60 = 4]>()
            .contract_lane::<m![H % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 60 = 4 # 16]>()
            .commit_trim::<m![H % 60 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(20 * i + 4));

        ctx.main
            .begin(down_weight_c.view())
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 60 = 4]>()
            .contract_lane::<m![H % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 60 = 4 # 16]>()
            .commit_trim::<m![H % 60 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(20 * i + 8));

        ctx.main
            .begin(down_weight_d.view())
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 60 = 4]>()
            .contract_lane::<m![H % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 60 = 4 # 16]>()
            .commit_trim::<m![H % 60 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(20 * i + 12));

        ctx.main
            .begin(down_weight_e.view())
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![H % 60 = 4, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H % 60 = 4]>()
            .contract_lane::<m![H % 60 = 4], m![1 # 8]>(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 4]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![H % 60 = 4 # 16]>()
            .commit_trim::<m![H % 60 = 4]>()
            .commit_view(down.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(20 * i + 16));
    }

    down.to_hbm(&mut ctx.tdma)
}

