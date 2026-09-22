
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
    let mut up: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, UpGateClusters, UpGateRows, m![L % 60]> = DmTensor::new();

    let up_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_weight_scale: DmTensor<f8e4m3, Chip, UpGateClusters, UpGateColumns, m![L % 60, H / 16 % 120]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);

    // Every transfer first, so the DMA engine never waits on arithmetic.
    let up_packed0: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        up_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(0)
            .to_dm(&mut ctx.tdma);
    let gate_packed0: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        gate_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(0)
            .to_dm(&mut ctx.tdma);
    let up_packed1: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        up_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(12)
            .to_dm(&mut ctx.tdma);
    let gate_packed1: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        gate_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(12)
            .to_dm(&mut ctx.tdma);
    let up_packed2: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        up_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(24)
            .to_dm(&mut ctx.tdma);
    let gate_packed2: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        gate_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(24)
            .to_dm(&mut ctx.tdma);
    let up_packed3: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        up_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(36)
            .to_dm(&mut ctx.tdma);
    let gate_packed3: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        gate_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(36)
            .to_dm(&mut ctx.tdma);
    let up_packed4: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        up_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(48)
            .to_dm(&mut ctx.tdma);
    let gate_packed4: DmTensor<f4e2m1, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> =
        gate_weight_packed
            .view()
            .tile::<m![L % 60], 12, m![L / 60, L % 60 = 12 # 60, H]>(48)
            .to_dm(&mut ctx.tdma);

    let up_scale0: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            up_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(0),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let up_weight0: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(up_packed0.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_scale0)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(up_weight0.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(0));

    let gate_scale0: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            gate_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(0),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let gate_weight0: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(gate_packed0.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale0)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(gate_weight0.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(0));

    let up_scale1: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            up_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(12),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let up_weight1: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(up_packed1.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_scale1)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(up_weight1.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12));

    let gate_scale1: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            gate_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(12),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let gate_weight1: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(gate_packed1.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale1)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(gate_weight1.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(12));

    let up_scale2: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            up_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(24),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let up_weight2: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(up_packed2.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_scale2)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(up_weight2.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(24));

    let gate_scale2: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            gate_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(24),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let gate_weight2: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(gate_packed2.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale2)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(gate_weight2.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(24));

    let up_scale3: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            up_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(36),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let up_weight3: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(up_packed3.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_scale3)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(up_weight3.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(36));

    let gate_scale3: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            gate_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(36),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let gate_weight3: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(gate_packed3.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale3)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(gate_weight3.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(36));

    let up_scale4: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            up_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(48),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let up_weight4: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(up_packed4.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &up_scale4)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(up_weight4.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(up.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(48));

    let gate_scale4: VrfTensor<f32, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H / 16 % 120]> = ctx
        .sub
        .begin(
            gate_weight_scale
                .view()
                .tile::<m![L % 60], 12, m![L % 60 = 12 # 60, H / 16 % 120]>(48),
        )
        .fetch::<m![L % 60 = 12], m![H / 16 % 120]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 16 / 8 % 15], m![H / 16 % 8]>()
        .to_vrf();

    let gate_weight4: DmTensor<bf16, Chip, UpGateClusters, UpGateColumns, m![L % 60 = 12, H % 1920]> = ctx
        .main
        .begin(gate_packed4.view())
        .fetch::<m![L % 60 = 12], m![H % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L % 60 = 12, H / 4 % 480], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gate_scale4)
        .vector_widen_concat::<m![L % 60 = 12, H / 8 % 240], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(gate_weight4.view())
        .fetch::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .collect::<m![L % 60 = 12, H / 16 % 120], m![H % 16]>()
        .contract_outer::<m![L % 60 = 12, H / 32 % 60], m![H % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 60 = 12]>()
        .contract_lane::<m![L % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<UpGateRows, m![L % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![L % 60 = 12 / 4], m![L % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![L % 60 = 12 % 4]>()
        .commit_view(gate.view_mut().tile::<m![L % 60], 12, m![L % 60 = 12 #{!} 60]>(48));

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

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();

    let down_weight_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);

    // Every transfer first, so the DMA engine never waits on arithmetic.
    let down_packed0: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> =
        down_weight_packed
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(0)
            .to_dm(&mut ctx.tdma);
    let down_packed1: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> =
        down_weight_packed
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(12)
            .to_dm(&mut ctx.tdma);
    let down_packed2: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> =
        down_weight_packed
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(24)
            .to_dm(&mut ctx.tdma);
    let down_packed3: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> =
        down_weight_packed
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(36)
            .to_dm(&mut ctx.tdma);
    let down_packed4: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> =
        down_weight_packed
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(48)
            .to_dm(&mut ctx.tdma);

    let down_scale0: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(0),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

    let down_weight0: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = ctx
        .main
        .begin(down_packed0.view())
        .fetch::<m![H % 60 = 12], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale0)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight0.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(0));

    let down_scale1: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(12),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

    let down_weight1: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = ctx
        .main
        .begin(down_packed1.view())
        .fetch::<m![H % 60 = 12], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale1)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight1.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(12));

    let down_scale2: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(24),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

    let down_weight2: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = ctx
        .main
        .begin(down_packed2.view())
        .fetch::<m![H % 60 = 12], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale2)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight2.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(24));

    let down_scale3: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(36),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

    let down_weight3: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = ctx
        .main
        .begin(down_packed3.view())
        .fetch::<m![H % 60 = 12], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale3)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight3.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(36));

    let down_scale4: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L / 16 % 120]> =
        ctx.sub
            .begin(
                down_weight_scale
                    .view()
                    .tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(48),
            )
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 16 / 8 % 15], m![L / 16 % 8]>()
            .to_vrf();

    let down_weight4: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = ctx
        .main
        .begin(down_packed4.view())
        .fetch::<m![H % 60 = 12], m![L % 1920]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_scale4)
        .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    ctx.main
        .begin(down_weight4.view())
        .fetch::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .collect::<m![H % 60 = 12, L / 16 % 120], m![L % 16]>()
        .contract_outer::<m![H % 60 = 12, L / 32 % 60], m![L % 32], _, _, _>(&x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 60 = 12]>()
        .contract_lane::<m![H % 60 = 12], m![1 # 8]>(LaneMode::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
        .commit_trim::<m![H % 60 = 12 % 4]>()
        .commit_view(down.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>(48));

    down.to_hbm(&mut ctx.tdma)
}

