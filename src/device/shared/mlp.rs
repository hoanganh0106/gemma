
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Replicated, Slice};

const INVSQRT2: f32 = 0.70710678118f32;

pub(crate) type UpGateRows = m![L / 60];
pub(crate) type UpGateRowsPaired = m![L / 120, 1 # 2];

pub(crate) type AutoUgCluster2 = m![L # 16384 / 8192];
pub(crate) type AutoUgSlices2 = m![L # 16384 / 32 % 256];
pub(crate) fn project_up_and_gate(
    ctx: &mut Context,
    x_trf: &TrfTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![1], m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>, gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>, gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> (DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>, DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>) {
    let up_p: DmTensor<f4e2m1, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32, H]> = up_weight_packed.to_dm(&mut ctx.tdma);
    let up_s: DmTensor<f8e4m3, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32, H / 16]> = up_weight_scale.to_dm(&mut ctx.tdma);
    let gate_p: DmTensor<f4e2m1, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32, H]> = gate_weight_packed.to_dm(&mut ctx.tdma);
    let gate_s: DmTensor<f8e4m3, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32, H / 16]> = gate_weight_scale.to_dm(&mut ctx.tdma);
    let mut up: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32]> = DmTensor::new();
    let mut gate: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32]> = DmTensor::new();
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(up_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(0))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(up_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(0))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_up_0: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_up_0.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(0));
        ctx.main.begin(partial_up_0.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(4));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(gate_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(0))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(gate_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(0))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_gate_0: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_gate_0.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(0));
        ctx.main.begin(partial_gate_0.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(4));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(up_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(8))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(up_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(8))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_up_8: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_up_8.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(8));
        ctx.main.begin(partial_up_8.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(12));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(gate_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(8))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(gate_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(8))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_gate_8: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_gate_8.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(8));
        ctx.main.begin(partial_gate_8.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(12));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(up_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(16))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(up_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(16))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_up_16: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_up_16.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(16));
        ctx.main.begin(partial_up_16.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(20));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(gate_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(16))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(gate_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(16))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_gate_16: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_gate_16.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(16));
        ctx.main.begin(partial_gate_16.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(20));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(up_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(24))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(up_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(24))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_up_24: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_up_24.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(24));
        ctx.main.begin(partial_up_24.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(up.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(28));
    }
    {
        let sv: VrfTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H / 16]> = ctx.sub
            .begin(gate_s.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H / 16]>(24))
            .fetch::<m![L # 16384 % 32 = 8], m![H / 16]>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 128], m![H / 16 % 8]>().to_vrf();
        let w: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, H]> = ctx.main
            .begin(gate_p.view().tile::<m![L # 16384 % 32], 8, m![L # 16384 % 32 = 8 # 32, H]>(24))
            .fetch::<m![L # 16384 % 32 = 8], m![H]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L # 16384 % 32 = 8, H / 4], m![H % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sv)
            .vector_widen_concat::<m![L # 16384 % 32 = 8, H / 8], m![H % 8]>()
            .vector_final().cast::<bf16, m![H % 8 # 16]>().commit_trim::<m![H % 8]>().commit();
        let partial_gate_24: DmTensor<f32, Chip, AutoUgCluster2, AutoUgSlices2, m![L # 16384 % 32 = 8, 1 # 8]> = ctx.main.begin(w.view()).fetch::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .collect::<m![L # 16384 % 32 = 8, H / 16], m![H % 16]>()
            .contract_outer::<m![L # 16384 % 32 = 8, H / 32], m![H % 32], _, _, _>(&x_trf)
            .contract_packet::<m![1]>().contract_time::<m![L # 16384 % 32 = 8]>()
            .contract_lane::<m![L # 16384 % 32 = 8], m![1 # 8]>(LaneMode::Interleaved)
            .commit_trim::<m![1 # 8]>().commit();
        ctx.main.begin(partial_gate_24.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(0))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(24));
        ctx.main.begin(partial_gate_24.view().tile::<m![L # 16384 % 32 = 8], 4, m![L # 16384 % 32 = 4 # 8, 1 # 8]>(4))
            .fetch::<m![L # 16384 % 32 = 4], m![1 # 8]>().collect::<m![L # 16384 % 32 = 4], m![1 # 8]>()
            .cast::<bf16, m![1 # 16]>().transpose::<m![1], m![L # 16384 % 32 = 4 # 16]>()
            .commit_trim::<m![L # 16384 % 32 = 4]>()
            .commit_view(gate.view_mut().tile::<m![L # 16384 % 32], 4, m![L # 16384 % 32 = 4 #{!} 32]>(28));
    }
    let uh: HbmTensor<bf16, Chip, m![L]> = up.to_hbm(&mut ctx.tdma);
    let gh: HbmTensor<bf16, Chip, m![L]> = gate.to_hbm(&mut ctx.tdma);
    (uh.to_dm(&mut ctx.tdma), gh.to_dm(&mut ctx.tdma))
}


pub(crate) fn feedforward(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, Cluster, Replicated, m![H]>,
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
    // All Replicated input slices contain the same normalized H vector.
    // Select its first physical copy; no reordering or precision change.
    let x_one: DmTensor<bf16, Chip, Cluster, Slice, m![H]> = unsafe { x.reshape() };
    let x_hbm: HbmTensor<bf16, Chip, m![H]> = x_one.to_hbm(&mut ctx.tdma);
    let x2: DmTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![H]> = x_hbm.to_dm(&mut ctx.tdma);
    let x_trf: TrfTensor<bf16, Chip, AutoUgCluster2, AutoUgSlices2, m![1], m![H]> = ctx.sub.begin(x2.view())
        .fetch::<m![H / 16], m![H % 16]>().collect::<m![H / 16], m![H % 16]>().to_trf();


    let (up, gate) = project_up_and_gate(
        ctx,
        &x_trf,
        up_weight_packed,
        gate_weight_packed,
        up_weight_scale,
        gate_weight_scale,
    );
    let x = geglu(ctx, up, gate, up_global_scale, gate_global_scale);
    // EXP-004B: skip the intermediate UpGateRowsPaired -> DownRows full-L
    // materialization. project_down converts directly to its consumer layout.
    let down = project_down(ctx, &x, down_weight_packed, down_weight_scale);

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
    up: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    gate: DmTensor<bf16, Chip, Cluster, UpGateRows, m![L % 60]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> {
    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let up_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let up: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
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

    let gate_global_scale: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> =
        gate_global_scale.to_dm(&mut ctx.tdma);
    let gate_global_scale_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(gate_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
        .main
        .begin(gate.view())
        .fetch::<m![L / 4 % 15], m![L % 4 # 16]>()
        .switch::<UpGateRowsPaired, m![L / 4 % 15, L / 60 % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![L / 4 % 15, L / 60 % 2], m![L % 4 # 16]>()
        .commit_trim::<m![L % 4]>()
        .commit();

    let gate: DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
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

    let gelu: DmTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
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

    let gelu_vrf: VrfTensor<f32, Chip, Cluster, UpGateRowsPaired, m![L % 120]> = ctx
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

pub(crate) type DownRows = m![H / 120, 1 # 8];
pub(crate) type DownRowsByColumns = m![H / 120, L / 1920];

pub(crate) type AutoDownCluster2 = m![H / 1920];
pub(crate) type AutoDownSlices2 = m![H / 60 % 32, L / 1920];
pub(crate) type AutoDownRows2 = m![H / 60 % 32, 1 # 8];
pub(crate) fn project_down(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, UpGateRowsPaired, m![L % 120]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, Cluster, Slice, m![H]> {
    let x_hbm: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x2: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![L % 1920]> = x_hbm.to_dm(&mut ctx.tdma);
    let mut down2: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownRows2, m![H % 60]> = DmTensor::new();
    let s_all: DmTensor<f8e4m3, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60, L / 16 % 120]> = down_weight_scale.to_dm(&mut ctx.tdma);
    {
        // EXP-113 merged dequant group rows 0..15
        let pg: DmTensor<f4e2m1, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = down_weight_packed.view()
            .tile::<m![H % 60], 16, m![H / 60, H % 60 = 16 # 60, L]>(0).to_dm(&mut ctx.tdma);
        let sg = s_all.view().tile::<m![H % 60], 16, m![H % 60 = 16 # 60, L / 16 % 120]>(0);
        let svg: VrfTensor<f32, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L / 16 % 120]> = ctx.sub.begin(sg)
            .fetch::<m![H % 60 = 16], m![L / 16 % 120]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 128 % 15], m![L / 16 % 8]>().to_vrf();
        let wg: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = ctx.main.begin(pg.view())
            .fetch::<m![H % 60 = 16], m![L % 1920]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 16, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &svg)
            .vector_widen_concat::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_final().cast::<bf16, m![L % 8 # 16]>().commit_trim::<m![L % 8]>().commit();
        let w0 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(0);
        let t0: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w0)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t0)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(0));
        let w8 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(8);
        let t8: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w8)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t8)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(8));
    }
    {
        // EXP-113 merged dequant group rows 16..31
        let pg: DmTensor<f4e2m1, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = down_weight_packed.view()
            .tile::<m![H % 60], 16, m![H / 60, H % 60 = 16 # 60, L]>(16).to_dm(&mut ctx.tdma);
        let sg = s_all.view().tile::<m![H % 60], 16, m![H % 60 = 16 # 60, L / 16 % 120]>(16);
        let svg: VrfTensor<f32, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L / 16 % 120]> = ctx.sub.begin(sg)
            .fetch::<m![H % 60 = 16], m![L / 16 % 120]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 128 % 15], m![L / 16 % 8]>().to_vrf();
        let wg: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = ctx.main.begin(pg.view())
            .fetch::<m![H % 60 = 16], m![L % 1920]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 16, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &svg)
            .vector_widen_concat::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_final().cast::<bf16, m![L % 8 # 16]>().commit_trim::<m![L % 8]>().commit();
        let w16 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(0);
        let t16: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w16)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t16)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(16));
        let w24 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(8);
        let t24: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w24)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t24)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(24));
    }
    {
        // EXP-113 merged dequant group rows 32..47
        let pg: DmTensor<f4e2m1, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = down_weight_packed.view()
            .tile::<m![H % 60], 16, m![H / 60, H % 60 = 16 # 60, L]>(32).to_dm(&mut ctx.tdma);
        let sg = s_all.view().tile::<m![H % 60], 16, m![H % 60 = 16 # 60, L / 16 % 120]>(32);
        let svg: VrfTensor<f32, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L / 16 % 120]> = ctx.sub.begin(sg)
            .fetch::<m![H % 60 = 16], m![L / 16 % 120]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 128 % 15], m![L / 16 % 8]>().to_vrf();
        let wg: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 16, L % 1920]> = ctx.main.begin(pg.view())
            .fetch::<m![H % 60 = 16], m![L % 1920]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 16, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &svg)
            .vector_widen_concat::<m![H % 60 = 16, L / 8 % 240], m![L % 8]>()
            .vector_final().cast::<bf16, m![L % 8 # 16]>().commit_trim::<m![L % 8]>().commit();
        let w32 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(0);
        let t32: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w32)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t32)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(32));
        let w40 = wg.view().tile::<m![H % 60 = 16], 8, m![H % 60 = 8 # 16, L % 1920]>(8);
        let t40: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w40)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t40)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(40));
    }
    {
        // EXP-113 merged dequant group rows 48..59
        let pg: DmTensor<f4e2m1, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 12, L % 1920]> = down_weight_packed.view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>(48).to_dm(&mut ctx.tdma);
        let sg = s_all.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, L / 16 % 120]>(48);
        let svg: VrfTensor<f32, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 12, L / 16 % 120]> = ctx.sub.begin(sg)
            .fetch::<m![H % 60 = 12], m![L / 16 % 120]>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 128 % 15], m![L / 16 % 8]>().to_vrf();
        let wg: DmTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 12, L % 1920]> = ctx.main.begin(pg.view())
            .fetch::<m![H % 60 = 12], m![L % 1920]>().fetch_table_lookup::<f8e4m3>().fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
            .vector_init().vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H % 60 = 12, L / 4 % 480], m![L % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &svg)
            .vector_widen_concat::<m![H % 60 = 12, L / 8 % 240], m![L % 8]>()
            .vector_final().cast::<bf16, m![L % 8 # 16]>().commit_trim::<m![L % 8]>().commit();
        let w48 = wg.view().tile::<m![H % 60 = 12], 8, m![H % 60 = 8 # 12, L % 1920]>(0);
        let t48: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 8], m![L % 1920]> = ctx.sub.begin(w48)
            .fetch::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 8, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t48)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 8 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 8 # 16]>().commit_trim::<m![H % 60 = 8]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 8, m![H % 60 = 8 #{!} 60]>(48));
        let w56 = wg.view().tile::<m![H % 60 = 12], 4, m![H % 60 = 4 # 12, L % 1920]>(8);
        let t56: TrfTensor<bf16, Chip, AutoDownCluster2, AutoDownSlices2, m![H % 60 = 4], m![L % 1920]> = ctx.sub.begin(w56)
            .fetch::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>()
            .collect::<m![H % 60 = 4, L / 16 % 120], m![L % 16]>().to_trf();
        ctx.main.begin(x2.view()).fetch::<m![L / 16 % 120], m![L % 16]>().collect::<m![L / 16 % 120], m![L % 16]>()
            .contract_outer::<m![L / 32 % 60], m![L % 32], _, _, _>(&t56)
            .contract_packet::<m![1]>().contract_time::<m![1]>()
            .contract_lane::<m![1], m![H % 60 = 4 # 8]>(LaneMode::Interleaved)
            .vector_init().vector_inter_slice_reduce::<AutoDownRows2, m![1]>(InterSliceReduceOpF32::Add)
            .vector_final().cast::<bf16, m![H % 60 = 4 # 16]>().commit_trim::<m![H % 60 = 4]>()
            .commit_view(down2.view_mut().tile::<m![H % 60], 4, m![H % 60 = 4 #{!} 60]>(56));
    }
    let out_hbm: HbmTensor<bf16, Chip, m![H]> = down2.to_hbm(&mut ctx.tdma);
    out_hbm.to_dm(&mut ctx.tdma)
}
