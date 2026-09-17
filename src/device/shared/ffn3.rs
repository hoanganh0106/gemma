//! Feed-forward, second layout: whole weight rows for up/gate, the block dequantization folded
//! into an f8 x f8 contraction (see `ffn2`), and the normalized activation gathered into every
//! slice by the Switch instead of travelling through HBM.

use furiosa_opt_std::prelude::*;

use super::mlp::{DownClusters, DownRows, DownRowsByColumns, UpGateClusters, UpGateRowsPaired};
use crate::axes::{H, L};
use crate::device::layout::{Cluster, Slice};
use crate::{Chip, EPS};

axes![Rep32 = 32, Ring8 = 8, T2 = 2, I2 = 2, Xc = 1920, Xh = 3840, Zd = 3840];

const H_F32: f32 = H::SIZE as f32;
const INVSQRT2: f32 = 0.70710678118f32;

/// Gain applied to the activation before it is split into f8 terms (a power of two, undone
/// exactly after the contraction).
const UP_GAIN: f32 = 128.0;
const DOWN_GAIN: f32 = 4096.0;

/// 32 real copies of eight 480-column pieces.
type Pieces = m![Rep32, H / 480];
type PiecesX = m![Rep32, Xh / 480];
/// The same 256 slices once each ring of eight has gathered the whole vector.
type Gathered = m![Rep32, Ring8];
/// One cluster's 7680 rows, 30 whole rows a slice.
pub(crate) type RowSlices = m![L / 30 % 256];

/// RMSNorm on the pieces (rounded to bf16 as the reference does), split into two exact f8 terms,
/// gathered into every slice and stored in the TRF as two lanes of 3840 columns.
fn normalize_quantize(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> TrfTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![T2], m![Xh]> {
    let x: DmTensor<bf16, Chip, UpGateClusters, Pieces, m![H % 480]> = x.to_dm(&mut ctx.tdma);

    let mean_square: DmTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let reduced: DmTensor<f32, Chip, UpGateClusters, Gathered, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Gathered, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, UpGateClusters, Gathered, m![1 # 8]> = ctx
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
    let rms: DmTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, UpGateClusters, Pieces, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<bf16, Chip, UpGateClusters, Pieces, m![H % 480]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Rename the columns Xh; the two f8 terms sit side by side under T2.
    let normalized: DmTensor<bf16, Chip, UpGateClusters, PiecesX, m![T2 = 1, Xh % 480]> =
        unsafe { normalized.reshape() };
    let mut q: DmTensor<f8e4m3, Chip, UpGateClusters, PiecesX, m![T2, Xh % 480]> = DmTensor::new();
    ctx.main
        .begin(normalized.view())
        .fetch::<m![T2 = 1, Xh / 16 % 30], m![Xh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xh / 4 % 120], m![Xh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_GAIN)
        .vector_widen_concat::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xh % 8 # 32]>()
        .commit_trim::<m![Xh % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xh % 480]>(0));
    let q1_vrf: VrfTensor<f32, Chip, UpGateClusters, PiecesX, m![T2 = 1, Xh % 480]> = ctx
        .sub
        .begin(q.view().tile::<m![T2], 1, m![T2 = 1 # 2, Xh % 480]>(0))
        .fetch::<m![T2 = 1, Xh / 32 % 15], m![Xh % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .to_vrf();
    ctx.main
        .begin(normalized.view())
        .fetch::<m![T2 = 1, Xh / 16 % 30], m![Xh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xh / 4 % 120], m![Xh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_GAIN)
        .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
        .vector_widen_concat::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xh % 8 # 32]>()
        .commit_trim::<m![Xh % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xh % 480]>(1));

    // All-gather along each ring of eight: every slice ends with both terms of the whole vector.
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![T2, Xh]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![T2], m![Xh % 480]>()
        .switch::<Gathered, m![T2, Xh / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![T2, Xh / 32], m![Xh % 32]>()
        .commit_trim::<m![Xh % 32]>()
        .commit();
    ctx.sub
        .begin(gathered.view())
        .fetch::<m![T2, Xh / 32], m![Xh % 32]>()
        .collect::<m![T2, Xh / 32], m![Xh % 32]>()
        .to_trf()
}

/// A third of an up or gate matrix (10 whole rows a slice: even row counts stay 256-byte aligned) in one pass, one decode table: per-block
/// partial sums out, the two activation terms already added.
macro_rules! block_sums {
    ($ctx:ident, $w:ident, $x_trf:ident, $z:ident, $start:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, RowSlices, m![L % 30 = 10, H]> = $w
            .view()
            .tile::<m![L % 30], 10, m![L / 30, L % 30 = 10 # 30, H]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, Gathered, m![L % 30 = 10, Xh]> =
            unsafe { packed.reshape() };
        $ctx.main
            .begin(packed.view())
            .fetch::<m![L % 30 = 10], m![Xh]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30 = 10, Xh / 32], m![Xh % 32]>()
            .contract_outer::<m![L % 30 = 10, Xh / 64], m![Xh % 64], _, _, _>(&$x_trf)
            .contract_packet::<m![Xh / 16 % 4]>()
            .contract_time::<m![L % 30 = 10, Xh / 64]>()
            .contract_lane::<m![L % 30 = 10, Xh / 64, T2], m![Xh / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Xh / 16 % 4]>()
            .vector_intra_slice_reduce::<T2, m![L % 30 = 10, Xh / 64], m![Xh / 16 % 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(UP_GAIN)
            .vector_widen_pad::<m![Xh / 16 % 4 # 8]>()
            .vector_final()
            .commit_trim::<m![Xh / 16 % 4]>()
            .commit_view($z.view_mut().tile::<m![L % 30], 10, m![L % 30 = 10 #{!} 30, Xh / 16]>($start));
    }};
}

/// Six rows of block scales to the VRF, then sum over blocks of (partial sum x scale).
macro_rules! scale_tile {
    ($ctx:ident, $z:ident, $scale:ident, $out:ident, $start:literal) => {{
        let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30 = 6, Xh / 16]> = $ctx
            .sub
            .begin($scale.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, Xh / 16]>($start))
            .fetch::<m![L % 30 = 6], m![Xh / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = 6, Xh / 16 / 8], m![Xh / 16 % 8]>()
            .to_vrf();
        $ctx.main
            .begin($z.view().tile::<m![L % 30], 6, m![L % 30 = 6 # 30, Xh / 16]>($start))
            .fetch::<m![L % 30 = 6, Xh / 16 / 8], m![Xh / 16 % 8]>()
            .collect::<m![L % 30 = 6, Xh / 16 / 8], m![Xh / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 30 = 6, Xh / 16 / 4], m![Xh / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Xh, m![L % 30 = 6], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .transpose::<m![L % 30 = 6 / 2], m![L % 30 = 6 % 2 # 8]>()
            .commit_trim::<m![L % 30 = 6 % 2]>()
            .commit_view($out.view_mut().tile::<m![L % 30], 6, m![L % 30 = 6 #{!} 30]>($start));
    }};
}

/// x -> a two-lane TRF of exact f8 terms (1920 columns a slice): lane 0 = f8(g x), lane 1 = f8(g x - lane 0).
macro_rules! quantize {
    ($ctx:ident, $x:ident, $cl:ty, $sl:ty, $gain:expr) => {{
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
        let q: DmTensor<f8e4m3, Chip, $cl, $sl, m![Zd]> = unsafe { q.reshape() };
        let trf: TrfTensor<f8e4m3, Chip, $cl, $sl, m![Zd / 1920], m![Zd % 1920]> = $ctx
            .sub
            .begin(q.view())
            .fetch::<m![Zd / 32], m![Zd % 32]>()
            .collect::<m![Zd / 32], m![Zd % 32]>()
            .to_trf();
        trf
    }};
}

/// One 12-row tile of the down projection (column groups of 1920).
macro_rules! down_tile {
    ($ctx:ident, $w:ident, $scale:ident, $x_trf:ident, $out:ident, $start:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, L % 1920]> = $w
            .view()
            .tile::<m![H % 60], 12, m![H / 60, H % 60 = 12 # 60, L]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, Zd % 1920]> =
            unsafe { packed.reshape() };
        let scale_vrf: VrfTensor<f32, Chip, DownClusters, DownRowsByColumns, m![H % 60 = 12, Zd / 16 % 120]> = $ctx
            .sub
            .begin($scale.view().tile::<m![H % 60], 12, m![H % 60 = 12 # 60, Zd / 16 % 120]>($start))
            .fetch::<m![H % 60 = 12], m![Zd / 16 % 120]>()
            .fetch_cast::<f32>()
            .collect::<m![H % 60 = 12, Zd / 16 / 8 % 15], m![Zd / 16 % 8]>()
            .to_vrf();
        $ctx.main
            .begin(packed.view())
            .fetch::<m![H % 60 = 12], m![Zd % 1920]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H % 60 = 12, Zd / 32 % 60], m![Zd % 32]>()
            .contract_outer::<m![H % 60 = 12, Zd / 64 % 30], m![Zd % 64], _, _, _>(&$x_trf)
            .contract_packet::<m![Zd / 16 % 4]>()
            .contract_time::<m![H % 60 = 12, Zd / 64 % 30]>()
            .contract_lane::<m![H % 60 = 12, Zd / 64 % 30, Zd / 1920], m![Zd / 16 % 4 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Zd / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Zd, m![H % 60 = 12], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(DOWN_GAIN)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<DownRows, m![H % 60 = 12]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![H % 60 = 12 / 4], m![H % 60 = 12 % 4 # 16]>()
            .commit_trim::<m![H % 60 = 12 % 4]>()
            .commit_view($out.view_mut().tile::<m![H % 60], 12, m![H % 60 = 12 #{!} 60]>($start));
    }};
}

/// Four neighbouring slices (30 rows each) meet in one: the 120-value row group the GeGLU works
/// on. The global scale rides the same pass.
fn regroup(
    ctx: &mut Context,
    y: &DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> {
    let global_scale: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> =
        global_scale.to_dm(&mut ctx.tdma);
    let global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![1 # 8]> = ctx
        .sub
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    ctx.main
        .begin(y.view())
        .fetch::<m![L / 2 % 15], m![L % 2 # 8]>()
        .switch::<UpGateRowsPaired, m![L / 2 % 15, L / 30 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![L / 2 % 15, L / 30 % 4], m![L % 2 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![L % 2 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_widen_pad::<m![L % 2 # 8]>()
        .vector_final()
        .commit_trim::<m![L % 2]>()
        .commit()
}

fn geglu(
    ctx: &mut Context,
    up: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]>,
    gate: &DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]>,
) -> DmTensor<bf16, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> {
    let gelu: DmTensor<f32, Chip, UpGateClusters, UpGateRowsPaired, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
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

pub(crate) fn feedforward(
    ctx: &mut Context,
    residual: &HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
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
    let x_trf = normalize_quantize(ctx, residual, pre_ff_rms_weight);

    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![L % 30, Xh / 16]> =
        unsafe { up_scale.reshape() };
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![L % 30, Xh / 16]> =
        unsafe { gate_scale.reshape() };

    let mut up_z: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xh / 16]> = DmTensor::new();
    let mut gate_z: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xh / 16]> = DmTensor::new();
    let mut up: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30]> = DmTensor::new();

    block_sums!(ctx, up_weight_packed, x_trf, up_z, 0);
    block_sums!(ctx, gate_weight_packed, x_trf, gate_z, 0);
    block_sums!(ctx, up_weight_packed, x_trf, up_z, 10);
    block_sums!(ctx, gate_weight_packed, x_trf, gate_z, 10);
    block_sums!(ctx, up_weight_packed, x_trf, up_z, 20);
    block_sums!(ctx, gate_weight_packed, x_trf, gate_z, 20);

    scale_tile!(ctx, up_z, up_scale, up, 0);
    scale_tile!(ctx, gate_z, gate_scale, gate, 0);
    scale_tile!(ctx, up_z, up_scale, up, 6);
    scale_tile!(ctx, gate_z, gate_scale, gate, 6);
    scale_tile!(ctx, up_z, up_scale, up, 12);
    scale_tile!(ctx, gate_z, gate_scale, gate, 12);
    scale_tile!(ctx, up_z, up_scale, up, 18);
    scale_tile!(ctx, up_z, up_scale, up, 24);
    scale_tile!(ctx, gate_z, gate_scale, gate, 18);
    scale_tile!(ctx, gate_z, gate_scale, gate, 24);

    let up: DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]> = unsafe { up.reshape() };
    let gate: DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]> = unsafe { gate.reshape() };

    let up = regroup(ctx, &up, up_global_scale);
    let gate = regroup(ctx, &gate, gate_global_scale);
    let x = geglu(ctx, &up, &gate);

    // Both clusters contract over all of L, so the activation goes through chip-wide HBM.
    let x: HbmTensor<bf16, Chip, m![L]> = x.to_hbm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, DownClusters, DownRowsByColumns, m![L % 1920]> = x.to_dm(&mut ctx.tdma);
    let x_trf = quantize!(ctx, x, DownClusters, DownRowsByColumns, DOWN_GAIN);

    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, L / 16 % 120]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_scale: DmTensor<f8e4m3, Chip, DownClusters, DownRowsByColumns, m![H % 60, Zd / 16 % 120]> =
        unsafe { down_scale.reshape() };

    let mut down: DmTensor<bf16, Chip, DownClusters, DownRows, m![H % 60]> = DmTensor::new();
    down_tile!(ctx, down_weight_packed, down_scale, x_trf, down, 0);
    down_tile!(ctx, down_weight_packed, down_scale, x_trf, down, 12);
    down_tile!(ctx, down_weight_packed, down_scale, x_trf, down, 24);
    down_tile!(ctx, down_weight_packed, down_scale, x_trf, down, 36);
    down_tile!(ctx, down_weight_packed, down_scale, x_trf, down, 48);

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
