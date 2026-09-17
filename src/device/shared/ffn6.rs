//! Feed-forward: the down projection is split by COLUMN halves over the two clusters, so
//! the GeGLU output never leaves its cluster (no HBM round trip); only the two partial results meet
//! in HBM at the very end. Up and gate use whole weight rows, the block dequantization is folded
//! into an f8 x f8 contraction, and the normalized activation is gathered into every
//! slice by the Switch instead of travelling through HBM.

use furiosa_opt_std::prelude::*;

use super::mlp::UpGateClusters;
use super::rmsnorm::ReducingSlices;
use crate::axes::{Dummy8, H, L};
use crate::device::layout::Cluster;
use crate::{Chip, EPS};

// Columns inside the contraction: Ut / Dt count the 64-column packets of an up/gate row (3840
// columns) or of a down half-row (7680), Xp is the packet. The scale tiles reduce Ut / Dt only, so
// four partial sums (Xp / 16) leave the Vector Engine per row: a full commit packet, no transpose
// (the Transpose Engine is main-context only, the scale tiles run on the sub context).
axes![Rep32 = 32, Ring8 = 8, T2 = 2, Xh = 3840, Rep4 = 4, Ring64 = 64, Xg = 120, Xd = 7680, Ut = 60, Dt = 120, Xp = 64, C2 = 2, Lead4 = 4, Hg = 128, Pw = 64];

const H_F32: f32 = H::SIZE as f32;
const INVSQRT2: f32 = 0.70710678118f32;

/// Gain applied to the activation before it is split into f8 terms (a power of two, undone
/// exactly after the contraction).
const UP_GAIN: f32 = 128.0;
/// The GeGLU output is quantized BEFORE the up projection's global scale is applied (that scale
/// moved to the last pass), so it is ~1e4 times larger than the scaled value.
const DOWN_GAIN: f32 = 0.25;

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
) -> (TrfTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![1], m![T2, Ut, Xp]>, DmTensor<bf16, Chip, UpGateClusters, Pieces, m![H % 480]>) {
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
    let rms: DmTensor<f32, Chip, UpGateClusters, Gathered, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Gathered, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
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
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![T2, Ut, Xp]> =
        unsafe { gathered.reshape() };
    let x_trf = ctx
        .sub
        .begin(gathered.view())
        .fetch::<m![T2, Ut, Xp / 32], m![Xp % 32]>()
        .collect::<m![T2, Ut, Xp / 32], m![Xp % 32]>()
        .to_trf();
    (x_trf, x)
}

/// A third of an up or gate matrix (10 whole rows a slice: even row counts stay 256-byte aligned) in one pass, one decode table: per-block
/// partial sums out, the two activation terms already added.
macro_rules! block_sums {
    ($ctx:ident, $w:ident, $x_trf:ident, $z:ident, $start:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, RowSlices, m![L % 30 = 10, H]> = $w
            .view()
            .tile::<m![L % 30], 10, m![L / 30, L % 30 = 10 # 30, H]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, Gathered, m![L % 30 = 10, Ut, Xp]> =
            unsafe { packed.reshape() };
        $ctx.main
            .begin(packed.view())
            .fetch::<m![L % 30 = 10, Ut], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30 = 10, Ut, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![L % 30 = 10, Ut, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![L % 30 = 10, Ut]>()
            .contract_lane::<m![L % 30 = 10, Ut], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .commit_trim::<m![Xp / 16]>()
            .commit_view($z.view_mut().tile::<m![L % 30], 10, m![L % 30 = 10 #{!} 30, Ut, Xp / 16]>($start));
    }};
}

/// A whole up or gate matrix (30 whole rows a slice) in ONE pass, one decode table.
macro_rules! block_sums_all {
    ($ctx:ident, $w:ident, $x_trf:ident) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, RowSlices, m![L % 30, H]> = $w.to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp]> =
            unsafe { packed.reshape() };
        let z: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp / 16]> = $ctx
            .main
            .begin(packed.view())
            .fetch::<m![L % 30, Ut], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30, Ut, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![L % 30, Ut, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![L % 30, Ut]>()
            .contract_lane::<m![L % 30, Ut], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .commit_trim::<m![Xp / 16]>()
            .commit();
        z
    }};
}

/// Up to eight rows of block scales to the VRF (8 KB), then sum over blocks of (partial sum x scale).
macro_rules! scale_tile {
    ($ctx:ident, $z:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30 = $len, Ut, Xp / 16]> = $ctx
            .sub
            .begin($scale.view().tile::<m![L % 30], $len, m![L % 30 = $len # 30, Ut, Xp / 16]>($start))
            .fetch::<m![L % 30 = $len, Ut / 2], m![Ut % 2, Xp / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![L % 30 = $len, Ut / 2], m![Ut % 2, Xp / 16]>()
            .to_vrf();
        $ctx.main
            .begin($z.view().tile::<m![L % 30], $len, m![L % 30 = $len # 30, Ut, Xp / 16]>($start))
            .fetch::<m![L % 30 = $len, Ut / 2], m![Ut % 2, Xp / 16]>()
            .collect::<m![L % 30 = $len, Ut / 2], m![Ut % 2, Xp / 16]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![L % 30 = $len, Ut], m![Xp / 16]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Ut, m![L % 30 = $len], m![Xp / 16]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(UP_GAIN)
            .vector_widen_pad::<m![Xp / 16 # 8]>()
            .vector_final()
            .commit_trim::<m![Xp / 16]>()
            .commit_view($out.view_mut().tile::<m![L % 30], $len, m![L % 30 = $len #{!} 30, Xp / 16]>($start));
    }};
}

/// A cluster's 7680 GeGLU rows in groups of 120, four real copies of each group.
type Groups = m![L / 120 % 64, Rep4];
/// The same slices named by the down projection's columns (a cluster owns 7680 of them).
type GroupsX = m![Xd / 120, Rep4];
/// Every slice of a cluster once the 64 groups have been gathered.
type DownSlices = m![Ring64, Rep4];
/// Down-projection rows: 15 a slice, every row in both clusters (each cluster contracts its own
/// half of the columns).
type DownRowSlices = m![H / 30, H % 2];

/// The four partial sums of a row added up (main context: it needs the Transpose Engine to pack
/// two rows a commit packet).
fn fold4(
    ctx: &mut Context,
    y: &DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xp / 16]>,
) -> DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]> {
    let y: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30]> = ctx
        .main
        .begin(y.view())
        .fetch::<m![L % 30], m![Xp / 16 # 8]>()
        .collect::<m![L % 30], m![Xp / 16 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Xp / 16]>()
        .vector_intra_slice_reduce::<Xp, m![L % 30], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 / 2], m![L % 30 % 2 # 8]>()
        .commit_trim::<m![L % 30 % 2]>()
        .commit();
    unsafe { y.reshape() }
}

/// Four neighbouring slices (30 rows each) meet: every one of the four ends with the 120-value
/// row group the GeGLU works on.
fn regroup(
    ctx: &mut Context,
    y: &DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]>,
) -> DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> {
    ctx.main
        .begin(y.view())
        .fetch::<m![L / 2 % 15], m![L % 2 # 8]>()
        .switch::<Groups, m![L / 2 % 15, L / 30 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![L / 2 % 15, L / 30 % 4], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

/// `fold4` for the gate, with the gate's global scale riding the pass. The scalar is loaded into one
/// slice of every group of four (a cheap strided load) and handed to the other three by the Switch; a
/// plain load into all 256 slices cost 8-9K device cycles of DMA time.
fn fold4_scaled(
    ctx: &mut Context,
    y: &DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xp / 16]>,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]> {
    let global_scale: DmTensor<f32, Chip, UpGateClusters, m![L / 120 % 64, 1 # 4], m![1 # 8]> =
        global_scale.to_dm(&mut ctx.tdma);
    // Only the first slice of a group holds the scalar. An all-gather over the group hands every
    // slice the four slices' words in slice order; word 0 is the scalar, the other three are never read.
    let global_scale: DmTensor<f32, Chip, UpGateClusters, m![L / 120 % 64, Lead4], m![1 # 8]> =
        unsafe { global_scale.reshape() };
    let global_scale: DmTensor<f32, Chip, UpGateClusters, Groups, m![Lead4, 1 # 8]> = ctx
        .main
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<Groups, m![1, Lead4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![1, Lead4], m![1 # 8]>()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let global_scale: DmTensor<f32, Chip, UpGateClusters, Gathered, m![Lead4, 1 # 8]> =
        unsafe { global_scale.reshape() };
    let global_scale_vrf: VrfTensor<f32, Chip, UpGateClusters, Gathered, m![1 # 8]> = ctx
        .sub
        .begin(global_scale.view().tile::<m![Lead4], 1, m![Lead4 = 1 # 4, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let y: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30]> = ctx
        .main
        .begin(y.view())
        .fetch::<m![L % 30], m![Xp / 16 # 8]>()
        .collect::<m![L % 30], m![Xp / 16 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Xp / 16]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &global_scale_vrf)
        .vector_intra_slice_reduce::<Xp, m![L % 30], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![L % 30 / 2], m![L % 30 % 2 # 8]>()
        .commit_trim::<m![L % 30 % 2]>()
        .commit();
    unsafe { y.reshape() }
}

fn geglu(
    ctx: &mut Context,
    up: &DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]>,
    gate: &DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]>,
) -> DmTensor<bf16, Chip, UpGateClusters, Groups, m![L % 120]> {
    let gelu: DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> = ctx
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
    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> = ctx
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

/// The GeGLU output, group by group, split into two exact f8 terms; then all 64 groups of the
/// cluster gathered into every slice and stored in the TRF (two lanes of 7680 columns, named as
/// four quarters of 1920).
fn quantize_gather(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![L % 120]>,
) -> TrfTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![1], m![T2, Dt, Xp]> {
    let x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![T2 = 1, Xg]> = unsafe { x.reshape() };
    let mut q: DmTensor<f8e4m3, Chip, UpGateClusters, Groups, m![T2, Xg]> = DmTensor::new();
    ctx.main
        .begin(x.view())
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xg / 4], m![Xg % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_GAIN)
        .vector_widen_concat::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xg % 8 # 32]>()
        .commit_trim::<m![Xg % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xg]>(0));
    let q1_vrf: VrfTensor<f32, Chip, UpGateClusters, Groups, m![T2 = 1, Xg]> = ctx
        .sub
        .begin(q.view().tile::<m![T2], 1, m![T2 = 1 # 2, Xg]>(0))
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xg / 4], m![Xg % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_GAIN)
        .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
        .vector_widen_concat::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xg % 8 # 32]>()
        .commit_trim::<m![Xg % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xg]>(1));

    // The 64 groups are the columns the cluster owns: name them Xd / 120 and gather along the
    // ring of 64 (stride 4: slices with the same copy index form a ring).
    let q: DmTensor<f8e4m3, Chip, UpGateClusters, GroupsX, m![T2, Xd % 120]> = unsafe { q.reshape() };
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![T2, Xd]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![T2, Xd / 24 % 5], m![Xd % 24]>()
        .switch::<DownSlices, m![T2, Xd / 24 % 5, Xd / 120]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![T2, Xd / 24 % 5, Xd / 120], m![Xd % 24 # 32]>()
        .commit_trim::<m![Xd % 24]>()
        .commit();
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![T2, Dt, Xp]> =
        unsafe { gathered.reshape() };
    ctx.sub
        .begin(gathered.view())
        .fetch::<m![T2, Dt, Xp / 32], m![Xp % 32]>()
        .collect::<m![T2, Dt, Xp / 32], m![Xp % 32]>()
        .to_trf()
}

/// Five rows of the down projection (a cluster's half of the columns) in one pass.
macro_rules! down_block_sums {
    ($ctx:ident, $w:ident, $x_trf:ident, $z:ident, $start:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = 5, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], 5, m![H / 30, H / 2 % 15 = 5 # 15, H % 2, L]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = 5, Dt, Xp]> =
            unsafe { packed.reshape() };
        $ctx.main
            .begin(packed.view())
            .fetch::<m![H / 2 % 15 = 5, Dt], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H / 2 % 15 = 5, Dt, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![H / 2 % 15 = 5, Dt, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![H / 2 % 15 = 5, Dt]>()
            .contract_lane::<m![H / 2 % 15 = 5, Dt], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .commit_trim::<m![Xp / 16]>()
            .commit_view($z.view_mut().tile::<m![H / 2 % 15], 5, m![H / 2 % 15 = 5 #{!} 15, Dt, Xp / 16]>($start));
    }};
}

/// Up to four rows of the down projection with the block scales applied INSIDE the pass: the
/// block sums leave the contraction, meet their scales (VRF, 8 KB = four rows) in the Vector Engine
/// and are reduced over the packet count; four partial sums per row are committed.
macro_rules! down_fused {
    ($ctx:ident, $w:ident, $x_trf:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { packed.reshape() };
        let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16]> = $ctx
            .sub
            .begin($scale.view().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, Dt, Xp / 16]>($start))
            .fetch::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .to_vrf();
        $ctx.main
            .begin(packed.view())
            .fetch::<m![H / 2 % 15 = $len, Dt], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![H / 2 % 15 = $len, Dt, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![H / 2 % 15 = $len, Dt]>()
            .contract_lane::<m![H / 2 % 15 = $len, Dt], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![Xp / 16]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Dt, m![H / 2 % 15 = $len], m![Xp / 16]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(DOWN_GAIN)
            .vector_widen_pad::<m![Xp / 16 # 8]>()
            .vector_final()
            .commit_trim::<m![Xp / 16]>()
            .commit_view($out.view_mut().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len #{!} 15, Xp / 16]>($start));
    }};
}

/// Up to four rows of block scales to the VRF (8 KB), then sum over the blocks of each column quarter.
#[allow(unused_macros)]
macro_rules! down_scale_tile {
    ($ctx:ident, $z:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let scale_vrf: VrfTensor<f32, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16]> = $ctx
            .sub
            .begin($scale.view().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, Dt, Xp / 16]>($start))
            .fetch::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .to_vrf();
        $ctx.main
            .begin($z.view().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, Dt, Xp / 16]>($start))
            .fetch::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .collect::<m![H / 2 % 15 = $len, Dt / 2], m![Dt % 2, Xp / 16]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![H / 2 % 15 = $len, Dt], m![Xp / 16]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
            .vector_intra_slice_reduce::<Dt, m![H / 2 % 15 = $len], m![Xp / 16]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(DOWN_GAIN)
            .vector_widen_pad::<m![Xp / 16 # 8]>()
            .vector_final()
            .commit_trim::<m![Xp / 16]>()
            .commit_view($out.view_mut().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len #{!} 15, Xp / 16]>($start));
    }};
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
) -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>, DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>) {
    let (x_trf, residual_pieces) = normalize_quantize(ctx, residual, pre_ff_rms_weight);
    // The input was loaded as 32 real copies of eight 480-column pieces: copy 0 of cluster 0 IS the
    // residual spread over the eight reducing slices, byte for byte (real data viewed as padded).
    let residual_spread: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> =
        unsafe { residual_pieces.reshape() };

    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> =
        up_weight_scale.to_dm(&mut ctx.tdma);
    let up_scale: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp / 16]> =
        unsafe { up_scale.reshape() };
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> =
        gate_weight_scale.to_dm(&mut ctx.tdma);
    let gate_scale: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp / 16]> =
        unsafe { gate_scale.reshape() };

    let mut up: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xp / 16]> = DmTensor::new();
    let mut gate: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30, Xp / 16]> = DmTensor::new();

    let up_z = block_sums_all!(ctx, up_weight_packed, x_trf);
    scale_tile!(ctx, up_z, up_scale, up, 0, 8);
    scale_tile!(ctx, up_z, up_scale, up, 8, 8);
    scale_tile!(ctx, up_z, up_scale, up, 16, 8);
    scale_tile!(ctx, up_z, up_scale, up, 24, 6);
    let gate_z = block_sums_all!(ctx, gate_weight_packed, x_trf);
    scale_tile!(ctx, gate_z, gate_scale, gate, 0, 8);
    scale_tile!(ctx, gate_z, gate_scale, gate, 8, 8);
    scale_tile!(ctx, gate_z, gate_scale, gate, 16, 8);
    scale_tile!(ctx, gate_z, gate_scale, gate, 24, 6);

    let up = fold4(ctx, &up);
    let gate = fold4_scaled(ctx, &gate, gate_global_scale);

    let up = regroup(ctx, &up);
    let gate = regroup(ctx, &gate);
    let x = geglu(ctx, &up, &gate);
    let x_trf = quantize_gather(ctx, x);

    let down_scale: DmTensor<f8e4m3, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, L / 16 % 480]> =
        down_weight_scale.to_dm(&mut ctx.tdma);
    let down_scale: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, Dt, Xp / 16]> =
        unsafe { down_scale.reshape() };

    let mut partial: DmTensor<f32, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, Xp / 16]> = DmTensor::new();
    down_fused!(ctx, down_weight_packed, x_trf, down_scale, partial, 0, 4);
    down_fused!(ctx, down_weight_packed, x_trf, down_scale, partial, 4, 4);
    down_fused!(ctx, down_weight_packed, x_trf, down_scale, partial, 8, 4);
    down_fused!(ctx, down_weight_packed, x_trf, down_scale, partial, 12, 3);

    // Sum the four column quarters while four neighbouring slices meet (rows interleaved two ways, so the pair is 30 consecutive rows).
    let partial: DmTensor<f32, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, Xp / 16]> = unsafe { partial.reshape() };
    let partial: DmTensor<f32, Chip, UpGateClusters, m![H / 30, 1 # 2], m![H % 30 # 64]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![H / 2 % 15], m![Xp / 16 # 8]>()
        .switch::<m![H / 30, 1 # 2], m![H / 2 % 15, H % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![H / 2 % 15, H % 2], m![Xp / 16 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Xp / 16]>()
        .vector_intra_slice_reduce::<Xp, m![H / 2 % 15, H % 2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .transpose::<m![H / 2 % 15], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    // The two clusters' partial results meet in HBM and come back divided over the eight slices the
    // post-norm reduces on; the add and both remaining global scales ride one pass there.
    // A scattered store of 120-byte pieces is slow (unaligned tails: 4.5K device cycles), so every
    // slice's 30 values sit at the head of a 256-byte element and the store is block-aligned; only
    // the heads come back. (An HBM tensor cannot end in padding: the pad words are a real axis.)
    let partial: DmTensor<f32, Chip, m![C2], m![Hg, 1 # 2], m![Pw]> = unsafe { partial.reshape() };
    let partial: HbmTensor<f32, Chip, m![C2, Hg, Pw]> = partial.to_hbm(&mut ctx.tdma);
    let partial: DmTensor<f32, Chip, Cluster, m![1 # 32, Hg / 16], m![C2, Hg % 16, Pw = 30]> = partial
        .view()
        .tile::<m![Pw], 30, m![C2, Hg, Pw = 30 # 64]>(0)
        .to_dm(&mut ctx.tdma);
    let partial: DmTensor<f32, Chip, Cluster, ReducingSlices, m![C2, H % 480]> = unsafe { partial.reshape() };

    let down_global_scale: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        unsafe { down_global_scale.reshape() };
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let up_global_scale: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        unsafe { up_global_scale.reshape() };
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let out: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![H / 4 % 120, C2], m![H % 4]>()
        .collect::<m![H / 4 % 120, C2], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &up_global_scale_vrf)
        .vector_intra_slice_reduce::<C2, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![H % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    (out, residual_spread)
}

/// `rmsnorm::normalize_spread` for an input that already sits on the eight reducing slices (the
/// shared one takes a whole vector in one slice and spreads it with a DMA copy first).
pub(crate) fn normalize_spread_in_place(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
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
    let reduced_mean_square: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(reduced_mean_square.view())
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
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
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
        .commit()
}

/// Post-norm, residual add and layer gate: (x / rms * w + r) * g in ONE pass, rounded to bf16 once at
/// the end (the three-pass form rounded after each step).
/// `rmsnorm::normalize_spread` for an input that already sits on the eight reducing slices (the
/// shared one takes a whole vector in one slice and spreads it with a DMA copy first).
pub(crate) fn normalize_add_gate_in_place(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_dm: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
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
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = layer_scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
