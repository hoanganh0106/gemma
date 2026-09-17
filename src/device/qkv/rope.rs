
use furiosa_opt_std::prelude::*;

use crate::Chip;
use super::{HeadCopyClusters, HeadCopySlices, RopeTable};

type Cluster = HeadCopyClusters;
type Slice = HeadCopySlices;
use crate::axes::{Ds, E, Gs};

/// The cos row and the sin row of this position, side by side, a real copy in every head slice.
pub(crate) type RopeRows = DmTensor<bf16, Chip, Cluster, Slice, m![RopeTable, Ds]>;

/// The cos and sin rows of this position, a real copy in every head slice of both clusters.
///
/// A `dma_gather` lands in ONE slice of cluster 0 whatever its type claims (device: a `Dummy2`
/// cluster or replicated slices on the gather left cluster 1 / heads 1..3 with garbage), and
/// `dm_cluster_shuffle` only lowers the swap `[1, 0]`. HBM is the one place both clusters can
/// load real copies from, so the rows are parked there and loaded back into every head slice --
/// both rows in one tensor, so there is one store, one ExplicitSync and one load.
pub(crate) fn load_tables(
    ctx: &mut Context,
    rope_offset: &HbmTensor<i32, Chip, m![1]>,
    cos: &HbmTensor<bf16, Chip, m![E, Ds]>,
    sin: &HbmTensor<bf16, Chip, m![E, Ds]>,
) -> RopeRows {
    type One = DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![RopeTable = 1, Ds]>;
    let cos: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Ds]> = cos.dma_gather_scaled(rope_offset);
    let sin: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![Ds]> = sin.dma_gather_scaled(rope_offset);
    let cos: One = unsafe { cos.reshape() };
    let sin: One = unsafe { sin.reshape() };

    let mut rows: DmTensor<bf16, Chip, m![1 # 2], m![1 # 256], m![RopeTable, Ds]> = DmTensor::new();
    ctx.sub
        .begin(cos.view())
        .fetch::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .collect::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![RopeTable], 1, m![RopeTable = 1 #{!} 2, Ds]>(0));
    ctx.sub
        .begin(sin.view())
        .fetch::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .collect::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .commit_trim::<m![Ds % 16]>()
        .commit_view(rows.view_mut().tile::<m![RopeTable], 1, m![RopeTable = 1 #{!} 2, Ds]>(1));

    let rows: HbmTensor<bf16, Chip, m![RopeTable, Ds]> = rows.to_hbm(&mut ctx.tdma);
    rows.to_dm(&mut ctx.tdma)
}

/// Runs per slice, one KV head to a slice, on the cluster that projected the head.
pub(crate) fn apply_rope(
    ctx: &mut Context,
    q: &DmTensor<bf16, Chip, Cluster, Slice, m![Gs, Ds]>,
    k: &DmTensor<bf16, Chip, Cluster, Slice, m![Ds]>,
    rows: &RopeRows,
) -> (
    DmTensor<bf16, Chip, Cluster, Slice, m![Gs, Ds]>,
    DmTensor<bf16, Chip, Cluster, Slice, m![Ds]>,
) {
    let cos_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> = ctx
        .sub
        .begin(rows.view().tile::<m![RopeTable], 1, m![RopeTable = 1 # 2, Ds]>(0))
        .fetch::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> = ctx
        .sub
        .begin(rows.view().tile::<m![RopeTable], 1, m![RopeTable = 1 # 2, Ds]>(1))
        .fetch::<m![RopeTable = 1, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let first_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(0);
    let second_half_q = q.view().tile::<m![Ds], 128, m![Gs, Ds = 128 # 256]>(128);

    let mut rotate_half_q: DmTensor<bf16, Chip, Cluster, Slice, m![Gs, Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(128),
        );

    ctx.main
        .begin(second_half_q)
        .fetch::<m![Gs], m![Ds = 128]>()
        .collect::<m![Gs, Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(
            rotate_half_q
                .view_mut()
                .tile::<m![Ds], 128, m![Gs, Ds = 128 #{!} 256]>(0),
        );

    let first_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(0);
    let second_half_k = k.view().tile::<m![Ds], 128, m![Ds = 128 # 256]>(128);

    let mut rotate_half_k: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = DmTensor::new();

    ctx.main
        .begin(first_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(128));

    ctx.main
        .begin(second_half_k)
        .fetch::<m![1], m![Ds = 128]>()
        .collect::<m![Ds = 128 / 16], m![Ds = 128 % 16]>()
        .commit_trim::<m![Ds = 128 % 16]>()
        .commit_view(rotate_half_k.view_mut().tile::<m![Ds], 128, m![Ds = 128 #{!} 256]>(0));

    let q_cos: DmTensor<f32, Chip, Cluster, Slice, m![Gs, Ds]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin: DmTensor<f32, Chip, Cluster, Slice, m![Gs, Ds]> = ctx
        .main
        .begin(rotate_half_q.view())
        .fetch::<m![Gs, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Gs, Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let q_sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Gs, Ds]> = ctx
        .sub
        .begin(q_sin.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_q: DmTensor<bf16, Chip, Cluster, Slice, m![Gs, Ds]> = ctx
        .main
        .begin(q_cos.view())
        .fetch::<m![Gs, Ds / 8], m![Ds % 8]>()
        .collect::<m![Gs, Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &q_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_cos: DmTensor<f32, Chip, Cluster, Slice, m![Ds]> = ctx
        .main
        .begin(k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &cos_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin: DmTensor<f32, Chip, Cluster, Slice, m![Ds]> = ctx
        .main
        .begin(rotate_half_k.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &sin_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    let k_sin_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![Ds]> = ctx
        .sub
        .begin(k_sin.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();

    let result_k: DmTensor<bf16, Chip, Cluster, Slice, m![Ds]> = ctx
        .main
        .begin(k_cos.view())
        .fetch::<m![Ds / 8], m![Ds % 8]>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, &k_sin_vrf)
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit();

    (result_q, result_k)
}
