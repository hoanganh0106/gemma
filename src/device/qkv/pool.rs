//! LEVER C: every small dependency-free bf16 load of the kernel (three row scales, two head-norm
//! weights, the parked RoPE rows) goes into a tile of ONE DM tensor. Tile writes of one tensor are
//! chained by the scheduler in PROGRAM ORDER (each `to_dm_view` depends on the previous one), so
//! the DMA queue holds them back to back in the order written here, and all of them precede the
//! consumer of the LAST tile written.
use furiosa_opt_std::prelude::*;

use super::{Dummy2, HeadCopy4, HeadCopyClusters, HeadCopySlices, Pool, RopeTable};
use crate::Chip;
use crate::axes::{Ds, Ps, Qs};

pub(crate) type SmallPool = DmTensor<bf16, Chip, HeadCopyClusters, HeadCopySlices, m![Pool, Ds]>;
pub(crate) type Vrf1 = VrfTensor<f32, Chip, HeadCopyClusters, HeadCopySlices, m![Ds]>;
pub(crate) type Vrf2 = VrfTensor<f32, Chip, HeadCopyClusters, HeadCopySlices, m![Pool = 2, Ds]>;

pub(crate) const V_SCALE: usize = 0;
pub(crate) const K_SCALE: usize = 1;
pub(crate) const K_NORM: usize = 2;
pub(crate) const Q_NORM: usize = 3;
pub(crate) const Q_SCALE: usize = 4;
pub(crate) const ROPE: usize = 6;

fn load_per_head(ctx: &mut Context, pool: &mut SmallPool, scale: &HbmTensor<bf16, Chip, m![Ps]>, index: usize) {
    // Ps = (Ns, Ds) row-major and Ns = (cluster, head-in-cluster): relabel only.
    let scale = unsafe { scale.view().reshape::<Chip, m![Dummy2, HeadCopy4, Pool = 1, Ds]>() };
    scale.to_dm_view(&mut ctx.tdma, pool.view_mut().tile::<m![Pool], 1, m![Pool = 1 #{!} 8, Ds]>(index));
}

fn load_shared(ctx: &mut Context, pool: &mut SmallPool, weight: &HbmTensor<bf16, Chip, m![Ds]>, index: usize) {
    let weight = unsafe { weight.view().reshape::<Chip, m![Pool = 1, Ds]>() };
    weight.to_dm_view(&mut ctx.tdma, pool.view_mut().tile::<m![Pool], 1, m![Pool = 1 #{!} 8, Ds]>(index));
}

/// `order` = the chain order of the six loads (program order of the tile writes).
#[expect(clippy::too_many_arguments)]
pub(crate) fn load(
    ctx: &mut Context,
    q_scale: &HbmTensor<bf16, Chip, m![Qs]>,
    k_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    v_scale: &HbmTensor<bf16, Chip, m![Ps]>,
    q_norm: &HbmTensor<bf16, Chip, m![Ds]>,
    k_norm: &HbmTensor<bf16, Chip, m![Ds]>,
    rope_rows: &HbmTensor<bf16, Chip, m![RopeTable, Ds]>,
) -> SmallPool {
    let mut pool: SmallPool = DmTensor::new();
    load_per_head(ctx, &mut pool, v_scale, V_SCALE);
    load_per_head(ctx, &mut pool, k_scale, K_SCALE);
    load_shared(ctx, &mut pool, k_norm, K_NORM);
    {
        let rows = unsafe { rope_rows.view().reshape::<Chip, m![Pool = 2, Ds]>() };
        rows.to_dm_view(&mut ctx.tdma, pool.view_mut().tile::<m![Pool], 2, m![Pool = 2 #{!} 8, Ds]>(ROPE));
    }
    load_shared(ctx, &mut pool, q_norm, Q_NORM);
    {
        // Qs = (Ns, Gs, Ds): the two query groups of a head are tiles 4 and 5.
        let scale = unsafe { q_scale.view().reshape::<Chip, m![Dummy2, HeadCopy4, Pool = 2, Ds]>() };
        scale.to_dm_view(&mut ctx.tdma, pool.view_mut().tile::<m![Pool], 2, m![Pool = 2 #{!} 8, Ds]>(Q_SCALE));
    }
    pool
}

pub(crate) fn vrf1(ctx: &mut Context, pool: &SmallPool, index: usize) -> Vrf1 {
    ctx.sub
        .begin(pool.view().tile::<m![Pool], 1, m![Pool = 1 # 8, Ds]>(index))
        .fetch::<m![Pool = 1, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf()
}

pub(crate) fn vrf2(ctx: &mut Context, pool: &SmallPool, index: usize) -> Vrf2 {
    ctx.sub
        .begin(pool.view().tile::<m![Pool], 2, m![Pool = 2 # 8, Ds]>(index))
        .fetch::<m![Pool = 2, Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Pool = 2, Ds / 8], m![Ds % 8]>()
        .to_vrf()
}
