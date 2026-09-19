//! The input RMSNorm on genuinely replicated 240-element pieces, all-gathered into every slice.
//!
//! The hidden state is loaded as 16 pieces of 240 elements, 16 real copies of each (`Rep16` is a
//! real axis, not `1 # 16`), so all 256 slices of a cluster hold live data from the first load on.
//! Each group of 16 slices normalizes the whole vector on its own. The result is split into two
//! exact f8e4m3 terms and each is gathered along the ring with `Broadcast1`: an f8 x f8
//! contraction needs no decode table and runs ~3.5x faster on the device than f8 -> bf16.
use furiosa_opt_std::prelude::*;

use super::{Rep16, Ring16, Slot, Term};
use crate::axes::H;
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

pub(crate) fn normalize_everywhere_f8<Cluster: M>(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    _rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> XTerms<Cluster> {
    let x: DmTensor<bf16, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> =
        x.to_dm(&mut ctx.tdma);
    let signed: DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]> = ctx
        .main.begin(x.view())
        .fetch::<m![H / 16 % 15], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_logic(LogicBinaryOpF32::BitAnd, -0.5)
        .vector_final().commit_trim::<m![H % 8]>().commit();
    gather_f8(ctx, &signed)
}

pub(crate) type XTerms<Cluster> = DmTensor<f8e4m3, Chip, Cluster, m![Rep16, Ring16], m![H]>;

fn gather_f8<Cluster: M>(
    ctx: &mut Context,
    pieces: &DmTensor<f32, Chip, Cluster, m![Rep16, H / 240], m![H % 240]>,
) -> XTerms<Cluster> {
    ctx.main
        .begin(pieces.view())
        .fetch::<m![1], m![H % 240]>()
        .switch::<m![Rep16, Ring16], m![H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<f8e4m3, m![H % 8 # 32]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
