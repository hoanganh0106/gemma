//! Everything `ops::sliding_project_qkv` runs on: the hidden state normalized on replicated
//! pieces and all-gathered into every slice, whole-row Q/K/V weights, and the head norms and
//! RoPE done per KV head on the cluster that projected the head.
use furiosa_opt_std::prelude::*;


use crate::axes::{Ds, Dummy2, Gs, Ns};

pub(crate) mod headnorm;
pub(crate) mod proj;
pub(crate) mod proj8;
pub(crate) mod rope;
pub(crate) mod xnorm;
pub(crate) mod xnorm8;

axes![Rep16 = 16, Ring4 = 4, Ring8 = 8, Ring16 = 16, HeadCopy4 = 4, RopeTable = 2, Term = 2];

/// The two clusters named by KV head: cluster 0 owns heads 0..3, cluster 1 heads 4..7.
/// `Qs = Ns*Gs*Ds` and `Ps = Ns*Ds` row-major, so `Qs / 2048 == Ps / 1024 == Ns / 4`.
pub(crate) type HeadClusters = m![Ns / 4];
/// One KV head per slice (slices 0, 64, 128, 192 of a cluster).
pub(crate) type HeadSlices = m![Ns % 4, 1 # 64];
/// Q's 256 row slices of a cluster spelled by head: `Qs / 8 % 256 == (Ns % 4, Gs, Ds / 8)`.
pub(crate) type QueryRowsByHead = m![Ns % 4, Gs, Ds / 8];
/// K/V's 256 row slices of a cluster spelled by head: `Ps / 4 % 256 == (Ns % 4, Ds / 4)`.
pub(crate) type KvRowsByHead = m![Ns % 4, Ds / 4];
/// All 256 slices of a cluster as 16 replicas of a 16-slice ring.
pub(crate) type Everywhere = m![Rep16, Ring16];
/// A real copy in every head slice of both clusters, for tensors every head shares.
pub(crate) type HeadCopyClusters = m![Dummy2];
pub(crate) type HeadCopySlices = m![HeadCopy4, 1 # 64];

/// The normalized hidden state is multiplied by this power of two before it is split into f8
/// terms (and the projections divided by it again): it lifts the residual term x - f8(x), about
/// |x| / 16 .. |x| / 256, out of f8e4m3's subnormal range, where the device flushed it to zero.
pub(crate) const X_PRESCALE: f32 = 8.0;
