//! Everything `ops::sliding_project_qkv` runs on: the hidden state normalized on replicated
//! pieces and all-gathered into every slice, whole Q/K/V weight rows strided 8-way over the row
//! slices, every small load chained in one pool tensor (`pool`, which also parks K's head so that
//! the RoPE rows' reload is ordered after K's contraction), and the head norms and RoPE done per
//! KV head on the cluster that projected the head.
use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Dummy2, Gs, Ns, Ps, Qs};

pub(crate) mod headnorm;
pub(crate) mod pool;
pub(crate) mod proj8;
pub(crate) mod rope;
pub(crate) mod xnorm8;

axes![Rep16 = 16, Ring16 = 16, HeadCopy4 = 4, RopeTable = 2, Term = 2, Pool = 10, Kv = 2, Slot = 2];

/// Whole weight rows, STRIDED over the row slices of a head: the innermost slice axis is the low
/// digit of the row index (`Ds % 8`, `Ds % 4`), so consecutive HBM rows (3,840-byte, 256-aligned
/// runs) go to different slices and the DMA feeds several slices at once: measured 530-595
/// B/cycle against 490-510 for 8 (4) contiguous rows per slice. A slice holds rows 8 (4) apart;
/// `proj8` puts them back in order with one `InterTranspose` pass after the contraction. The
/// head axis stays OUTERMOST so the head gather remains a ring of 64.
pub(crate) type QueryRowSlices = m![Ns % 4, Gs, Ds / 64, Ds % 8];
/// K/V: 8 consecutive rows go to 8 different slices, like Q. Slice `(Ds / 32, Ds % 4, Ds / 4 % 2)`
/// holds rows `Ds / 8 % 4`; the sort swaps the `Ds % 4` slice sub-axis (stride 2) with the row axis.
pub(crate) type KeyValueRowSlices = m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2];

/// The two clusters named by KV head: cluster 0 owns heads 0..3, cluster 1 heads 4..7.
/// `Qs = Ns*Gs*Ds` and `Ps = Ns*Ds` row-major, so `Qs / 2048 == Ps / 1024 == Ns / 4`.
pub(crate) type HeadClusters = m![Ns / 4];
/// One KV head per slice (slices 0, 64, 128, 192 of a cluster).
pub(crate) type HeadSlices = m![Ns % 4, 1 # 64];
/// Q's 256 row slices of a cluster after the sort: 8 consecutive rows each, `(Ns % 4, Gs, Ds / 8)`.
pub(crate) type QueryRowsByHead = m![Ns % 4, Gs, Ds / 8];
/// K/V's 256 row slices of a cluster after the sort: 4 consecutive rows each, `(Ns % 4, Ds / 4)`.
pub(crate) type KvRowsByHead = m![Ns % 4, Ds / 4];
/// All 256 slices of a cluster as 16 replicas of a 16-slice ring.
pub(crate) type Everywhere = m![Rep16, Ring16];
/// A real copy in every head slice of both clusters, for tensors every head shares.
pub(crate) type HeadCopyClusters = m![Dummy2];
pub(crate) type HeadCopySlices = m![HeadCopy4, 1 # 64];

/// The normalized hidden state is multiplied by this power of two before it is split into f8
/// terms, and the projections divided by it again (both exact). It lifts the residual term
/// x - f8(x), which is |x| / 16 .. |x| / 256, out of f8e4m3's subnormal range (below 2^-6) for
/// |x| around 1; |x| up to 448 / 8 = 56 still fits the first term.
pub(crate) const X_PRESCALE: f32 = 8.0;
