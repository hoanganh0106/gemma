
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub(crate) type Cluster = m![1 # 2];
pub(crate) type Slice = m![1 # 256];

pub(crate) type Replicated = m![Dummy256];

/// Hands each of the feed-forward's 128 row groups the half of the hidden state it
/// contracts. The DMA engine splits H across two slices; replicating those halves across the
/// row groups is the Switch Engine's job, because one transfer cannot do both.
pub(crate) fn broadcast_feedforward_hidden<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![1 # 128, H / 1920], m![H % 1920]>,
) -> DmTensor<bf16, Chip, Cluster, m![L / 60 % 128, H / 1920], m![H % 1920]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 1920]>()
        .switch::<m![L / 60 % 128, H / 1920], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16 % 120], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

/// The output projection's rows split over the chip's two clusters: cluster 0 owns rows
/// 0..1919, cluster 1 owns 1920..3839. The clusters load from HBM on separate paths, so each
/// reading half the weight halves the transfer the kernel is bound by.
pub(crate) type OutputClusters = m![H / 1920];

/// Q's rows over the chip's two clusters; K and V split the same way under their own row
/// count. Each cluster reads only its half of the weight, and the two read in parallel.
pub(crate) type QueryClusters = m![Qs / 2048];
pub(crate) type KeyValueClusters = m![Ps / 1024];

/// The hidden state as `sliding::projection` contracts it: within a cluster, 32 row groups by
/// 8 column groups, every slice holding the 480 columns it needs and nothing else.
pub(crate) type QkvColumns = m![Qs / 64 % 32, H / 480];
/// The same 256 slices named by the K/V row count.
pub(crate) type KvColumns = m![Ps / 32 % 32, H / 480];
/// Where the 8 column partials of a Q row group meet.
pub(crate) type QueryRowsReduced = m![Qs / 64 % 32, 1 # 8];
/// The same for a K or V row group, which has half as many rows.
pub(crate) type KeyValueRowsReduced = m![Ps / 32 % 32, 1 # 8];

pub(crate) fn broadcast_hidden(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![H]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![H]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![H]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();

    unsafe { x.reshape() }
}

/// One cluster's 16 row groups, each spread over the 16 slices holding one sixteenth of
/// the contracted columns, so all 256 slices of the cluster take part. `ops::sliding_attention_output`
/// loads x straight into this shape with the DMA engine; the Switch ring used to broadcast it.
pub(crate) type SlidingOutputColumns = m![H / 120 % 16, Qs / 256];

/// Where the sixteen column partials of a row group meet.
pub(crate) type SlidingOutputRows = m![H / 120 % 16, 1 # 16];

pub(crate) fn broadcast_full_heads(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, Slice, m![Qf]>,
) -> DmTensor<bf16, Chip, Cluster, Replicated, m![Qf]> {
    let x: DmTensor<bf16, Chip, Cluster, m![Dummy256], m![Qf]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![1], m![Qf]>()
        .switch::<m![Dummy256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qf / 16], m![Qf % 16]>()
        .commit_trim::<m![Qf % 16]>()
        .commit();

    unsafe { x.reshape() }
}
