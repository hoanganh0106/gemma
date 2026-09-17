
use furiosa_opt_std::prelude::*;

use crate::Chip;
use crate::axes::*;

pub(crate) type Cluster = m![1 # 2];
pub(crate) type Slice = m![1 # 256];

pub(crate) type Replicated = m![Dummy256];


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
pub(crate) type QkvColumns = m![Qs / 32 % 64, H / 960];
/// Fans the normalized hidden state out to the 64 row groups that contract it. Each slice
/// already holds its quarter of H, so this only replicates -- the DMA engine will not lower a
/// replication whose slice mapping still has a live axis beside it, which the Switch ring will.
pub(crate) fn broadcast_qkv_hidden<Cluster: M>(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, m![1 # 64, H / 960], m![H % 960]>,
) -> DmTensor<bf16, Chip, Cluster, QkvColumns, m![H % 960]> {
    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![H % 960]>()
        .switch::<QkvColumns, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 16 % 60], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}

/// The same 256 slices named by the K/V row count.
pub(crate) type KvColumns = m![Ps / 16 % 64, H / 960];
/// Where the 8 column partials of a Q row group meet.
pub(crate) type QueryRowsReduced = m![Qs / 32 % 64, 1 # 4];
/// The same for a K or V row group, which has half as many rows.
pub(crate) type KeyValueRowsReduced = m![Ps / 16 % 64, 1 # 4];

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
