# P104 explicit weight DM probe

P104 replaced the implicit weight `to_dm` with an explicit `WeightTile::new()` destination and `to_dm_view`, preserving the exact destination layout.

The SDK 0.8.1 release build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The explicit DMA destination changes the ownership/dataflow chain and is rejected before Arena.
