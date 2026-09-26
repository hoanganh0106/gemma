# Split family boundary

P106 reached graph initialization and failed at the ring32 gather with `gather src residue has no live target axis`.

P107 attempted to reduce each 60-row half directly before commit. SDK 0.8.1 rejected the commit with `Slice size must be preserved`.

The missing adaptation cannot be expressed as a partial-slice commit. Further split work requires a new full-size intermediate layout and consumer mapping; it is a separate architecture generation, not a local DMA refinement.
