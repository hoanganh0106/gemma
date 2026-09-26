# P105 K2 fetch order probe

P105 kept all DMA allocations, layouts, arithmetic, and gather topology unchanged, but evaluated the `rms_weight` vector fetch before the `scale` fetch.

The SDK 0.8.1 release build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The fetch order is coupled to the ownership/dataflow chain. P105 is rejected before Arena.
