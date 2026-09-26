# P096 weight DMA order probe

P096 materialized `weight.to_dm(...)` before producing `input_trf`, with all layouts and arithmetic unchanged.

The SDK 0.8.1 build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The producer reordering breaks the downstream gather ownership/dataflow invariant. P096 is rejected before Arena.
