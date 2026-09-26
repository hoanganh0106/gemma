# P103 K2 DMA order probe

P103 is scoped to `sliding/output31.rs` and queues the `rms_weight.to_dm()` allocation before residual and scale DMA allocations.

The SDK 0.8.1 release build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The DMA declaration order is coupled to the gather ownership/dataflow. P103 is rejected before Arena.
