# P100 weight fetch 32 probe

P100 changed the weight fetch tile from `Aa / 16, Aa % 16` to `Aa / 32, Aa % 32`, leaving the collected and contracted layout unchanged.

The SDK 0.8.1 release build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The fetch shape is not compatible with the downstream ownership/dataflow. P100 is rejected before Arena.
