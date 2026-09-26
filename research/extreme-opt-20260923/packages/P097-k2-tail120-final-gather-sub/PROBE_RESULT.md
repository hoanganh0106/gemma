# P097 final gather Sub probe

P097 moved only the final `p` gather producer from Main to Sub while preserving the `Broadcast1` 32x8 geometry.

The SDK 0.8.1 build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. Final gather placement on Sub is incompatible with the downstream ownership/dataflow. P097 is rejected before Arena.
