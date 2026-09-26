# P098 gather 64x4 probe

P098 kept the total gather payload at 256 and changed the final Broadcast1 decomposition from `slice1: 32, slice0: 8` to `slice1: 64, slice0: 4`.

The SDK 0.8.1 build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. This decomposition is incompatible with the downstream ownership/layout. P098 is rejected before Arena.
