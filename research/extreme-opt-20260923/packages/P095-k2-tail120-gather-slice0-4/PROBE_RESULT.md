# P095 gather slice0 probe

P095 changed only the final tail/120 `Broadcast1` gather from `slice0: 8` to `slice0: 4`.

The SDK 0.8.1 release build completed, but graph initialization failed before schedule generation with `gather src residue has no live target axis`. The reduced slice ownership is invalid for the downstream gather layout. P095 is rejected before Arena.
