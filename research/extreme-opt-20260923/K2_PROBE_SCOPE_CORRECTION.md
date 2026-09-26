# K2 probe scope correction

The authoritative K2 schedule for P078 is generated from `src/device/sliding/output31.rs`. The P078/P041 diff confirms the H/120, ring32, `sw`-on-Sub, and epsilon-hoist changes there.

Packages P091 through P100 were exploratory copies that modified `src/device/audio/projection.rs`; their compiler failures are retained as isolated experiments but are not K2 evidence and must not be used in candidate ranking.

Future K2 probes must modify `src/device/sliding/output31.rs` and regenerate the schedule before Arena testing.
