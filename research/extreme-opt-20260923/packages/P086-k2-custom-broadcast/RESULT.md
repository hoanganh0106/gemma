# P086 custom broadcast probe

Parent: P078-k2-tail120. Replaced the validated `Broadcast1 { slice1: 32, slice0: 1 }` gather with `CustomBroadcast { ring_size: 256 }`.

Compiler gate: rejected with `Switch ring size mismatch: the slot sweep computed 32, but 256 was asserted`. The P043 custom broadcast topology does not apply to P078's physical H/120 mapping.
