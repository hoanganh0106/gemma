# P088 Broadcast1 16x2 probe

Parent: P078-k2-tail120. Changed only the `mean` gather config from `Broadcast1 { slice1: 32, slice0: 1 }` to `{ slice1: 16, slice0: 2 }`.

Compiler gate: rejected with `Switch OutSlice mismatch`; the requested output is `Vr # 256`, while this factorization produces `(16 # 128, H / 120 % 2)`. The 32x1 factorization is required by the physical layout.
