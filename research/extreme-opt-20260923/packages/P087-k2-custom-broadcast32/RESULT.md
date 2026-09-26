# P087 custom broadcast32 probe

Parent: P078-k2-tail120. Replaced the validated `Broadcast1` gather with `CustomBroadcast { ring_size: 32 }`, using the compiler-reported physical ring size.

Compiler gate: PASS, but schedule worsened from 21,337 cycles / 42 instructions to 21,456 cycles / 43 instructions. Reject before runtime; the specialized Broadcast1 lowering is cheaper than the generic custom path.
