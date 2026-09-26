# P106 split weight and contract probe

P106 is the first full split prototype in the P078 family. It uses two HBM-to-DM 60-row weight transfers and two contracts committing into the low/high halves of the H/120 `z` tile; the tail/120 ring32 path is unchanged.

The Rust release build completed, but runtime graph initialization failed with
`gather src residue has no live target axis`. The split contract commit/layout
is not accepted by the downstream gather ownership.

The static compiler path was subsequently run independently and did emit a
schedule: **22,895 cycles / 56 instructions**. Its three long spans are
`1936..8902`, `8902..15868`, and `3206..16968`, showing that the two weight
DMAs and duplicated contract work add cost instead of hiding the 13,383-cycle
single-DMA critical span of P078. P106 remains rejected before Arena.
