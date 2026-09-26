# P110 K2 contract128 packet probe

## Hypothesis

Use a larger Qs contraction grouping (`Qs/128 % 2`, `Qs%128`) while retaining
the single full H/120 weight owner, hoping to reduce contract bookkeeping.

## Result

The static compiler rejected the variant at the output commit:

```text
mir: OutPacket must be 1 or 2 flits (32 bytes each), got 128 bytes
```

The larger Qs grouping cannot reach the existing `z` commit layout. This is a
legality boundary, before schedule generation; no performance or runtime claim
is made.

## Decision

Reject `Qs=128` for the current H/120 output owner. The legal packet boundary
remains the P078 `Qs%64` contract grouping.
