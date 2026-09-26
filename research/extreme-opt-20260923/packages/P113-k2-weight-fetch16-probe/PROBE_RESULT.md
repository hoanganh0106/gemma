# P113 K2 weight fetch16 probe

## Hypothesis

Use a finer weight fetch/collect layout (`Qs/16%16`, `Qs%32`) while keeping
the P078 `Qs%64` contract and output owner.

## Result

The SDK rejected the layout before schedule generation:

```text
contract_outer: OutPacket's packed cells must equal the innermost 2 cells of Time
```

The contract mapping is coupled to the fetched time/packet layout. A fetch
layout change cannot be evaluated as a local DMA optimization without changing
the contract output mapping, which is already bounded by the flit constraints.
