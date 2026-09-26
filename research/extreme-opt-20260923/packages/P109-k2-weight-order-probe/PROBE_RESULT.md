# P109 K2 weight order probe

## Hypothesis

Move the `weight.view().to_dm()` issuance after input materialization and
quantization to expose additional DMA/TU overlap.

## Result

The Furiosa SDK 0.8.1 static compile passed. The emitted schedule is:

```text
21337 cycles / 42 instructions
```

The weight DMA remains a `1936..15319` span of 13,383 cycles, and the total
makespan is byte-for-byte schedule-equivalent in length to P078. The compiler
does not expose a new overlap from source declaration order.

## Decision

Reject before runtime testing. Further progress requires changing the weight
data movement/layout or its dependency graph, not reordering the declaration.
