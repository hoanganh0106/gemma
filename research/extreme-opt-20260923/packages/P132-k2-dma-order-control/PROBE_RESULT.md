# P132 K2 DMA order control

## Hypothesis

Issue weight DMA before the x DMA by changing source order, to shorten the
13,383-cycle weight transfer's position on the critical path.

## Result

Static compile passed, but the compiler produced the same schedule as P126:

```text
total: 20,495 cycles
x.to_dm: 1003..1936
weight.to_dm: 1936..15319
contract: 15319..16693
```

Changing source order did not change scheduler order or cost. No runtime or
Arena test was run.

## Interpretation

The current scheduler already chooses the x DMA first because it enables
quantisation while the weight DMA runs. This is not a source-order debt; the
DMA critical path is unchanged under this control.
