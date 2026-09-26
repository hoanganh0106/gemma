# P150 K2 Final Multiply Order Probe

## Hypothesis

Swap the two final vector multiply assignments so `inv_rms` uses `Mul0` and
`sw` uses `Mul1`, while preserving the same product and dataflow.

## Result

The SDK 0.8.1 compiler accepted the candidate and emitted:

```text
20,495 cycles / 37 instructions
```

This is identical to P122. The ALU assignment has no static benefit, so P122
remains preferred as the simpler proven ordering. No hardware submission was
made.
