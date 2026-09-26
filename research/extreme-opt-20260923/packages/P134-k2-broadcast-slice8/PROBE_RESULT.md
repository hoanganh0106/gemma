# P134 K2 broadcast slice8 probe

## Hypothesis

Use the `Broadcast1 { slice1: 32, slice0: 8 }` topology seen in other device
families for P122's mean gather, instead of `slice0: 1`.

## Gate 1 result

Rejected by MIR switch shape validation:

```text
Switch OutSlice mismatch: expected (32, H / 120 % 8), got Vr # 256
```

The topology is not interchangeable with P122's `Vc/Tail` mapping. No
schedule or runtime test was produced.
