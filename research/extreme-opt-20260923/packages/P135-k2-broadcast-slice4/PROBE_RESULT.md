# P135 K2 broadcast slice4 probe

## Hypothesis

Use `Broadcast1 { slice1: 32, slice0: 4 }` for P122's mean gather to search
the intermediate topology between the current factor 1 and rejected factor 8.

## Gate 1 result

Rejected by MIR switch shape validation:

```text
Switch OutSlice mismatch: expected (32 # 64, H / 120 % 4), got Vr # 256
```

No schedule or runtime test was produced. Within P122's output shape, the
current `slice0: 1` is the only tested legal Broadcast1 factor.
