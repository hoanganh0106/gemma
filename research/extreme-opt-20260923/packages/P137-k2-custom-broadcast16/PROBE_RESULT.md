# P137 K2 custom broadcast16 probe

## Hypothesis

Use `CustomBroadcast { ring_size: 16 }` for the P122 mean gather, matching the
number of logical H/120 groups.

## Gate 1 result

Rejected by MIR topology validation:

```text
Switch ring size mismatch: the slot sweep computed 32, but 16 was asserted
```

The P122 source/output mapping requires a 32-slot switch ring despite the
logical group count. No schedule or runtime test was produced.
