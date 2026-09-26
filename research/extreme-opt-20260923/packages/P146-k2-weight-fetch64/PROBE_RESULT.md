# P146 K2 Weight Fetch 64 Probe

## Hypothesis

Use 64-column fetch and collect tiles before the existing contraction, instead
of the proven 32-column flits, to reduce fetch/collect setup.

## Result

Rejected by SDK 0.8.1 MIR validation:

```text
Collect output packet must be exactly 32 bytes (one flit):
64 elements = 64 bytes
```

The weight fetch packet is hardware-limited to one 32-byte flit for this
`f8e4m3` path. A 64-element packet cannot reach `commit_view`, so the larger
tile cannot be used to reduce setup. No schedule was emitted.
