# P141 K2 interleaved outer32 family

## Hypothesis

Make the contract outer tile finer (`Qs / 32` with `Qs % 32`) so the Lq axis
introduced by `LaneMode::Interleaved` can divide contract time.

## Gate 1 result

Rejected by the same time divisibility invariant:

```text
OutTime (H % 120, Lq) must divide
Time (H % 120, Qs / 32 % 8)
```

Changing outer packet width does not expose Lq as a valid contract time axis.
Interleaved would require a different lane-to-time ownership mapping rather
than a local outer-tile change. No schedule or runtime test was produced.
