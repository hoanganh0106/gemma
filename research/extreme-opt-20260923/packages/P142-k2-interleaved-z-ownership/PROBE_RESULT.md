# P142 K2 interleaved Z ownership probe

## Hypothesis

Give the Interleaved contract output an explicit `Lq` time axis in `Z`, rather
than forcing it into the Sequential output layout.

## Gate 1 result

The compiler rejects the contract before the new Z ownership can be checked:

```text
OutTime (H % 120, Lq) must divide
Time (H % 120, Qs / 32 % 8)
```

The existing outer tile still lacks a representable Lq time factor. A true
Interleaved Z family must change the contract mapping itself before output
ownership can be evaluated. No schedule or runtime test was produced.
