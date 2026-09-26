# P139 K2 interleaved packet alignment

## Hypothesis

Adapt the Interleaved contract packet declaration from `1 # 8` to the
compiler-reported `Lq # 8` output packet, while retaining the P122 commit
mapping.

## Gate 1 result

The packet mismatch moved to the time axis:

```text
contract_lane (Interleaved): OutTime mismatch
Outer portion of OutTime must equal Time: expected H % 120, got (H % 120, Lq)
```

Interleaved changes both packet and time geometry; it cannot feed the P122
`z.view_mut()` without a new transpose/ownership family. No schedule or
runtime test was produced.
