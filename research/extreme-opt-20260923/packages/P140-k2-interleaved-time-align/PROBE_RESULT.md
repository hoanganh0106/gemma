# P140 K2 interleaved time alignment

## Hypothesis

Adapt `contract_time` to the Interleaved lane's reported `(H % 120, Lq)`
time geometry and keep the matching `Lq # 8` output packet.

## Gate 1 result

Rejected by the contract divisibility rule:

```text
contract_time: OutTime (H % 120, Lq) must divide
Time (H % 120, Qs / 64 % 4)
```

The interleaved lane introduces an Lq axis that the existing contract time
factor cannot divide. It therefore requires a different outer/packet/time
tiling family, not a local declaration change. No schedule or runtime test
was produced.
