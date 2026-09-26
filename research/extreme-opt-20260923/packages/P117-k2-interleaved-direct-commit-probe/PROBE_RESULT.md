# P117 K2 interleaved direct commit probe

## Result

Following the legal Interleaved pattern, the probe removed the K2 Lq
reduction and used:

```text
contract_lane::<H%120, Lq#8>(Interleaved)
cast::<bf16, Lq#16>()
```

The static compiler accepted it at **21,471 cycles / 42 instructions**. P078
is **21,337 / 42**, so the direct Interleaved path is 134 cycles slower even
before accounting for the missing Lq accumulation. It is therefore rejected
as both a cost candidate and a semantics-complete K2 implementation.
