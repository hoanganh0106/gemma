# P138 K2 interleaved contract lane on P122

## Hypothesis

Use `LaneMode::Interleaved` in the P122 contract while preserving its existing
reduction and direct reshape consumer topology.

## Gate 1 result

Rejected at the contract output boundary:

```text
contract_lane (Interleaved): OutPacket mismatch
Expected: Lq # 8, got: 1 # 8
```

The interleaved lane changes the packet geometry required by `z.view_mut()`;
the existing transpose/commit path cannot consume it. No schedule or runtime
test was produced.
