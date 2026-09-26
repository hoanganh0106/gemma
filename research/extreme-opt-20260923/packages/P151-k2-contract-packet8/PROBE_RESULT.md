# P151 K2 Contract Packet 8 Probe

## Hypothesis

Use a padded packet mapping in `contract_packet`:

```rust
.contract_packet::<m![1 # 8]>()
```

The goal was to see whether the contraction could retain the same flit work
with less packet normalization.

## Result

The SDK 0.8.1 compiler accepted the probe, but emitted exactly the baseline
static result:

```text
20,495 cycles / 37 instructions
```

The packet padding is normalized away by lowering and produces no improvement.
P122 remains the preferred source because its packet mapping is simpler. No
hardware submission was made.
