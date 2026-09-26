# P161 direct output-owner consumer

## Mechanism probe

The projection `z` was kept in its `OutputClusters/Rows` owner and the RMS
consumer was allowed to infer its output mapping. The compiler accepted the
direct producer-to-consumer path far enough to expose only VRF operand mapping
errors.

Remaining errors require explicit `VrfTensor::reshape` Element mappings for
`scale`, `sw`, `residual`, and `inv_rms`; the final output mapping also needs
to be inferred before the HBM commit. This is a consumer mapping problem,
not the earlier `commit_view` ownership rejection.

No schedule or Arena binary was produced.

## Refinement result

Explicit producer-owner VRF reshapes and inferred commit Element mappings
removed the Rust type errors. The remaining switch mapping reaches the NPU
sequencer, which panics with:

```text
crates/furiosa-opt/npu-mapping-impl/src/sequencer/smapping.rs:627:
modulo must divide the size
```

This is a concrete mapper topology invariant for the direct consumer path;
the probe is not an Arena candidate yet.

Changing the switch input from `H / 1920` to `H / 120 % 16` and keeping the
required output `(H / 240 % 8, 32)` reproduces the same sequencer panic. The
direct consumer mapping therefore needs a different switch topology, not a
further modulo spelling change.

Historical P077's `Broadcast1 { slice1: 16 }` pattern was also tested. Its
direct-owner adaptation reports a 256/512 slice-size mismatch; balancing the
input and output sizes then reaches the same sequencer modulo panic. The
ring4/row60 topology cannot be transplanted into the direct OutputClusters
owner without a new redistribution mapping.
### CustomBroadcast ring-2 topology probes

- `m![1 # 32, Vr]` with `ring_size: 2`: MIR rejects the switch because input/output slice sizes are `256` and `1024`.
- `m![1 # 8, Vr]` with `ring_size: 2`: MIR computes slot sweep `256`, so ring size 2 is invalid.
- The same `m![1 # 8, Vr]` mapping with `ring_size: 256` passes the ring-size assertion but fails sequenceability with unconsumed `(H / 120 % 8, 1 # 16), 1`.
- No binary or Arena submission was produced from these probes. The source is restored to the prior Broadcast1 probe topology.
