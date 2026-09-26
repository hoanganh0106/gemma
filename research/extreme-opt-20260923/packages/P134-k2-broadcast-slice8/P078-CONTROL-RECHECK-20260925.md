# P078 control recheck — 2026-09-25

## Command

```text
CARGO_INCREMENTAL=0 cargo build --release --bin test_kernels
timeout 45s ./target/release/test_kernels
```

## Result

The source rebuild completed successfully with Furiosa SDK 0.8.1. Runtime
graph construction then failed before correctness or cycle collection with:

```text
gather src residue has no live target axis
```

P108 produced the same failure, and its default control source matches P078
apart from environment-gated probe code. Therefore this is a current baseline
graph/hardware-state failure, not evidence against the P108 column-split
mechanism. No Arena result is inferred.
