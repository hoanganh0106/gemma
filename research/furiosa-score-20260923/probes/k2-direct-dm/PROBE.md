# K2 direct SRAM redistribution feasibility probe

Prepared 2026-09-23 from the current working tree. This directory began as an isolated copy of Cargo.toml, rust-toolchain.toml, and src only. No main source files were changed. Exact K2 compile passed on SDK 0.8.1; hardware correctness and performance remain unmeasured.

SDK API confirmed by reading installed furiosa-opt-std 0.8.1 sources in `/home/hoanganh/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/furiosa-opt-std-0.8.1/`:

- `src/tensor/memory.rs:957`: `DmTensor::to_dm<Chip2, Cluster2, Slice2, Element2>` supports explicit mapping changes subject to layout checks.
- `src/tensor/memory/redistribute.rs:585`: deferred `cluster_swap().to_dm_view` accepts a destination Slice mapping and a tiled destination Element view.
- `cluster_swap` swaps exactly two real clusters and requires live source clusters.

The old K2 stores 3840 projected BF16 values to an HBM temporary, then reloads those values into cluster 0 for RMSNorm. This probe gathers each cluster's 1920 projected values into one local SRAM slice and appends the peer cluster's values using a fused cluster swap plus gather. Cluster 0 then contains the full H vector in order. The cluster-1 assembled vector has reverse half order and is explicitly discarded by converting its physical cluster position to padding. A final local SRAM redistribution places the live cluster-0 values into the unchanged H480 tail mapping.

The arithmetic, projection, FP8 decomposition, epsilon, RMS weights, output weights, residual, and final store are unchanged. Feasibility risks are: SDK lowering may reject a redistribution into an Element tile with Slice relayout; allocation/version dependencies may serialize the two writes; SRAM cross-cluster transfers may still require expensive synchronization. A successful compile proves only legal lowering. Runtime correctness requires the official current three-seed fixture, and a speed claim requires paired unpinned hardware measurements of the same frozen binary.

Primary API documentation: https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html and https://github.com/furiosa-ai/furiosa-opt/blob/main/CHANGES.md .

## Result

Command (WSL, run in this probe directory):

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule k2.json > compile.log 2>&1
```

Compilation passed: `Finished 1 compiled, 14 not matching ops::sliding_attention_output`.

- Whole-kernel schedule max end: **26,142 cycles**, **55 instructions**.
- Parent task's freshly measured SDK 0.8.1 baseline: **24,718 cycles**, **44 instructions**. Delta: **+1,424 cycles (+5.76%)**, 11 additional instructions. The old 22,843-cycle ledger uses a different compiled artifact and is not the comparison baseline.
- Local gather is 446 cycles (`16693..17139`), peer transfer 446 (`18751..19197`), and final local relayout 449 (`20797..21246`).
- Two internal `ExplicitSync` nodes remain/add: `17151..17751` and `19197..19797`. Both are followed by a 1000-cycle gap before the next SRAM DMA. Final `ExplicitSync` is `25542..26142`. Thus SRAM-only redistribution does **not** imply synchronization-free execution.
- Source SHA256: `736F112B1175F6CAE664E710A7BE438829DE6E960557628908229722D6B77EA0`.
- Schedule SHA256: `E8F9CCB0F87644C58171E5BC99933D782E663F6A9A4AF7B137D63E76A411F1AD`.

Decision: **retain as a feasible but statically slower research probe; do not promote**. No full test binary was built, no fixture/runtime correctness test was run, and no Arena job was submitted. The remaining promising direction is reducing the number of synchronization boundaries (for example local RMS plus one scalar peer exchange), rather than merely replacing HBM storage with multiple SRAM copies.
