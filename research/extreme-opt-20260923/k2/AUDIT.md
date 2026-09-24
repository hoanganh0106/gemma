# K2 isolated optimization campaign

## Invariants

All candidates start from the frozen `../baseline` Cargo.toml, Cargo.lock, rust-toolchain.toml and src. Only `src/device/sliding/output31.rs` is varied. Root source is untouched. SDK remains 0.8.1. Public op signature, input tensors, output allocation and residual update are unchanged.

Every candidate preserves the output projection, per-channel weight scales, RMSNorm weights, epsilon and residual addition. No fixture data is embedded. No fixture shape/seed recognition, timing substitution or CPU-affinity changes are introduced.

The parent kernel already uses two FP8 activation levels with scale 16, then a BF16 projection boundary. All candidates retain both levels and this BF16 boundary. Direct VRF candidates preserve each FP32 operation, operand order and physical mapping. Retiling candidates change the FP32 summation tree and therefore require hardware correctness validation.

## Candidate families

- C01: previously validated inverse-RMS direct VRF, reproduced from the frozen source.
- C02: C01 plus direct Main SW result to VRF, eliminating a second DM round trip. Arithmetic is identical.
- C03: C02 plus removal of the final identity reshape. `TailDm<bf16>` already has cluster type `Vc = m![1 # 2]`; source and target types are identical.
- C04: C03 with the SW vector pipeline on Sub. Same operands, stage order and physical layout; different context reservation.
- C05: C04 with inverse-RMS vector pipeline on Sub.
- C06/C07/C09/C10: 240/960/120/1920 values per tail group. All 3840 logical H values participate; sum order and resource pressure change.
- C08: local BF16 projection partitions remain on their producing cluster. Each cluster computes its local contribution divided by global H; only that scalar is exchanged. Global mean is local + peer; epsilon is added to peer before the sum to fit one FMA per pipeline. The mathematical function is preserved, with changed FP32 grouping. Every row receives global inverse RMS before its weighted residual update. Compilation and runtime validation are required before relying on mapping correctness.
- C11: use the total sum rather than mean in RMS normalization. Scale epsilon by H, and multiply SW by sqrt(H), cancelling the removed division by H mathematically. FP32 reassociation changes.
- C12: compute `1 / sqrt(t)` using divider argument swapping rather than `sqrt(t) / t`. No native rsqrt exists in the installed SDK. Hardware support and numerical behavior must be checked.
- C13: change projection slice tile from 120 rows x 256 columns to 60 x 512. This keeps all 256 slices active, keeps each slice's FP8 weight payload at 30 KiB, and doubles contiguous HBM bytes per row. It preserves all Qs terms and both quantization levels, but changes FP32 reduction grouping. Its input DMA moves complete Gs=2, Ds=256 chunks for each Ns group before a physical-order reshape.
- C14: explicitly gather the eight H-group partials with `CustomBroadcast { ring_size: 8 }`, reduce those eight time values inside each slice, then run the unchanged epsilon/sqrt/div scalar pass. This targets reduction topology; it does not approximate reciprocal square root.
- C15/C16: C14 with 16 groups of 240 / 32 groups of 120 values and matching ring size.
- C17: repair C08's unsupported non-inner H reduction by gathering its 16 strided partials over the 256-slice ring, then using intra-slice sum before the scalar cluster exchange.
- C18/C19: start from C15, load each x block once per cluster into actual initialized source slices, then replicate with Switch routing. C18 fuses F32 routing into each of the two existing quantization passes. C19 routes BF16 once in a separate pass and keeps the original quantization function unchanged.
- C20/C21: substitute fixed Broadcast1 for the padded-input x replication. Both are rejected by the compiler's mapping implementation (`modulo must divide the size`); the experiments do not relabel uninitialized padded slices as valid data.
- C22: substitute fixed Broadcast1 for the RMS all-gather of sixteen real scalar partials. This topology compiles and removes the routing-table DMA instruction while retaining all arithmetic.

## Final static selection

**C22-fixed-rms-ring16: 22,317 cycles / 42 instructions** is the final candidate selected for the parent's integrated hardware validation. It reduces the frozen 24,718-cycle baseline by 2,401 cycles (9.7136%). Exact hashes, dependency identity, one-file source-difference assertion, and full diff are recorded in `selection.json` and `candidates/C22-fixed-rms-ring16/source.diff`. All 22 candidates have completed compilation attempts; none remains queued by this subtask.

## Promising topology result

C14 compiled at **22,400 cycles / 43 instructions**, down from C03's 24,409. It replaces the 2,579-cycle implicit inter-slice reduction/scalar pass with a 289-cycle explicit gather+sum and the unchanged 281-cycle scalar sqrt/div pass. See `candidates/C14-ring8/SEMANTICS.md` for the arithmetic and mapping audit. C15 combines this with the 240-value tail and compiles at **22,317 / 43**. Neither static result is a runtime speed or correctness claim.

## Concrete negative evidence

C13 compiled at 24,850 cycles, slower than C03's 24,409. Weight DMA rose from 13,383 to 13,512 cycles (utilization 0.8642 to 0.8560); replicated x DMA rose from 933 to 1,318 cycles, although contraction fell from 1,374 to 1,298. Simply doubling the contiguous weight segment does not improve this layout, because input replication and DMA behavior offset the compute gain.

C06's 240-value tail reduces total static time to 24,318 cycles, but its inverse RMS instruction still lasts exactly 2,579 cycles, the same as the 480-value tail. Standalone sqrt should not be blamed for the whole duration; this motivates C14's explicit reduction topology.

C17 compiles at 23,794 / 43. Its scalar exchange incurs the inter-cluster wait, and its strided local-row final HBM store costs 2,040 cycles versus 648 in C14. The local-row scalar gather costs 1,031 cycles versus 289 in C14. Thus removing the full-vector HBM hop alone does not beat the improved cluster-0 tail.

C10's 1920-value tail fails allocation with `Cannot find evict target for Vrf (required: AllocRequest { wait_targets: [], size: 3932160 })`. C11's sum-based algebra compiles at 24,410 / 41, the same schedule length as its direct-VRF C04 parent, so it offers no static reason to accept changed FP32 association.

## Measurement status and reproducibility

`ledger.json` is generated from every immutable candidate source and emitted schedule. Compiler cycles are only a screening metric. Runtime correctness and speed decisions belong to the parent's frozen all-three-kernel packages.

All new compile entry points use `flock /tmp/furiosa-extreme-20260923.lock` around the entire cargo-furiosa-opt invocation. Early C01/C02 were compiled before common serialization was adopted; selected candidates must be recompiled under the common lock. C03's initial compile was interrupted while replacing the old launcher; its valid rerun uses its own isolated full source tree.

No candidate has been promoted to root source by this subtask.
