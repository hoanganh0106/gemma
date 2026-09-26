# Extreme optimization campaign — 2026-09-23

User requests aggressive optimization of all three kernels while preserving every model input and complying with competition rules. No skipped input RMS weights, scales, RoPE, cache writes, fixture special cases, or changes to the official harness in a submitted package.

## Frozen starting point

Current source is byte-identical to the research baseline: 53/53 source hashes match. SDK 0.8.1, locked dependencies, official public three-seed fixture SHA256 338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359. Existing frozen baseline binary 021b9009886682add70e20ee3481bf5bcd9711b4b3535d1cba0821c46a37a39d, Arena 78017/78115, all outputs PASS. Starting static tuple 42539 / 24718 / 110520.

## Search streams

- K1: transfer the parallel V/RMS idea from A010 while retaining full weighted normalization and current activation representation; direct VRF; RoPE transfers. Historical A010's omission of input weights is forbidden.
- K2: start from the tested inverse-RMS direct-VRF delta; examine remaining scalar materialization, final store command, and distributed sumsq with minimal cross-cluster synchronization.
- K3: direct VRF, lookup provisioning reuse through valid tiling, scale DMA coalescing, partial-sum transfer, and dequantization/compute overlap.

Each agent writes only its isolated subdirectory; root owns integration and all Arena submissions. Main sources remain frozen until a validated integrated candidate is selected. Compiler schedules screen candidates; runtime correctness and whole-package medians determine selection. Additional numerical stress cases are diagnostic and do not replace the official harness.

## Validation and delivery

Freeze hashes, build all three device kernels, use full official harness, keep profiling and CPU policy identical, require all 15 output checks, measure promising candidates against the frozen parent. Preserve negative results. End with the best defensible tested package, exact source delta, measured score sensitivity, and unresolved limitations; do not claim a global optimum from finite experiments. Official submission is a separate external final-state decision because the contest evaluates the last submission.

## Completed selection

P005 (`packages/P005-fixed-rings`) passed two adjacent public control/candidate pairs with 15/15 checks in all four successful jobs. Paired geometric runtime improvements were 7.42% and 7.40%. Extended diagnostic ranges revealed two extreme cases outside tolerance in the frozen baseline; P005 produced the same maximum errors there. The reviewed P005 K1/K2/K3 sources are now promoted into the main workspace, normalized to its LF line endings, and a clean locked main-workspace build produced a binary byte-identical to the P005 Arena-tested executable. See `RESULTS.md` and `SEMANTIC_AUDIT.md`.

## P041 development — closing the gap to the other team

### Current standing (2026-09-23)

The other team reports `70.158 / 32.708 / 198.051` (K1 / K2 / K3). Our best measured tuple is `qkv69382` (K1) + P005 K2/K3, circa `79.3K / 39.6K / 216K`. P041 itself (`P041-k2-direct-tail-dma`, K2-only) measured `79.914 / 38.140 / 216.460` at Arena 83329 — K2 improved over P005 (39.6K → 38.1K) but still ~14% behind the other team's 32.7K, and the K1/K3 components are unchanged from the weaker tuple. **The gap is in all three kernels, so P041 alone is not sufficient; it is the K2 backbone of the next integrated candidate.**

### What "parallel flow" means here

The RNGD chip has 16 slices per cluster across 2 clusters. Each kernel already runs data-parallel over those 16 slices. "Running in parallel" therefore cannot mean inter-kernel concurrency (the host harness calls `sliding_project_qkv` → `sliding_attention_output` → `decoder_feedforward` strictly in sequence, one token at a time). The achievable parallelism is intra-kernel:

- **Cluster-level**: `OutputClusters = H/1920` splits the 3840 output rows across the two clusters so each reads half the weight in parallel (see `layout.rs`). P041 already exploits this.
- **Main / Sub TU overlap**: the two tensor-unit contexts (`device.main`, `device.sub`) can issue independently; a Sub command that depends on nothing can be handed off early ("paid off the critical chain") while Main or DMA do other work.
- **DMA / compute overlap**: HBM→DM loads (`to_dm`) can run while a prior vector/contract pass executes, if their tensors are independent allocations (P041 deliberately gives `residual_dm`, `scale_dm`, `rms_dm` separate pools to break version-chain dependencies).

The remaining cycles are critical-path stalls, not idle parallelism we forgot to turn on.

### P041 K2 tail — where the cycles are

`output31.rs::project_normalize_add` reshapes the tail from 4 serial Main/Sub commands (device-trace 4,445 cycles idle) to:
1. sum-of-squares (1,063) → 2. ring-16 reduce + eps + sqrt (661) → 3. `to_vrf` of rms (439) → 4. final `z * s / rms * w + r` (2,276).

The 2026-09-23 P041delta removed the `DivF` from the final pass (computes `1/rms` instead, so the divider runs 8× not 240×) and precomputed `sw = s * w` inside the cross-cluster sync wait. That cut the tail but the final pass is still 2,276 cycles and dominates. Open questions for further K2 parallel flow:

- **P054 (proposed)**: overlap the `sum-of-squares` Main pass with the `contract_tile` output DMA (`z.to_dm`) and the `scale_dm`/`rms_dm` Sub fetches, which currently serialise behind the projection's `z` commit. The `sw` precompute already runs in the sync wait; extend that free window to cover the `mean` ring-gather too if its tensors are version-independent.
- **P055 (proposed)**: fuse the `inv_rms` compute (eps + sqrt + div) into the same pass that produces `mean` by staging `EPS_SCALED` add before the gather — trades one extra VRF staging for removing the separate `inv_rms` `to_vrf` round trip (439 cycles in the original, residual after P041).

### K3 warm-up stall — the clearest parallel-flow win

`ffn7.rs::normalize_quantize` opens with an explicit `warm` Sub command whose only purpose is to pay a ~10× first-command issuer stall (2,696 device cycles for a 267-cycle `StoVrf`) off the critical chain. The DMA then idles 2,464 cycles between the norm-weight load and the decode-table load. This is a known fixed tax on every K3 invocation (~2.5K cycles ≈ 1.1% of K3's 216K). The other team's 198K K3 implies they either avoided this tax or shortened it.

- **P056 (proposed)**: move the `warm` issue earlier (immediately after kernel entry, before the first HBM load) so the issuer stall hides behind the `x.to_dm` latency instead of before the decode-table load; or replace the no-op `warm` with a real first dependency-free Sub pass (e.g. the `rms_weight.to_dm` fetch_cast) so the stall buys useful work. Target: recover most of the 2,464 idle DMA cycles.

### K1 weighted-norm overlap

`qkv69382::normalize_native_input` keeps `input_rms_weight` live and does weighted RMSNorm before the one-term FP8 quantization. The other team's 70.1K K1 is ~12% under our 79.3K. P047–P053 already isolated several K1 ideas on the P041 base (P024 head-RMS Main-to-VRF, P050 A010 parallel-V, P051/P052 direct RoPE broadcast/vscale, P053 native scale) but none produced a repeatable Arena win over qkv69382. The untested parallel-flow angle:

- **P057 (proposed)**: overlap `normalize_native_input`'s RMS reduction (currently a serial Sub `to_vrf` after a Main `fetch_cast`/`collect`) with the Q/K/V weight `to_dm` loads by issuing the weight DMAs at kernel entry and letting the RMS Sub pass run on the Sub TU while Main streams weights. Requires the weight DMAs to not advance the `x`/RMS pool version chain (use separate allocations as P041 does for K2).

### Validation gate for the next batch

Each proposed package (P054–P057) writes only its isolated subdirectory; root integrates and Arena-submits. Freeze source hashes, build all three kernels, require 15/15 official-harness checks, and compare against the frozen P041 parent with paired adjacent jobs (the within-job K2 variance of 17.6% in `DIAG_K2_WITHINJOB_C01_20260917` means single-run deltas below ~3K cycles on K2 are inconclusive — use interleaved paired samples). Preserve negative results. Do not promote a component that only wins inside noise.
