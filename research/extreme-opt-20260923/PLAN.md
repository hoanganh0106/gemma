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
