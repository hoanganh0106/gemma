# K3 compiler probe ledger — 2026-09-23

All probe source files are immutable snapshots of `src/device/shared/ffn7.rs`, based on `../baseline`. Only `work/src/device/shared/ffn7.rs` is overwritten between sequential compiles. No main source or harness changes are made here. Use a candidate's `ffn7.rs`, not the mutable work copy.

## Baseline and method

- SDK `furiosa-opt-std = 0.8.1`, copied lock and toolchain.
- Baseline current-source schedule: 110,520 cycles, 170 instructions.
- `compile.sh NAME...` runs exact `ops::decoder_feedforward`, `CARGO_INCREMENTAL=0`, and whole-command `flock /tmp/furiosa-extreme-20260923.lock`. Early runs used the shared root target; remaining probes use the prepared Linux cache `/tmp/furiosa-extreme-20260923-target` to avoid Windows filesystem overhead. `compile-extra.sh` has its own `work-extra` source copy and the same lock.
- v01 completed before the common outer lock was introduced. It needs locked reconfirmation if selected. The first v02 attempt was interrupted during the transition and produced no schedule; that attempt is not evidence.
- At the Linux-cache transition, v06 and v12 were waiting for the lock and had not started a compiler. Their drivers were stopped and these unstarted probes queued again on the Linux cache; completed schedules were retained.
- `summarize.py` produces `ledger.json`, hashes and static deltas. Static cycles are compiler estimates, not hardware timing.

## Semantic constraints

Every variant retains all packed weights, block scales, three global scales, both RMS weights, residual addition, layer scalar and model epsilon. Direct VRF variants preserve FP32 operation order, per-slice element order and all existing BF16 conversion boundaries. Tiling changes preserve per-output contraction/reduction order; the sequence between independent output rows can change.

Existing baseline approximations are not new optimizations here: Erf GeGLU, two-FP8 activation expansion with fixed gain and the baseline fused postnorm BF16 rounding. They require reference-based stress validation as well as the public harness.

## Directions

- v01/v02/v03/v04/v14 remove DM round trips for pre-RMS, post-RMS and GeGLU through SDK 0.8.1 direct VRF primitives.
- v05/v06 change down tile size, trading LUT setup count and SRAM/TRF live allocation against overlap.
- v07 flattens contiguous up/gate scale rows before DMA. This tests compiler packetization; it does not assume a 240-byte row always causes a separate physical HBM transaction.
- v08/v09/v10 load the entire down matrix once, then process DM views. This separates legal DMA mapping from compute tiling and tests 10+5,15,5+5+5 views.
- v11 removes the old unused warmup read from uninitialized DM. Functional output should be unchanged; issuer startup timing can change.
- v12 loads up and gate into separate positions of one local matrix axis, allowing one lookup/contraction chain for both matrices. It preserves both matrix contributions while trading earlier compute overlap for one fewer lookup setup.
- v13 replaces the partial-output HBM bridge with real local/peer SRAM transfers into contribution slots. The surviving cluster keeps the original contribution ordering. Synchronization cost may negate the shorter data path.
- v15 combines down15 with post-RMS/GeGLU direct VRF to test reduced command count while retaining all arithmetic.
- v16/v17 replace both inferred RMS slice reductions with explicit eight-way switch gathers and scalar sums, using baseline/down15 tiles respectively. v18/v19 isolate this change to post-RMS because pre-RMS ring changes add scheduling cost.
- v20 substitutes the fixed `Broadcast1(8,1)` switch pattern for the equivalent custom post-RMS ring. The intended benefit is to remove routing-table provisioning without changing the participating slices or arithmetic.

## Hardware validation

The root agent coordinates full builds and Arena submissions. No Arena job is submitted by this subtask. Candidate acceptance requires the same frozen binary/source/fixture identity, all three public kernels passing, stress results and paired measurements. Compile success alone is insufficient.
