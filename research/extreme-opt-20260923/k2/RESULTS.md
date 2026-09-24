# K2 campaign result

## Selection for integrated hardware validation

**C22-fixed-rms-ring16: 22,317 static cycles / 42 instructions**, versus the frozen source's 24,718 / 44. This is 2,401 cycles (9.713%) lower in the compiler model. Runtime correctness and stable speed must be decided using the parent's frozen full-kernel packages.

The candidate combines direct VRF for SW and inverse RMS, removal of one identity reshape, sixteen 240-value tail groups, and explicit fixed-topology gathering of all sixteen group partials before summation. The scalar epsilon/sqrt/div sequence, both FP8 levels, projection BF16 boundary, every channel scale, every RMSNorm weight and every residual input remain present.

Only `src/device/sliding/output31.rs` differs from the frozen baseline in the candidate's full source tree. Cargo.toml, Cargo.lock and rust-toolchain.toml are byte-identical. `selection.json` records hashes; `candidates/C22-fixed-rms-ring16/source.diff` contains the exact complete diff.

## Main evidence

- C14 proved the mechanism: implicit 2,579-cycle cross-slice reduction/scalar pass became an explicit 289-cycle ring8 gather+sum plus the unchanged 281-cycle scalar pass, reducing total to 22,400 cycles.
- C15 used ring16 with 240 values per group and reduced total further to 22,317 cycles.
- C22 kept that schedule length while replacing CustomBroadcast with fixed Broadcast1 and removing one routing-table DMA/instruction.
- A larger contiguous weight tile (60x512 instead of 120x256) worsened DMA and x replication costs.
- Exchanging only scalar means across clusters required extra synchronization and a costly strided final store, losing to the improved cluster-0 tail.
- Sum-based normalization and direct reciprocal arithmetic did not improve the schedule; neither is included.
- The x broadcast experiments attempted fewer HBM copies, but generic routing added setup and/or exposed excessive routing time. Fixed topology cannot simply treat padded uninitialized slices as actual source values.

## All screened candidates

- C01-invrms-vrf: 24,451 cycles / 43 instructions.
- C02-sw-vrf: 24,409 cycles / 42 instructions.
- C03-sw-vrf-clean: 24,409 cycles / 41 instructions.
- C04-sw-sub: 24,410 cycles / 41 instructions.
- C05-inv-sub: 24,416 cycles / 41 instructions.
- C06-tail240: 24,318 cycles / 41 instructions.
- C07-tail960: 26,618 cycles / 41 instructions.
- C08-local-scalar: COMPILE_FAIL.
- C09-tail120: 24,396 cycles / 41 instructions.
- C10-tail1920: COMPILE_FAIL.
- C11-sum-norm: 24,410 cycles / 41 instructions.
- C12-direct-reciprocal: 24,410 cycles / 41 instructions.
- C13-weight512: 24,850 cycles / 41 instructions.
- C14-ring8: 22,400 cycles / 43 instructions.
- C15-ring16-tail240: 22,317 cycles / 43 instructions.
- C16-ring32-tail120: 22,411 cycles / 43 instructions.
- C17-local-ring: 23,794 cycles / 43 instructions.
- C18-x-fused-broadcast: 28,164 cycles / 43 instructions.
- C19-x-bf16-broadcast: 22,712 cycles / 45 instructions.
- C20-x-fused-fixed: COMPILE_FAIL.
- C21-x-bf16-fixed: COMPILE_FAIL.
- C22-fixed-rms-ring16: 22,317 cycles / 42 instructions.

## Boundaries

No root source was edited by this subtask, no Arena or MOA job was submitted by it, and no fixture/harness/scoring/timing changes are part of any candidate. Dynamic tensor values are always consumed at runtime. Compiler PASS is separate from numerical PASS. Rejected sources remain isolated as experiment evidence.

Selected source, complete schedule, compile log, exact diff, full copied source tree, dependency locks, semantic derivation for the explicit-gather method, and the per-candidate ledger are retained in this directory.
