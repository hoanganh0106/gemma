# K3 optimization results

## Candidate recommended for hardware evaluation

`v20-post-ring8-fixed/ffn7.rs` is the preferred completed compiler screen: **108,244 cycles / 169 instructions**, versus the frozen baseline **110,520 / 170**. This is a 2,276-cycle modeled reduction (2.06%), with one fewer instruction. It is not a hardware performance claim.

Source SHA256: `B75A1F551B4E299E29DC4723C021F171673FEBB6503ADA6A163E11B6C89B1211`.

Schedule SHA256: `AC93B9D6339F08E4773FA11E23E6BFC2F9EC9C80DF45A487973CAE4F61532504`.

It makes three changes:

1. GeGLU's Sub Vector Engine result goes straight to VRF. The removed intermediate was FP32, so no BF16 rounding boundary was removed.
2. Post-FFN RMS gathers the eight slice partials explicitly with a fixed `Broadcast1 { slice1: 8, slice0: 1 }`, then sums those eight terms within each destination slice. The previous inferred inter-slice reduction is removed. Addition association may differ in FP32; the set of terms is unchanged.
3. The scalar post-RMS epsilon/square-root pass writes directly to VRF. Epsilon, square root, weighted division, residual addition, layer scaling and the final BF16 cast are unchanged.

Pre-FFN weighted RMS, both FP8 activation terms, every packed matrix, every block/global scale and the three five-row down tiles remain as in the baseline.

The prior custom-ring version `v18` has the same modeled makespan but170 instructions. Using the fixed broadcast pattern removes its anonymous719-cycle routing-table DMA load: DmaLoad count22 ->21. That DMA was overlapped, so removing it does not lower the static makespan further. The fixed and custom patterns address the same eight contiguous physical neighbors.

## Why the pre-RMS ring change is excluded

`v16-ring8` changes both RMS reductions and reaches **108,786 / 172**. Keeping the pre-RMS schedule unchanged improves that result by **542 cycles and two instructions**. A locally cheaper reduction can still worsen the kernel schedule through overlap and startup dependencies.

The SDK vector pipeline is Fp -> IntraSliceReduce -> FpDiv -> Widen. Returning to the Fp stage to apply epsilon and square root after an intra-slice sum is not supported; therefore the explicit sum and scalar operations are kept in separate commands. No approximation or distributed epsilon was introduced to merge them.

## Other compiler evidence

- Post-RMS direct VRF alone: **110,253 / 169**.
- GeGLU direct VRF alone: **110,520 / 169**.
- Both above: **110,253 / 168** (`v14`, a useful low-change hardware control).
- Adding pre-RMS direct VRF: **110,332 / 167**. Its isolated result is **110,599 / 169**, so the extra command elimination costs modeled cycles.
- Down15 instead of 5+5+5: **113,225 / 130**. It removes 40 instructions but sacrifices compute/DMA overlap.
- Down15 plus direct post-RMS/GeGLU VRF: **112,958 / 128**.
- Down15 plus both explicit RMS rings: **111,491 / 132**.
- Down15 plus post-RMS ring only: **110,949 / 130**.
- Down3 across five tiles: **113,457 / 202**. More lookup setup and commands outweigh finer overlap.
- Flattened up/gate scale DMA: **110,520 / 172**. No modeled DMA benefit; two extra mapping instructions.
- SRAM local/peer partial transfer: **111,816 / 183**. Added synchronization and instructions outweigh the shorter storage path in this implementation.
- Paired up/gate lookup, after repairing a tiled-view lowering restriction: **158,067 / 180**. Each transformed weight DMA grows from **24,612 to 48,676 cycles**, and the combined contraction waits for both matrices. This representation is rejected.
- Loading the full down matrix and computing one15-row view reproduces **113,225 / 130**.
- Loading the full down matrix then using valid10+5 row views: **112,185 / 152**. This removes18 commands but adds1,665 modeled cycles against the baseline; it has no hardware result in this subtask.
- Removing the unused Sub warmup: **110,523 / 168**, three modeled cycles worse despite two fewer instructions. The existing warmup is retained in the preferred candidate; startup behavior still needs hardware evidence to change it.

The first paired-up/gate and full-down10+5 probes failed because SDK 0.8.1 does not lower `reshape` on tiled views. Repairs retain the view's axis or reshape the full allocated tensor before tiling. The equivalent unrepaired five-row version was skipped after the restriction was established.

Full per-variant hashes, statuses and model costs are in `ledger.json`; every compiled variant has its own `compile.log` and `schedule.json`. No candidate here was promoted to the main source by this subtask.

The final baseline-to-preferred-source diff is `v20-post-ring8-fixed/source.diff`. It was audited: only the three changes described above occur. All compiler processes from this subtask completed; the only skipped variant is the documented duplicate tiled-view restriction.

## Runtime ownership and limits

The root task builds frozen integrated packages, runs all three public kernels, runs a separate stress fixture, and measures paired packages. A source-only compiler result cannot establish that a candidate is correct or faster on hardware.

The root reported a first public all-kernel PASS for the GeGLU-only candidate in P001. The first down15 package P002 also passed all15 outputs; its K3 median was effectively flat against that control (222,562 vs222,403 cycles in one pair). Thus the 40-instruction reduction has no demonstrated runtime win yet. The authoritative job logs and frozen-package provenance live in the parent research directory, not in this compiler-only ledger.

The baseline's existing Erf GELU, fixed quantization gains and fused BF16 rounding still require stress validation. These are preserved baseline choices, not new shortcuts. Inputs including non-unit RMS weights, all global scales and the layer scalar must continue to affect the output.
