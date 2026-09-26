# K2 inverse RMS direct VRF feasibility probe

Prepared 2026-09-23 in an isolated copy of the current Cargo.toml, rust-toolchain.toml and src. Main source files are unchanged.

Replace inverse-RMS Main output -> DM -> Sub fetch/collect -> VRF with `vector_final().to_vrf(&mut device.sub)`. Keep the same FP32 sum-of-squares, epsilon, sqrt, reciprocal, and physical slice layout. Preserve the formerly DM-side symbolic reshape using SDK 0.8.1 `VrfTensor::reshape` (the grouping is unchanged in physical wire order).

The installed SDK defines Main `.to_vrf(&mut device.sub)` and `VrfTensor::reshape` in `src/engine/collect.rs:110` and `src/tensor/memory.rs:1509`. Hardware-doc research compiled a minimal direct-output example. This full K2 probe checks whether its additional inter-slice reduction and FPU stages can use the same output path and whether eliminating the DM round trip shortens the kernel schedule.

Potential tradeoff: Main-to-VRF producer reserves Sub as well, reducing overlap. Compile success is not numerical or hardware performance validation.

## Result

Exact K2 compile **passed** with SDK 0.8.1. Command from this probe directory:

```sh
CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule k2.json > compile.log 2>&1
```

- Baseline verified from `../../k2.schedule.json`: **24,718 cycles / 44 instructions**.
- Candidate `k2.json`: **24,451 cycles / 43 instructions**, **267 fewer cycles (-1.080%)**.
- The inverse-RMS producer remains `20223..22802`. The formerly exposed Sub preload `22802..23069` disappears, allowing the final Main pass to start at 22802 and end at 23203. Weight DMA, projection, HBM hop, and prior tail timings are unchanged.
- Source SHA256: `02630EA698CC31D5749AE62248C3E0688A50E185DE511B409E9AFF16FC649879`.
- Schedule SHA256: `433C1390E2557E5AD0661B8CA4161AD489EFDB5CF9312DE4C70D22CF252A39C3`.

Decision after the initial static probe: **retain as a promising static candidate**. Compilation confirms this full reduction/FPU pipeline can write VRF directly. Arithmetic and physical per-slice mapping are preserved.

Subsequent runtime validation is recorded in [RUNTIME.md](RUNTIME.md): the full public binary was built from the root-matching dependency lock, frozen, and validated on Arena **78085** with **15/15 output checks passing**. A bounded ABBA follow-up uses the same frozen packages. That report contains hardware medians and the current performance-confidence decision; the static reduction alone is not a hardware speed claim.

The old source-level claim that every vector result must be committed to DM before VRF loading is no longer a valid restriction for SDK 0.8.1. Similar opportunities in FFN deserve separate one-change probes before combination.
