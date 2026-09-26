# K1: load Q/K/V weights before projection consumers

This is an isolated source probe copied from the current working tree on 2026-09-23. It compiled successfully with SDK 0.8.1; it has not been run on RNGD.

The only source differences are `src/ops.rs` and `src/device/sliding/qkv_head_local.rs`. The three weight DMA loads and their existing ownership-preserving reshapes are moved into `prefetch_projection_weights`, called before input normalization. The projection helpers consume the resulting DM tensors. Tensor mappings, contractions, scales, rounding boundaries, normalization, RoPE, and output writes are unchanged. No fixture values are used.

Hypothesis: expose all three independent weight transfers before the projection consumers so the compiler may load K and V earlier and reduce DMA gaps. A compiler that already reorders the graph optimally may emit the same schedule; this remains a useful negative result. Longer DM lifetimes may also make allocation or overlap worse.

Weight storage is 31,457,280 bytes (30 MiB) across 512 active slices: Q uses 30,720 bytes/slice, K and V each use 15,360 bytes/slice, totaling 61,440 bytes (60 KiB)/slice. The compiled allocation must still account for activation, projection, normalization, and scratch buffers; these arithmetic sizes alone do not establish peak DM usage.

Baseline source SHA256:

- `src/ops.rs`: `C75F578DC24603450EF7EBA136B12AA90CFD8DB6E928B3B320C8B34C1BD09F1C`
- `src/device/sliding/qkv_head_local.rs`: `E40F4ABFE84DEE3F3BBA2776B164E52382261D83130C7F71193E2FE4A6981C84`

Probe source SHA256:

- `src/ops.rs`: `A289EBF9E9AA213BB4A72F0B42678321C4C0204E488D1E08AE547FCC11A2BFD9`
- `src/device/sliding/qkv_head_local.rs`: `733F5FDB188CEF125547B7FB8E43B78D3C5A1D844EEA6A9DDDC8D3BF5DD17433`

Result: no static improvement. Both the current baseline and this probe have makespan 42,539 cycles, 154 instructions and 117 tensors. Every instruction has the same index, type, context, lifetime, history and utilization. Source descriptions and seven input/output tensor-ID references differ, so the serialized instruction graphs are not byte-identical. Q weight DMA remains 2,848-16,360, K remains 23,786-30,816, and V remains 31,267-38,297. Moving independent loads earlier in source did not change instruction timing or resource assignments. This result does not support hardware testing this probe for a performance win.

Artifacts: `compile.log`, `qkv-schedule.json`, and `comparison.json`. Schedule SHA256: `06934B4BC9314ADE77BF9C84A7DD1E4899AEFEEADCE940B673E3E580F18A6EE7`.

Compile command executed from this directory:

```sh
CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target CARGO_INCREMENTAL=0 cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule qkv-schedule.json
```
