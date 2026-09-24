# Semantic and submission audit

This campaign optimizes the current SDK 0.8.1 implementation, preserving its model operations. Compiler success is a screening condition, not a correctness result. Final selection also requires the complete official three-case harness and separately generated diagnostic inputs.

## K1: weighted input normalization is mandatory

`ops::sliding_project_qkv` must pass the runtime input RMS weight into `normalize_native_input`. The reviewed p14/p15 sources still load every input element and every input RMS weight, compute the global mean square plus epsilon, divide the input by the resulting RMS, multiply by the weight, and materialize the normalized BF16 activation. Dynamic calibration and both FP8 activation components remain. All Q/K/V matrices, channel scales, Q/K RMS weights, V normalization, dynamic RoPE offset, cosine/sine tables, and output/cache writes remain in the dataflow.

p15 fuses gathering already-BF16 projection values with channel scaling. The BF16 contraction boundary and subsequent BF16 scaled-output boundary remain; no projection term or scale is dropped. Direct VRF handoffs carry the same FP32 values previously stored in DM. RoPE's indirect DMA explicitly creates 16 initialized head copies across two clusters; subsequent reshape operations relabel or hide initialized copies rather than inventing data through padding.

The older A010 source is not imported wholesale: its unused input-weight parameter would violate this requirement. Its parallel-V scheduling idea was screened separately with weighted input normalization retained. The variants that changed the V statistics rounding boundary did not beat the better exact-dataflow candidates and were rejected.

## K2: every partial participates in the RMS

C14 retains eight groups of 480 values; C15 retains sixteen groups of 240 values. Both cover all 3,840 output channels exactly once. A Switch gather routes all group partials into each group's time dimension; an intra-slice sum then computes the global total. Epsilon, square root and reciprocal arithmetic remain. Channel scales, RMS weights, residual addition, both FP8 activation components and the projection BF16 boundary remain.

Changing reduction topology can change FP32 addition order. This is why a schedule improvement alone is insufficient, and the hardware checks use the original reference equations and tolerances.

## K3: all three quantized matrices and scale levels remain

The reviewed v14/v16/v18 family retains full up, gate and down matrices, packed-weight decoding, every local block scale, each runtime global scale, the GeGLU calculation, input/output RMS weights, epsilon, residual and runtime layer scalar. The existing two FP8 activation components and BF16 boundaries remain. Direct VRF handoffs remove materialization and reload instructions. Explicit ring gathers retain all eight RMS partials. v18 changes only the post-RMS reduction topology; the pre-RMS reduction and the three five-row down tiles remain.

## Measurement and package controls

- Official candidates keep all 53 source files frozen except the declared device implementation files and any explicitly recorded private-call adjustment in `ops.rs`. The official harness, public function signatures, shapes and tolerances remain intact.
- Diagnostic packages are separate artifacts. Their host harness and matching Python-generated references vary inputs independently; their timings are never reported as official score.
- Every hardware package records source, binary, fixture, dependency-lock and entrypoint hashes. Full builds compile all three kernels, with no CPU pinning and the same profiling policy as the control.
- Two early builds returned a stale shared-target binary (`fresh=true`) and were rejected before submission. The build helper now forces changed library sources to rebuild, rejects a fresh executable, checks its manifest path, and holds one release-build lock through copying and hashing the executable. An official/diagnostic pair may reuse a library only when all library source/config bytes and the actual library artifact hashes match the previous completed verified build; that origin is recorded explicitly. Rejected evidence is retained under `rejected-stale-build`.
- A score comparison uses a complete job's three medians. Individual minima from different jobs are never combined into a synthetic submission result.
- Private Arena jobs validate and measure candidates. No public source publication or official MOA last-submission replacement is part of these screening runs.

Passing finite tests does not prove correctness for every representable tensor or establish a global performance optimum. Final measured results and any remaining limitations belong in the campaign result report.
