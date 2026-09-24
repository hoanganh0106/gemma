# Independent K2/K3 correctness review

Reviewed 2026-09-23 by the K1 agent. Read-only review of candidate source and diffs against `../baseline`, using CodeGraph first. No candidate was edited and no build or runtime test was started for this review.

## Result

**No concrete correctness blocker found in the four reviewed candidates.** The changes preserve every runtime operand and all existing BF16 conversion boundaries. Explicit RMS gathering changes FP32 addition association; source review therefore establishes operand/mapping preservation, not bitwise equality or hardware tolerance PASS.

## Exact reviewed files

- `../k2/candidates/C15-ring16-tail240/output31.rs` — SHA256 `f35b137c59decf5c97f8203b36b5fae60349fbe75618c9d44f7157bdd3f4d472`.
- `../k2/candidates/C22-fixed-rms-ring16/output31.rs` — SHA256 `a4803e5982db94e082af9c9581226d1fda0c2451c4c5a969d0726e5302bad416`.
- `../k3/v18-post-ring8/ffn7.rs` — SHA256 `ce9e0c16f4e2d7d26cb8fc74182a2ad821bedc16253287969306dad3c839f5e2`.
- `../k3/v20-post-ring8-fixed/ffn7.rs` — SHA256 `b75a1f551b4e299e29dc4723c021f171673febb6503ada6a163e11b6c89b1211`.

## K2 C15 and C22

- Full `weight`, both FP8 activation components, `weight_scale`, `rms_weight`, and the incoming residual remain consumed. Matrix load and contraction are unchanged (lines 51–140). Channel scales participate both in squared norm (line 207) and the output numerator through `sw` (line 190); RMS weights remain only in the numerator. Residual addition remains after normalization (line 260).
- Tail mapping `m![1 # 16, H / 240]` has 16 initialized slices in cluster 0, covering all 3840 channels exactly once, 240 channels per active slice. Every tail operand uses this same mapping. The projected-vector HBM hop preserves logical row order before repartitioning; no matrix rows are dropped.
- The mean gathers all 16 initialized group partials as `H / 240` time values (lines 219–228). `Vr=16` names the 16 destination replicas. No padding dimension is included in the scalar sum. `Vr` to `H / 240` VRF reshape (line 246) preserves the same 16 physical active slices.
- Each group still divides its sum by `H_F32=3840`, and the 16 group means are summed exactly once. The former 8-by-480 grouping becomes 16-by-240, changing FP32 association and placement of per-group rounding; the mathematical mean is unchanged. Epsilon remains scaled by `16^2`; the existing reciprocal implementation `sqrt(t)/t` and final operation order remain unchanged.
- Contraction output stays BF16, tail arithmetic stays FP32, and final output conversion stays BF16. The removed `sw` and reciprocal DM hops were FP32-only, so direct VRF delivery removes no rounding boundary.
- **C22 differs from C15 by exactly one line:** `CustomBroadcast { ring_size: 16 }` becomes `Broadcast1 { slice1: 16, slice0: 1 }` at line 222. A contiguous 16-slice ring matches the active source layout and output time extent. It does not reinterpret padded slices as real source channels.

## K3 v18 and v20

- Relative to the frozen baseline, v18 changes only GeGLU intermediate transport and post-FF RMS transport/reduction. Pre-FF weighted RMS, up/gate/down packed matrices, every block scale, all three global scales, residual reuse, post-FF RMS weights, and layer scalar are retained.
- Runtime operand paths remain explicit: pre-FF RMS and matrix/block scales (lines 520–531), gate global scale (535–536), down scales and all three down tiles (539–545), down/up global scales (573–612), post-FF RMS weight/residual/layer scalar (668–721). The three down tiles cover the same 15 local rows without gaps or overlap.
- Post-FF `ReducingSlices=m![1 # 32,H / 480]` supplies eight initialized cluster-0 slices. Gathering `H / 480` produces eight partials at each of the eight active `Dummy8` destinations. Scalar VRF reshape from `Dummy8` to `H / 480` retains identical physical positions. The other 248 padded positions cannot contribute to the eight-value sum.
- All 3840 squared values still participate. Per-slice sum, division by 3840, EPS addition, square root, weight multiply, residual add, and layer-gate multiply are retained. Only association of the eight FP32 group partials changes.
- GeGLU direct VRF transfer replaces an FP32 DM commit/reload. Its FP32 arithmetic and subsequent BF16 conversion are unchanged. Post-RMS direct VRF replaces another FP32-only round trip. The final fused BF16 boundary is inherited unchanged from baseline.
- **v20 differs from v18 by exactly one line:** `CustomBroadcast { ring_size: 8 }` becomes `Broadcast1 { slice1: 8, slice0: 1 }` at line 645. The fixed contiguous ring matches all eight initialized group partials.
- Existing pre-FF sparse-copy and gate-scalar sparse-copy code is byte-identical to baseline. In the former, only Q4 member 0 is selected after the stride-8 replication; in the latter, only Lead4 member 0 is read after gathering. The new post-FF gather does not widen either path's dependence on padding.

## Validation boundary

No new fixture dependence, input-dependent bypass, constant replacement of runtime weights, or narrower FP8 quantization range was introduced by these deltas. Generic-input hardware tests remain the deciding evidence for the changed FP32 reduction association. This review makes no new runtime PASS or performance claim.
