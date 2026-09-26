# P079 K2 one-level FP8 correctness probe

Parent P041. Only `src/device/sliding/output31.rs` changed: quantize activation as one FP8 level instead of q0 plus FP8 residual q1. All weight, weight-scale, RMS-weight, residual, and output paths stay live. The hypothesis was to remove the second activation quantization and one contraction level.

Compiler SDK 0.8.1 accepts the implementation. Whole-K2 static schedule: 21,289 cycles / 30 instructions versus P041 21,290 / 42. The long weight DMA remains the static critical path; fewer instructions do not imply lower latency. Release build passed. Binary SHA256 `f9494016841eab28deff8fecee28ab95bc4bdcb9d0971c8b3fe183892979fdd9`.

Arena job 91962: K2 FAIL all three fixture seeds. Maximum absolute differences 0.09375, 0.10156, 0.09375; within-tolerance fractions 99.24%, 99.45%, 99.51%. K1/K3 pass. K2 cycle median 37,945 is invalid for promotion because correctness failed.

Mechanism conclusion: dropping the FP8 correction level loses required numerical precision on this fixture. The removed quantization/contraction work is intrinsic to this two-level representation; one-level architecture needs a different precision mechanism, such as a legal BF16 activation route, before hardware performance is relevant. P034's BF16 route compiled at 31,027 static cycles due to the explicit FP8-weight conversion/materialization, so its current implementation debt exceeds the P041 schedule by 9,737 cycles. Keep P041 as control; do not promote P079.
