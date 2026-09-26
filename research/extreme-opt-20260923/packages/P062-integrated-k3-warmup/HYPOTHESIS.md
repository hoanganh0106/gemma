P062 replaces the no-op WARM-UP Sub command in `ffn7.rs::normalize_quantize` with a real,
dependency-free Sub fetch of `rms_weight` (moved from later in the function to the kernel entry).
The first Sub command of K3 pays a ~10x issuer stall (2,696 device for a 267-cycle StoVrf); the
original code paid it on a dummy command whose value is never read, while the DMA idled 2,464 cycles.
P062 instead issues the real `rms_weight -> weight_vrf` load as that first Sub command, so the stall
buys useful work and the DMA/Main overlap can begin earlier. K1 (qkv69382) and K2 (P041 output31.rs)
sources are unchanged. This is an integrated base (P041 K2 + qkv69382 K1 + ffn7 K3) with one K3
critical-path fix; it must be Arena-validated (15/15) and paired against the P041 parent, because K3
within-job variance is smaller than K2's but still non-trivial.

RESULT (Arena 84319): 15/15 PASS. Medians K1=80,479 / K2=38,152 / K3=218,763.
Versus P041 parent (Arena 83329) K1=79,914 / K2=38,140 / K3=216,460:
- K1 +0.7%, K2 ≈0, K3 +1.1% — all within noise, no improvement.
- The warm-up removal did NOT recover the hypothesized ~2,464 idle DMA cycles; the Furiosa 0.8.1
  compiler already primes the issuer / overlaps the rms_weight load, so the no-op warm-up was
  effectively free. This is a NEGATIVE result: removing it changed the binary (SHA 76f6fd87...)
  but not the runtime.

LESSON: micro scheduling / warm-up / reorder changes are exhausted by the compiler. Real gains must
come from algorithmic reduction of arithmetic (e.g. K2 final-pass fusion, K1 norm restructuring),
not from rescheduling already-independent commands. Do NOT promote P062.
