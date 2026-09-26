P035 changes only K2 command issuance order. It preserves the exact x bytes, FP8 weight bytes, two-level activation quantization, contraction mapping, weight_scale, rms_weight, residual path, public entrypoint, and all arithmetic.

The parent issues the long weight DMA before the small x DMA and then quantizes x. Device profiling showed the weight DMA exposed for roughly 25.6k cycles. P035 issues x first, then the same weight DMA, then runs quantise_x before contract_tile. The intended effect is to let main/sub compute for quantise_x overlap the long TDMA weight transfer without changing ownership or numerical semantics.

Static schedule is a screening signal only. Promote only if full public correctness passes and repeated paired Arena measurements improve K2 against the exact P005-K1-081-K2K3-P005-final control.
