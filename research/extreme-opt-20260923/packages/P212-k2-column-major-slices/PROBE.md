# P212: column groups before row groups

Parent: P166 binary `5965d4748e2428ef04a59223599bc7a67a6bc8af7728ddc9ccdca29a4321422f`.
Only `src/device/sliding/output31.rs` changes. The public signature, harness,
fixture, K1 and K3 sources are inherited unchanged.

The P166 weight allocation is 15728640 bytes, equal to 3840*4096 FP8 values.
This probe does not claim to reduce logical weight traffic. It changes physical
slice order from `(H/120%16, Qs/256)` to `(Qs/256, H/120%16)`. The activation DMA
uses `(Ns, Gs, H/120%16)`, preserving the factor order when Ns/Gs become Qs/256.
The contract output changes to `(1#16, H/120%16)` before the existing tail DMA.

Hypothesis: changing distribution across physical slices changes DMA cost or
the projection-to-tail transfer cost. The same packet32 contraction and BF16
projection boundary remain. Consumer ownership legality, switch cost and actual
hardware transfer duration are unknown. A compiler success is not evidence of
speed; submit the full public harness and retain all three correctness tests.

Decision criterion: compare hardware with adjacent P166. If slower, retain the
result and inspect changed DMA/redistribution before attributing the regression.
No numerical speedup or 30000-cycle floor is assumed.

Build rejected at contract reduction: `VRU reduce axes must be innermost in
Partitioning, got reduce_labels=[Qs]`. No executable or Arena measurement was
produced. Whole-slice order permutation is not legal for this direct reduction.
