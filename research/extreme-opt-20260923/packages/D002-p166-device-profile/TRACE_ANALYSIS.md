# P166 K2 hardware trace analysis

Arena diagnostic job `102001` passed all 15 correctness checks. Trace timing is diagnostic and is not used as official performance evidence.

The three K2 task windows were `38032`, `38221`, and `38927` cycles. On both active clusters, every run contained one dominant DMA span:

- run 0: `25818` and `25206` cycles
- run 1: `25592` and `25112` cycles
- run 2: `25230` and `24936` cycles

The compiler schedule identifies the matching dominant transfer as `weight.view().to_dm(&mut device.tdma)`, scheduled for `13383` model cycles before `contract_tile` can begin. The trace shows the real transfer consuming about 25K cycles, roughly two thirds of the complete K2 task window.

In run 0 on cluster 0, the dominant DMA starts at task offset `1772` and ends at offset `27590`. Early quantization work overlaps only the beginning. The contract and tail cannot complete until this transfer finishes, leaving the weight materialization on the critical path.

The same run has about `10419` cycles after the weight DMA ends. A simple lower-bound model is therefore about `25.8K + 10.4K = 36.2K` cycles unless contraction or the tail can overlap additional weight slabs. Starting the existing whole-weight DMA at task offset zero can save at most about `1.8K` cycles, and P132 already showed that source reordering does not alter the compiler schedule.

The next architecture candidate should tile the output-row/H dimension and pipeline weight DMA with contraction. Each H slab has independent output rows, so a double-buffered producer can load slab N+1 while the tensor unit contracts slab N. The candidate must preserve the final tail ownership and prove that the whole-weight DMA dependency is replaced by slab-local dependencies in the generated schedule.

Existing boundaries matter: P106's two 60-row transfers compiled to two serialized DMA spans and then failed graph initialization at the ring32 gather; P107's ownership repair was rejected because the commit did not preserve slice size. A new implementation must solve this ownership boundary rather than merely reorder P106.

Changing ALU assignment, explicit DM ownership, or source ordering does not attack this measured bottleneck and should no longer be used as the primary search direction.
