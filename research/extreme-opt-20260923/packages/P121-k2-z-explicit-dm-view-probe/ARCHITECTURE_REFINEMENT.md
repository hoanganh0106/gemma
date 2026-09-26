# P078 K2 architecture refinement

## Family

P078 is the **tail/120 plus explicit ring32 gather** family in `src/device/sliding/output31.rs`.
It changes the K2 dataflow from 16 tail groups of 240 values to 32 logical groups of 120 values, keeps the projection BF16 boundary, and makes the cross-slice mean gather explicit.

## Cost model

| Component | Evidence | Classification |
|---|---:|---|
| Explicit ring32 reduction | P078 schedule 21,337 cycles / 42 instructions; P041 schedule 21,290 / 42 | intrinsic family cost plus mapper geometry |
| Removal of implicit scalar/reduction serialization | P078 hardware repeatedly beats P041 | intrinsic potential realized on hardware |
| Tail gather/layout overhead | P078 static schedule is 47 cycles above P041 | intrinsic for current H/120 geometry, removable only with a legal ownership mapping |
| `weight.to_dm` DMA | 13,383 schedule cycles | shared critical-path cost; no legal reduction found yet |
| `sw` on Sub | same static schedule, distinct binary; paired hardware gain | hardware scheduling potential |
| epsilon hoist before H reduction | same arithmetic and correctness; retained in P078 | implementation refinement |
| additional H/120 partitioning | required by ring32 ownership | currently intrinsic/unknown |

## Hardware evidence

Fresh paired checks remain positive and all-correct:

- 92721 P078 37,463 vs 92728 P041 37,868: -405 cycles.
- 93023 P078 37,777 vs 93029 P041 38,112: -335 cycles.
- 93040 P041 38,400 first vs 93045 P078 37,846 second: -554 cycles.

All jobs passed 15/15. These are frontier evidence, not an official promotion.

## Dependency DAG

`weight.to_dm -> contract_tile -> z.to_dm -> partial + scale/rms paths -> explicit ring32 gather -> inv_rms -> final output`

The compiler repeatedly rejects reordering or relocating nodes with `gather src residue has no live target axis`. This identifies an ownership/dataflow constraint, not a performance result. The legal family boundary is the current `Tail = m![1 # 8, H / 120]`, `Broadcast1 { slice1: 32, slice0: 8 }` topology.

## Five-question iteration record

1. **Hypothesis:** improve the tail/120 family by reducing critical-path DMA or issuer serialization while preserving ring32 ownership.
2. **Expected cost removed/hidden:** weight DMA or independent scale/RMS fetch latency.
3. **Observed:** packet/layout changes and producer/fetch reordering either violate SDK ownership or packet constraints before schedule generation.
4. **Classification:** the current failures are compiler/dataflow constraints; the 13,383-cycle weight DMA remains an unresolved shared critical-path cost.
5. **Next smallest discriminating experiment:** microprobe the weight DMA split at the actual `sliding/output31.rs` boundary, with a cost model comparing overlap against the extra DMA command and redistribution. Do not alter the ring32 consumer until that probe is measured.

## Split prototype result

P106 implemented the split and contract together using the previously legal 60-row contract pattern: two independent 60-row weight DMAs and two commits into the H/120 output tile. The Rust build passed, but graph initialization failed at the downstream gather with `gather src residue has no live target axis`. This is evidence that the P077 split contract pattern does not transfer directly into the ring32 consumer mapping. The split family remains an architecture hypothesis with an unresolved ownership adaptation; it is not a hardware result.
