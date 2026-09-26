# P211 device trace

Diagnostic Arena job **102320** succeeded, all 15 checks passed. K2 diagnostic
task medians were 40094 cycles; this is not an official performance comparison.
Official P211 job 102123 measured K2 39522, versus adjacent P166 control 102139
at 37668. The diagnostic binary SHA256 is
`2eb7b1588d0ffcb383af5f4ef086524f4eb19c4321f21f28cd54e4606552e1a7`.
All device/library source files match P211 byte for byte; only the host trace
collector and remote trace profile differ.

The old `trace_windows.py` fields `dominant_dma`, `before_dma`, and `after_dma`
refer to the single longest DMA. They are misleading for P211 because either
weight slab can be longest. Use the two chronologically ordered `large_dma`
entries in `arena-102320.trace.windows.json`.

Across three runs and both clusters, the first weight DMA takes 12798–13736
cycles and the second 13215–13717. The second starts 9 cycles after the first
ends in every cluster/run. Thus the two weight transfers do not overlap with
each other. A TuExec pass overlaps the second DMA by 1624–2446 cycles. This is
the first slab contract; the compiler model had only 898 cycles of overlap.
The actual hardware mechanism exists, but it hides too little work.

The second DMA ends at task offsets 28260–28909. The task still has
9944–12196 cycles after that point. For example, run 0 cluster 0 then executes
a 2478-cycle TuExec alongside a 1594-cycle DMA; later transfer, redistribution,
RMS, final output, and sync remain exposed. Trace span labels alone cannot
assign every later pass to a source line; this attribution uses the known source
order and compiler schedule, so it is an inference.

P166's single large DMA takes 24936–25818 cycles across its older diagnostic
trace. P211's two slab DMAs together take 26013–27161 cycles. Their sum is
roughly 1–2K longer, before accounting for the extra contract/redistribution
and tail geometry. Cross-session timings are diagnostic, not a paired speedup
proof. The official paired Arena result already shows P211 slower by 1854.

## Next decision

Simply tuning the current two-slab DMA/contract order is unlikely to reach
30000: second DMA completion is already near cycle 28–29K, and about 10–12K
of work remains. A viable follow-up needs the tail to consume each slab's
projection and compute local RMS partials before the second DMA completes, and
must remove a substantial part of the exposed redistribution/sync. That is a
new coupled producer/consumer dataflow, not a small packet or view tweak.
Measure the first partial's readiness and the final global RMS barrier in its
schedule, then use official Arena for correctness and performance. No 30K
floor or achievable speed is proven by this trace.
