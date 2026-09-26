P061 moved the three independent Sub-TU fetches (`residual`, `weight_scale`, `rms_weight`) ahead of
`contract_tile` in `output31.rs::project_normalize_add`, intending Sub to run them in parallel with the
Main contraction.

RESULT: NO-OP. The rebuilt `target/release/test_kernels` SHA256 is
`5c37d4a72abc88b88c4d4196d9f95c53d3f8a12ea3a4ba35dd8ee02d85e90d35` — byte-identical to P041's
submitted binary. The source reorder is real (verified by diff) but the Furiosa 0.8.1 compiler already
schedules independent Sub/Main/DMA commands in parallel; reshuffling the Rust issue order does not
change the emitted schedule. This confirms the Main/Sub TU overlap is already exploited by the baseline
P041 schedule, so this particular axis is exhausted.

Lesson: do not pursue pure instruction-reordering changes; the compiler handles them. Real K2 gains
must come from reducing the final-pass arithmetic (still ~2,276 cycles in the tail) or the scalar
materialization, not from rescheduling already-independent commands. See PLAN.md "P041 development"
section for the remaining open axes (P055 K3 warm-up stall, P056 K1 norm overlap, final-pass fusion).
