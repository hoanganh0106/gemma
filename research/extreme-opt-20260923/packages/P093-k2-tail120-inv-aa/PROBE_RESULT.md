# P093 inverse-constant probe

Replacing `.vector_fp_div(AA_F32)` with a scalar multiply by `1.0 / AA_F32` was rejected by the SDK type-state gate:

`IntraSliceReduce: CanTransitionTo<Fp>` is not implemented.

The reduction output cannot transition directly into `vector_fp_binary`. P093 is rejected before schedule generation and Arena.
