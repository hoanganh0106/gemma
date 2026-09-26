# P094 widen inverse probe

P094 inserted `vector_widen_pad::<m![1 # 8]>()` after the H reduction and then attempted scalar multiplication by `1.0 / Aa`.

The SDK still rejects the transition: `IntraSliceReduce` has no valid transition to `vector_fp_binary` after this widen operation. P094 is rejected before schedule generation and Arena.
