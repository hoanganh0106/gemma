# P145 K2 Reduce Padding 8 Probe

## Hypothesis

Keep the contraction's intra-slice reduction at padded width 8 and remove the
`vector_widen_pad::<m![1 # 8]>()` step after reducing from the two FP8 levels.

## Result

The WSL service recovered and the compiler rejected the probe:

```text
error[E0599]: no method named `vector_inter_slice_reduce` found
for ... VectorTensor ... IntraFirst ... Way4
```

With the reduction output widened to `1 # 8`, the type-level state no longer
implements the inter-slice reduction transition. The existing `1 # 4` output
followed by `vector_widen_pad::<m![1 # 8]>()` is therefore required by the
current vector API. No schedule was emitted.

The initial compile attempt also encountered the transient WSL service error:

```text
Wsl/Service/CreateInstance/E_ACCESSDENIED
```

That environment issue was resolved for the final compiler run. This probe is
now a mechanism-proven rejection, not an untested candidate.
