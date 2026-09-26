# P129 K2 direct mean reduction probe

## Hypothesis

Reduce the `H / 120` partials with `vector_inter_slice_reduce` directly,
removing the `switch` broadcast and the subsequent intra-slice reduction in
the P126 `mean` stage.

## Gate 1 result

Rejected by VISA commit shape validation:

```text
commit: input does not match pipeline
declared InSlice: [H_120=32, 1#8]
pipeline InSlice: [1#8]
```

The direct reduction does not produce the broadcast shape consumed by the
following RMS pass. No schedule or runtime test was produced.

## Interpretation

The current `switch` is carrying a real layout transformation: it turns the
per-row-group partial into the `Vr`-broadcast shape before reduction. The
materialization is therefore not removable by a direct reduction with the
available primitive. A future attempt must preserve that producer/consumer
shape contract or change the RMS consumer together with it.
