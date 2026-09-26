# P133 K2 epsilon-after-mean probe

## Hypothesis

Move `EPS_SCALED * H` from the full H-way partial reduction to a single
`EPS_SCALED` add after the cross-slice mean reduction.

## Gate 1 result

Rejected by the vector stage machine. SDK 0.8.1 does not allow an Fp binary
operation immediately after `vector_intra_slice_reduce`:

```text
IntraSliceReduce does not satisfy CanTransitionTo<Fp>
```

No schedule or runtime test was produced.

Trying the `vector_clip(ClipBinaryOpF32::Add, EPS_SCALED)` form from an older
Sub-context path also fails on this Main-context stage; the method is not
available after this reduction marker. Adding another intra-slice tag is
likewise illegal. Thus the alternate epsilon primitive does not open a legal
post-reduction path here.

## Interpretation

The epsilon placement is constrained by stage ordering. A legal post-reduce
version would need an intermediate `vector_final` and a new pass, which adds
materialization instead of removing work. P122's pre-reduction fused form is
the only legal low-overhead form found in this family.
