# P128 K2 no-`sw` control probe

## Hypothesis

Remove the precomputed `sw = scale * rms_weight` pass and perform the three
final multiplies directly in the output pass.

## Gate 1 result

Rejected by VISA lowering. The final pass with a third `MulF` fails:

```text
visa: while lowering TensorUnit
caused by: 13 is not available for op Binary(MulF)
```

No schedule was produced and no runtime or Arena test was run.

## Interpretation

The `sw` pass is not merely a movable convenience. With the current vector
engine/API, the final pass cannot carry three multiplies. Its cost is therefore
an intrinsic resource constraint for this dataflow, while the duplicate scale
fetch remains a separate unknown/API debt.

The five-question record is: hypothesis was three-multiply fusion; expected
cost removed was the `sw` pass; observed result was a VISA resource failure;
the loss is intrinsic to the current lowering; the next discriminating probe
is a producer-side representation that preserves both `scale` and `sw` without
adding a third final multiply.
