# P148 K2 Residual Before Concat Probe

## Hypothesis

Collect the residual in the narrow `H / 4 % 30, H % 4` layout and apply the
residual add before widening back to `H / 8 % 15, H % 8`.

## Result

Rejected by the SDK type-state API:

```text
no method named `vector_clip` found for ... VectorTensor ... Way4
```

`vector_clip` is only implemented for the widened Way8 state in this path.
The residual add therefore cannot be moved before `vector_widen_concat`.
No schedule was emitted.
