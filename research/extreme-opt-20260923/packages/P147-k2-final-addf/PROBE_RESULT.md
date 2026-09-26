# P147 K2 Final AddF Probe

## Hypothesis

Replace the final `vector_clip(ClipBinaryOpF32::Add, residual)` with ordinary
`vector_fp_binary(FpBinaryOp::AddF, residual)` to remove any clip-specific
lowering cost.

## Result

Rejected by the SDK type-state API:

```text
no method named `vector_fp_binary` found for ... VectorTensor ... Way8
```

After `vector_widen_concat`, the tensor is in `Way8`, while `vector_fp_binary`
requires the FP stage. `vector_clip` is the legal operation at this stage and
cannot be replaced without introducing another transition. No schedule was
emitted and no correctness claim is made.
