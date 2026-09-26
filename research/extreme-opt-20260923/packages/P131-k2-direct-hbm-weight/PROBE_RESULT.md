# P131 K2 direct HBM weight probe

## Hypothesis

Feed the projection weight directly from HBM into `contract_outer`, removing
the 13,383-cycle weight DMA materialization in P126.

## Gate 1 result

Rejected by the SDK API before scheduling. `MainContext::begin` requires a
`DmTensorView`; an `HbmTensorView` is not accepted:

```text
expected DmTensorView, found HbmTensorView
```

No schedule or runtime test was produced.

## Interpretation

The current contract primitive has an explicit DM residency requirement. The
P126 weight DMA is therefore an API/architecture boundary for this compiler
family, not an overlooked source-level copy that can be removed by changing
the mapping alias.
