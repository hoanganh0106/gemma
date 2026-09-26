# P127 K2 VRF scale direct probe

## Hypothesis

Reuse the already materialized `scale: TailVrf` as the producer for `sw`,
instead of fetching `scale_dm` a second time for the `scale * rms_weight`
pass. The P126 schedule shows that second Sub fetch/collect as an exposed
roughly 311-cycle operation.

## Gate 1 result

Rejected by the SDK type system. `SubContext::begin` in `furiosa-opt-std`
0.8.1 accepts a `DmTensorView`, while `scale.view()` is a `TensorView` of a
VRF tensor. The attempted direct producer therefore cannot be expressed by
the current API, and the following vector pipeline is unavailable as well:

```text
expected DmTensorView, found TensorView<f32, ...>
PositionBegin does not satisfy CanApplyVectorInit
```

No schedule was produced and no runtime or Arena test was run.

## Five-question record

1. Hypothesis: a VRF-to-VRF producer could remove the second scale fetch.
2. Expected cost removed: the `scale_dm` fetch/collect before `sw`.
3. Observed result: the API requires a DM source, so compilation stops before
   scheduling.
4. Current loss classification: API constraint; whether a lower-level fused
   producer can preserve both consumers remains unknown.
5. Smallest discriminating next probe: test whether one Sub pipeline can emit
   a reusable `scale` and `sw` representation, or whether the compiler/API
   requires two materializations. Keep P126 as control.
