# P084 reuse scale VRF probe

Parent: P078-k2-tail120. The `sw` pass attempted to consume the already materialized `scale` VRF tensor instead of fetching `scale_dm` a second time.

Compiler gate: rejected by the SDK type boundary. `Device::begin` accepts `DmTensorView`, while `scale.view()` is a VRF `TensorView`; the resulting tensor also has no legal `collect` transition. No binary or runtime measurement was made.
