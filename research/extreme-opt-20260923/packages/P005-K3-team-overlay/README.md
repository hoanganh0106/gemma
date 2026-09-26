# P005 K3 overlay

This overlay contains the P005 K3 kernel (`src/device/shared/ffn7.rs`).

## Integration

Copy `src/device/shared/ffn7.rs` over the matching file in the target checkout.
The P005 package uses the existing `src/ops.rs` entry point and shared module wiring; no additional K3 wiring file is required relative to the P005 baseline.

## Provenance

- Source package: `P005-fixed-rings`
- P005 Arena jobs: `78726`, `78756`
- K3 source SHA256: `b75a1f551b4e299e29dc4723c021f171673febb6503ada6a163e11b6c89b1211`
- Official fixture SHA256: `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`
- P005 correctness: 15/15 checks passed in both listed jobs

The K3 source is measured as part of the complete P005 package. Rebuild and rerun the team's full harness after integration.
