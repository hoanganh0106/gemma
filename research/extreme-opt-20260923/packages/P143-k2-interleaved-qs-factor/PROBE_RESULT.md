# P143 K2 Interleaved Qs Factor Probe

## Objective

Test whether the K2 contract can preserve the reduced `Qs / 64` ownership while
using an interleaved lane contract:

```rust
.contract_time::<m![H % 120, Qs / 64 % Lq]>()
.contract_lane::<m![H % 120, Lq], m![Lq # 8]>(LaneMode::Interleaved)
```

The intent was to expose a legal lane factor without changing the mathematical
K2 path.

## Result

Rejected by the SDK 0.8.1 compiler during contract validation.

```text
contract_time: Padding mismatch.
Non-reduced axes in OutTime (H % 120, Qs / 64 % 2)
do not preserve padding from Time (H % 120, Qs / 64 % 4)
```

## Mechanism conclusion

The interleaved lane factor changes the non-reduced time axis from the input
padding factor `% 4` to `% 2`. The contraction contract requires padding to be
preserved on non-reduced axes, so this family cannot be made legal by changing
only the lane factor. No schedule or performance result was produced.

## Provenance

- Package: `P143-k2-interleaved-qs-factor`
- Compiler: Furiosa SDK 0.8.1
- Basis: P122 K2 `z.reshape()` path
- Status: compile rejected; no hardware submission
