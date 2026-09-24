# K1 semantic and submission-scope audit

The strongest completed static candidate at this audit is **p15-fused-rope-head-broadcast**, source SHA256 `fd035d3a19dab1a88f2278cbd2603beb4f84bf8dcc96a43858e11aefd5213b12`. Static: **39,698 cycles / 136 instructions**, schedule SHA256 `db11fd772dbc3e1cd100551806a9e3643a831f00e4fa1a673c02e3454eeda23d`. Root owns numerical and hardware acceptance; this document does not turn static success into a runtime result.

## Scope

P15 changes only `src/device/sliding/qkv_head_local.rs`. Its `src/ops.rs`, all externally visible `#[device]` signatures, host harness, reference implementation, axes, dependency lock and toolchain match the frozen baseline byte-for-byte. `audit.py` verifies each candidate's changed-file set and lockfile identity; `audit.json` records the full candidate tree identity. An allowed file path alone is not sufficient for semantic correctness, so the active dataflow is checked below.

## Every input still contributes

- **x and input_rms_weight:** `normalize_native_input` loads both. It computes mean(x²) over all 3,840 hidden coordinates, adds EPS, applies sqrt, divides each x coordinate, multiplies the supplied corresponding weight and rounds the result to BF16. The input weight is never replaced by 1 or absorbed by a fixture-specific cancellation.
- **Dynamic activation representation:** The current normalized input determines max magnitude and quantization scale. High and residual FP8 components both remain in TRF and contribute to every Q/K/V projection. The old A010 assumption that input weight can be ignored is not used.
- **q_weight, k_weight, v_weight:** Every matrix byte remains loaded and contracted with both components. P15 does not sparsify, mask, sample or assume values in these matrices.
- **q_weight_scale, k_weight_scale, v_weight_scale:** Every supplied channel scale remains loaded and multiplied. The output of contraction is still rounded to BF16 before channel scale, and the scaled result is still rounded to BF16 before head normalization.
- **q_rms_weight and k_rms_weight:** Both supplied head weights remain applied after the complete 256-coordinate head RMS calculation with EPS. V retains its complete unweighted per-head RMS normalization.
- **cos, sin and rope_offset:** Both supplied tables are gathered at the supplied offset. The same half rotation and the same signed sine values are used. No fixed position or reconstructed trigonometric approximation replaces them.
- **kv_offset, k_cache, v_cache and q_out:** The original ops body still scatters K and V at kv_offset and writes the Q output. The entrypoint's output behavior and dynamic offsets are preserved.

## Three transformations in P15

1. **Direct VRF intermediates.** Input/head RMS scalars and RoPE sine products use the SDK 0.8.1 Main-to-VRF output instead of commit-to-DM then Sub reload. Arithmetic, sqrt, EPS, rounding and the physical slice distribution are preserved. Where a VRF handle is reshaped, already initialized replicas are relabeled; this is not a transfer or an initialization operation.
2. **Fuse head gather and channel scale.** BF16 projected rows convert exactly to FP32 before the switch, multiply the same channel scales and cast back to BF16. The former standalone head gather only copied BF16 values; removing its materialization does not remove a numerical rounding boundary because its input was already BF16. The contraction and post-scale BF16 boundaries remain.
3. **Request exactly the needed RoPE replicas.** `Cluster=m![2]` explicitly broadcasts to both clusters. `Slice=m![8, 1 # 32]` explicitly initializes 8 live copies per cluster, at slices 0,32,...,224. These positions match `[Ns % 4, Gs, 1 # 32]` in wire order. Relabeling yields the query-head table; retaining Gs=0 yields K positions 0,64,128,192. Padding is never used to manufacture data. This differs from gathering one live padded row and pretending all head replicas exist.

## Numerical changes consciously excluded from the winner

P10's old A010-style parallel V statistics use pre-BF16 scaled values for RMS while retaining a BF16 numerator. It is algebraically related but changes a rounding boundary. It also compiles slower than the exact fused approach, so no such change is included in P15.

P09 and P17 preserve BF16 values but change FP32 summation association across slices. Both are slower than the starting baseline and are not included. P16b's explicit input reductions are statically neutral on the p14 schedule and are not part of P15.

The baseline's two-component FP8 representation is itself approximate; preserving it is not a proof of exact equality to arbitrary FP32 reference arithmetic. Root's full official fixture and independent-input stress tests remain required. There is no fixture-specific branch, seed lookup, dropped function, CPU result substitution, altered scoring, or modified submission harness in this candidate.
