# Explicit eight-slice RMS gather

Source SHA256: `84B6EB69B2D02050B43EF64764F088A480C46549A99F0EE534A9942C5F342BD1`.
Schedule SHA256: `D5BFED4BCAA7DA54345BE83D2B8527C221070AC2820D809745E921F88A4B6C31`.

The frozen parent uses H=3840. After projection it retains a BF16 tensor `z` which is 16 times the unscaled projection. Channel scale `s`, RMSNorm weight `w` and residual `r` are runtime inputs.

For each of the eight 480-value groups, the unchanged first pass computes

`p[g] = sum_h_in_group((z[h] * s[h])^2) / 3840`.

The original next pass used `vector_inter_slice_reduce` to obtain `sum_g p[g]` in each group, then applied `epsilon * 16^2`, sqrt and division. The new pass uses the Switch Engine with ring size 8 to route all eight p values into the time dimension of each active output group. Intra-slice reduction sums that dimension. A separate scalar pass applies the original epsilon, sqrt and division.

The final pass is unchanged:

`BF16(z[h] * (s[h] * w[h]) * inv_rms + residual[h])`.

Every H value participates once in one group partial and once through that partial in the global sum. All eight groups receive the same global inverse RMS. All runtime channel scales, RMS weights, residual values, and the epsilon are used. Both FP8 activation levels and the projection BF16 materialization remain unchanged. No input is recognized, replaced or cached across calls.

The physical rename from slice axis Vr to H/480 has the same index order and size (eight), as in the parent. The mean tensor already contains an explicit copy of the same global value for every Vr index; the rename does not invent initialized data.

FP32 sum grouping differs from the original reduction engine, so hardware reference tests and stress fixtures remain required. Compiler success alone does not establish numerical correctness.

## Static result

- Parent direct-VRF C03: 24,409 cycles / 41 instructions.
- C14: 22,400 cycles / 43 instructions.
- Former inter-slice reduction plus scalar arithmetic: 2,579 cycles.
- Explicit gather plus intra-slice sum: 289 cycles.
- Standalone unchanged scalar arithmetic: 281 cycles.
- Net static reduction: 2,009 cycles (8.23% relative to C03; 9.38% relative to frozen 24,718-cycle baseline).

This isolates the improvement to reduction topology rather than approximating the normalization function.
