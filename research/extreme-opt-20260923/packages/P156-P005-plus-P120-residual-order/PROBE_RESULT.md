# P126 P005 plus P122 K2 isolated package

## Scope

This package is an isolated all-three compile candidate. It keeps P005's K1
and K3 sources and replaces only its K2 `src/device/sliding/output31.rs` with
the P122 direct-reshape source. P005's original K2 module was a different
family, so this is an integration candidate rather than a one-file diff from
P005.

## Static compile

All three exact operations compiled with SDK 0.8.1:

```text
K1  42,556 cycles / 189 instructions
K2  20,495 cycles / 37 instructions
K3 108,244 cycles / 169 instructions
```

Schedule hashes:

```text
K1 8FBC017BF30F524593809EF44EDB145327F6F1028304EB7AD21844BD597A7569
K2 6EB6B204F713E82DF8331D864E37FEC15F025373F289DA1DBEC36690000E4B71
K3 02B0AA09FD85FD0AB46A50E717E3DD4C1E62B8ACCEFA56A42AE029B73D6EE372

K2 source hash (`src/device/sliding/output31.rs`):

```text
525E7E77D46829C91547AC5C7BBDE9C1B66D384244ABB16A2C56F5FB52D776B0
```

The schedule files are `p126-k1.json`, `p126-k2.json`, and `p126-k3.json`.
```

## Gate status

Before the current validation, this candidate was compile-only. The local
harness still hits `gather src residue has no live target axis` for the K2
runtime control and P122, so local correctness remains unproven.

## Arena validation (2026-09-25)

P126 was first submitted with a binary whose build provenance was later found
to be stale. That result is retained only as an audit record and is not used
as current evidence.

```text
job: 94046
status: SUCCEEDED (provenance suspect; superseded)
full public harness: not accepted as current evidence
pass count: 15/15
medians: K1 80720, K2 40613, K3 216801
```

The clean rebuild produced `test_runtime` SHA256
`50B7E4929718A47DA24D21161AC223862D9DE6D7666F450C8D263793D35A09EC`.

## Arena validation, clean binary (2026-09-25)

```text
job: 94200
status: FAILED
full public harness: FAIL
pass count: 12/15
medians: K1 80228, K2 36973, K3 216758
failure: sliding_attention_output, all 3 runs
```

This is the current P126 hardware result. K1 and K3 passed. The integrated
K2 path caused `sliding_attention_output` numerical failures, so this
candidate is rejected for promotion despite its lower K2 cycle median.

## Integration recompile (2026-09-25)

The integrated P005 plus P122 package was recompiled with SDK 0.8.1 for
`ops::sliding_attention_output`. The emitted schedule is **20,495 cycles / 37
instructions**, identical to standalone P122. This confirms the K2 reshape
path survives integration with the P005 K1/K3 sources. It remains compile
evidence only; no new Arena submission was made.

## Comparison caveat

The older P122 hardware summaries point to logs under the P078 package
(`P122/.../arena-91914.summary.json` records that path), while job 94046 is a
fresh P126 integrated binary. Their K2 medians must therefore not be treated
as a paired control comparison. The 40,613-cycle result is valid P126
hardware evidence, but promotion or regression attribution requires a control
built and submitted from the same current package lineage.

The same package also completed a clean release `test_kernels` build with SDK
0.8.1 (`cargo furiosa-opt build --release --locked --bin test_kernels`, 3m28s).
The runtime graph was not rerun after the build because the WSL service became
unavailable again; this build result does not change the existing correctness
status.

## Runtime retry (2026-09-25)

The fixture runner was retried with the freshly built `test_runtime` binary.
It reached device initialization but terminated with:

```text
Device(Topology("No such file or directory (os error 2)"))
```

This is a missing local device/topology runtime before graph execution. It
 does not confirm or refute K2 numerical correctness.

## Arena validation, clean P156 binary (2026-09-25)

```text
job: 94352
status: SUCCEEDED
full public harness: PASS
pass count: 15/15
medians: K1 79917, K2 37682, K3 216373
binary_sha256: 390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6
```

The residual-DMA order variant is hardware-correct. Its binary is identical
to P155, so the 33-cycle difference is run variance rather than a distinct
optimization.

## Arena validation, clean P155 binary (2026-09-25)

```text
job: 94321
status: SUCCEEDED
full public harness: PASS
pass count: 15/15
medians: K1 80300, K2 37715, K3 216040
binary_sha256: 390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6
```

The explicit `to_dm_view` variant is hardware-correct and beats the P153
control K2 median of 39,736 cycles by 2,021 cycles. It is slower than the
invalid reshape family, but is the current best correctness-preserving K2
candidate measured from a clean package.
