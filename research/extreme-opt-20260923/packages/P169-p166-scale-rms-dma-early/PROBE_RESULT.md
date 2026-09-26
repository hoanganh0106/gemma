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
## Fresh Arena repeat — job 94891

- label: `p158-repeat-frontier-20260925b`
- binary SHA256: `390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6`
- result: `SUCCEEDED`, all 15 checks passed
- K1 median: `80138` cycles
- K2 median: `37601` cycles, samples `[37601, 37460, 38151]`
- K3 median: `215887` cycles
- Result artifact: `arena-94891.result.txt`

This is a fresh hardware sample for the P158 frontier. It supports the existing correctness-preserving P158 lineage; performance remains reported as a sample distribution because Arena noise is material.
## Reverse-order Arena candidate — job 94949

- label: `p158-reverse-candidate-20260925`
- result: `SUCCEEDED`, all 15 checks passed
- K1 median: `79295` cycles
- K2 median: `37763` cycles, samples `[37763, 37733, 38915]`
- K3 median: `216141` cycles
- Result artifact: `arena-94949.result.txt`

Against reverse-order control job 94947, P158 is lower by `2129` K2 cycles (`5.3%`). Combined with adjacent pair 94891/94926, this supports P158 as the current measured K2 frontier.
## Arena validation — job 95414

- label: `p166-packet32-20260925`
- clean binary SHA256: `5965D4748E2428EF04A59223599BC7A67A6BC8AF7728DDC9CCDCA29A4321422F`
- result: `SUCCEEDED`, all 15 checks passed
- K1 median: `82690` cycles
- K2 median: `37357` cycles, samples `[37183, 37357, 37694]`
- K3 median: `216882` cycles
- Result artifact: `arena-95414.result.txt`

This packet32 coupled mapping is the first new binary after the P158 frontier and is currently the measured K2 leader. It requires an adjacent control and a reverse-order pair before promotion.
## Paired Arena confirmation

- Adjacent pair: P153 control job `95428` K2 `39876`; P166 job `95430` K2 `38147`.
- Reverse-order pair: P166 job `95440` K2 `38053`; P153 control job `95441` K2 `40303`.
- All four jobs passed 15/15 correctness checks.
- P166 wins both pairs by `1729` and `2250` cycles respectively. The packet32 coupled candidate is promoted as the current measured K2 frontier over P158.
- Result artifacts: `arena-95430.result.txt`, `arena-95440.result.txt` and paired controls under P153.
## Third Arena pair

- P166 job `95532`: all 15 checks passed; K2 median `38010`, samples `[37881, 38438, 38010]`.
- P153 control job `95533`: all 15 checks passed; K2 median `39789`, samples `[40799, 39161, 39789]`.
- P166 wins this pair by `1779` cycles (`4.5%`). Across three adjacent/reversed pairs, P166 remains the measured K2 frontier.
- Result artifacts: `arena-95532.result.txt` and control `arena-95533.result.txt`.
## Scale/RMS early DMA probe

- Issued `scale` and `rms` DMA before the projection contract while keeping their VRF consumers after the contract.
- Clean SDK 0.8.1 build passed.
- Binary SHA256 `5965D4748E2428EF04A59223599BC7A67A6BC8AF7728DDC9CCDCA29A4321422F`, identical to P166.
- The scheduler canonicalized this DMA order; no Arena submission was made.
