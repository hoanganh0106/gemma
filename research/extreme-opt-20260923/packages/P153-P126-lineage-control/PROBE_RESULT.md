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

P126 was submitted after refreshing its stale source hash in `manifest.json`.
The immutable submission record is `submission-p126-k2-reshape-20260925.json`.

```text
job: 94046
status: SUCCEEDED
full public harness: PASS
pass count: 15/15
medians: K1 80720, K2 40613, K3 216801
```

This is the first current P126 hardware result. The K2 result is a valid
all-three package measurement; compare it against paired controls before
promoting it as a performance win.

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

## Paired Arena control (2026-09-25)

P153 was rebuilt with the parent K2 source and submitted with a fresh binary.

```text
job: 94266
status: SUCCEEDED
full public harness: PASS
pass count: 15/15
medians: K1 79888, K2 39736, K3 217727
binary_sha256: 76CD70D40F541BCAF63F73C88AD166E39BDDD09B6CB90D3C664D395116A913A
```

Against P126 job 94200, the paired control passes all checks while the
reshape candidate fails `sliding_attention_output` in all three runs. The
36,973-cycle P126 K2 median is therefore not a valid performance result.
## Adjacent Arena control — job 94926

- label: `p153-adjacent-control-20260925`
- binary SHA256: `76CD70D40F541BCAF63F73C88AD166E39BDDD09B6CB90D3C664D395116A913A`
- result: `SUCCEEDED`, all 15 checks passed
- K1 median: `80928` cycles
- K2 median: `39713` cycles, samples `[41045, 39493, 39713]`
- K3 median: `217223` cycles
- Result artifact: `arena-94926.result.txt`

This is an adjacent fresh control for the P158/P159 hardware samples. P158 job 94891 was lower by 2112 K2 cycles in this window; an order-reversed pair is still needed before treating the gap as stable.
## Reverse-order Arena control — job 94947

- label: `p153-reverse-control-20260925`
- result: `SUCCEEDED`, all 15 checks passed
- K1 median: `81022` cycles
- K2 median: `39892` cycles, samples `[40939, 39596, 39892]`
- K3 median: `216786` cycles
- Result artifact: `arena-94947.result.txt`
