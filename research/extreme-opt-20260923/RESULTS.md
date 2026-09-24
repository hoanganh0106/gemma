# Extreme kernel optimization results — 2026-09-23

## Main-workspace selection: P005 fixed rings

P005 remains the byte-verified kernel tuple in the main workspace. The isolated screening campaign currently has a score-leading K1 replacement (`qkv69382`) with P005 K2/K3, but it is not promoted to the main source or official submission. Candidate decisions use several independent signals: (1) semantic and official-fixture correctness, (2) exact source/binary/fixture provenance, (3) compiler static makespan, instruction count, resource timeline and mapping outcome, (4) repeated paired Arena measurements, and (5) a source-level explanation for any disagreement. Static schedule is a compiler-model signal, not a substitute for device measurements; Arena runs remain noisy and are not promoted from one favorable sample.

The integrated candidate is frozen at `packages/P005-fixed-rings`. Its device source hashes are:

- K1 `qkv_head_local.rs`: `fd035d3a19dab1a88f2278cbd2603beb4f84bf8dcc96a43858e11aefd5213b12`
- K2 `output31.rs`: `a4803e5982db94e082af9c9581226d1fda0c2451c4c5a969d0726e5302bad416`
- K3 `ffn7.rs`: `b75a1f551b4e299e29dc4723c021f171673febb6503ada6a163e11b6c89b1211`

P005 binary SHA256: `fdd026bee39375b11a60e4df8a1478e04ef88f8903bc86037a14d55c9d2ae6a6`.
Official fixture SHA256: `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`.
SDK 0.8.1, locked dependencies, official three-seed host harness, wrapper and public signatures are unchanged. Full build compiled all three kernels from the frozen source tree; executable freshness was false (newly built), Cargo manifest path matched P005, and the release build lock was held through hashing.

## Official-harness Arena evidence

| Frozen package / job | Status | K1 median | K2 median | K3 median | Checks |
|---|---:|---:|---:|---:|---:|
| Rebuilt baseline / 78725 | SUCCEEDED | 93,444 | 44,705 | 221,469 | 15/15 PASS |
| P002 fewer commands / 78359 | SUCCEEDED | 90,292 | 44,016 | 222,562 | 15/15 PASS |
| P003 custom ring / 78470 | FAILED | K1 86,154 | K2 stalled | not reached | K1 9/9 PASS; device left K2 request unanswered after 7 s |
| P005 fixed rings / 78726 | SUCCEEDED | 87,205 | 39,632 | 215,976 | 15/15 PASS |
| Rebuilt baseline rerun / 78755 | SUCCEEDED | 91,306 | 45,008 | 221,885 | 15/15 PASS |
| P005 rerun / 78756 | SUCCEEDED | 86,349 | 39,299 | 216,898 | 15/15 PASS |

P005's complete medians against rebuilt baseline 78725 yield a geometric-mean runtime improvement of **7.42%** for this adjacent job pair: `((93444/87205)*(44705/39632)*(221469/215976))^(1/3)-1`. All three kernel medians improved in that pair. The three-run samples also improved for all K2/K3 runs; K1 samples are close at one edge. This is an observed package comparison, not a guarantee about all Arena runs or an official leaderboard score. The contest baseline denominators in the earlier API snapshot are illustrative only.

The repeated pair (78755/78756) measures **7.40%** geometric-mean improvement by the same formula. All K1, K2 and K3 run samples in this pair are lower for P005 than the control samples. This repeat supports the first pair and reduces concern that the result came from one unusually favorable set of three runs. No per-kernel minima from separate jobs are combined.

## Diagnostic input range check

Separate diagnostic packages use a host harness and matching references that vary K1 input independently of its RMS weight, K2 input magnitudes, and K3 signed residual, layer scalar, and global scales. These fixture and host-harness changes are not present in P005 or any official candidate.

- Baseline diagnostic 78740: K1 passes all three runs. K2 fails run 0 for the `±0.001` activation (`max tolerance ratio 4.240482`). K3 fails run 2 under the `±3` activation, `layer_scalar=0.875`, and assigned endpoint global scales (`ratio 1.090487`). The original kernels therefore already exceed reference tolerance in two extreme combinations outside the official fixtures.
- P005 diagnostic 78746: the failing K2 and K3 outputs have the **same maximum deltas and tolerance ratios** as baseline 78740; K1 also has the same errors. P005 does not increase the measured numerical error in these comparisons. Both jobs end with the same 2-of-3 diagnostic test failures.
- Moderate generic K1 diagnostic 78222 passed all 15 output comparisons with independent x and non-unit input RMS weights.

Diagnostic failures are disclosed rather than represented as general correctness passes. P005's official harness passes all 15 checks.

## Semantic invariants

- **K1** keeps `input_rms_weight`: computes weighted RMS normalization on the live input, then dynamic two-component FP8 quantization. It retains every Q/K/V matrix and scale, Q/K RMS weights, V normalization, dynamic RoPE offset/tables, and all outputs/cache writes. The A010 source that omitted use of input weight is not transplanted.
- **K2** keeps both FP8 activation levels, projection BF16 boundary, all channel scales, RMS weight, epsilon, sqrt/div, residual and final BF16 output. It gathers all 16 initialized partials; only their FP32 addition grouping changes.
- **K3** retains pre-RMS, packed up/gate/down weights, all block and global scales, both FP8 levels, GeGLU, post-RMS weight, residual and runtime layer scalar. It gathers all eight initialized post-RMS partials; only FP32 summation association changes. Direct VRF transfers replace FP32-only DM round trips.
- Independent source review found no concrete dropped-input or initialized-slice mapping issue in C22/v20. The full audit is `SEMANTIC_AUDIT.md`, with per-kernel ledgers/audits in `k1/`, `k2/`, and `k3/`.

## Rejected or superseded evidence

- P001 direct-VRF package 78286 passed all 15 checks, but K1 first-pair median was slower than its control; it was superseded.
- P002's 40-command reduction in K3 did not improve K3 runtime against rebuilt control; do not select it for that change.
- P003's custom ring gather compiled but the device stalled in K2; no score is assigned. P005 uses a fixed broadcast mapping and passed the full harness.
- K3 one-tile down path lost overlap; explicit both-RMS rings were slower than post-only. Paired up/gate increased weight-DMA time from 24,612 to 48,676 cycles. SRAM peer bridge and scale flattening also regressed or were neutral.
- K1 A010-style V parallelism, full 512-way RoPE broadcast, and KV stripe were slower than p15 or rejected by SDK mapping. Keep weighted normalization intact.
- K2 60x512 weight tile, local-scalar cluster route, and generic x broadcast regressed; padded-source fixed broadcast was rejected by the compiler. C22 uses 16 actual initialized partials.

## Build provenance correction

Two early candidate builds pointed to a shared-target executable marked fresh and were quarantined before submission. `campaign.py` now forces a changed library crate to rebuild, rejects fresh executables, verifies the Cargo manifest path, and holds one lock through artifact copying and hashing. It records and checks crate artifact hashes before any library reuse. P000, P003, P005 and diagnostic binaries listed above have distinct recorded manifests and hashes. The extension test binaries use the same candidate kernels and a deliberately separate diagnostic host harness/fixture.

The finite compiler search across 21 K1 variants, 22 K2 candidates and 22 K3 variants is complete. Static schedules chose candidates for hardware screening; only frozen whole-package Arena results determine runtime selection. No official MOA last submission has been replaced by this private screening campaign.

## Main workspace promotion

After the two passing public Arena pairs and independent source audit, the three P005 kernels were copied into the main workspace and line endings were normalized to the workspace's original LF convention. The resulting root source hashes are K1 `1fd417f78581c26eca2a05653921931426dae7f14c227b90b2da40d945ae58f6`, K2 `52fd11980eb24abe64298f6535f7e1b630fc0d96b89068f40cbe0271772eb54d`, and K3 `b75a1f551b4e299e29dc4723c021f171673febb6503ada6a163e11b6c89b1211`.

The main workspace was rebuilt cleanly with `cargo furiosa-opt build --release --locked --bin test_kernels`. Its binary SHA256 is again `fdd026bee39375b11a60e4df8a1478e04ef88f8903bc86037a14d55c9d2ae6a6`, byte-identical to P005's Arena-tested executable. `git diff --check` passes for the three kernel files.

## Follow-up isolated parallel screens

These screens ran in unique package directories; none modified the active workspace source or submitted to MOA.

- K1 P013 explicitly staged the K and V matrices before projection, but the compiler retained the P005 K1 schedule at **39,698 / 136**. The K and V DMA intervals did not move. K1 P014 (interleaved KV lanes) and P015 (eight-row KV tile) were rejected by the SDK's packet and DM-allocation constraints. No Arena run was justified for these.
- K2 P014 ring4 was **24,974 / 43**, worse than P005's C22 **22,317 / 42**. P015 ring8 was **22,400 / 42**; P016's interleaved FP8 lanes were rejected. P017/P020 layout probes failed packet-width or SRAM-alignment limits; P018 was **22,321 / 42**. C22 remains the strongest K2 schedule screen. Weight DMA remains the main bottleneck.
- K3 P013–P015 tile reorder and weight prefetch variants, P018's alternate FP8 mapping, and P019's interleaved down-scale lane fold all compiled at **108,244 / 169**, equal to P005's schedule. P016 had an unavailable SDK enum; P017's 6+9 row split failed physical mapping. No K3 candidate from these screens has Arena evidence.
- K2 P023 tested VRF accumulation around the C22 half contraction. The compiler rejected `vector_fp_binary` after `vector_intra_slice_reduce` (`IntraSliceReduce` has no `CanTransitionTo<Fp>`), so no schedule or binary was emitted. It is a mapping/compiler dead end, not a runtime result; C22 remains **22,317 / 42**. Candidate source hash and compiler log are preserved in `packages/P023-k2-q-half-double-buffer/`.

## K1 candidate comparison: qkv69382 with P005 K2/K3

The distinct variant archived as `packages/P005-K1-081-K2K3-P005-final` changes K1 implementation and `src/ops.rs` dispatch only. Inspection confirmed its public `sliding_project_qkv` path calls `qkv69382::normalize_native_input` with `input_rms_weight`; that function calls the weighted RMSNorm helper before quantization. K2 and K3 sources match P005. It has a slower static K1 makespan (**42,556 cycles / 189 instructions**), so the runtime result is specifically hardware evidence and must not be inferred from the schedule.

The exact binary SHA256 is `76cd70d40f541b1caf63f73c88ad166e39bddd09b6cb90d3c664d395116a913a`; K1 source `qkv69382.rs` is `de2d84a4d582861bc08b2758e3a380e5e397ada9db9e21998da32f6a7103f2bd`; dispatch source `ops.rs` is `dae27e42419a7c29c3d00d6733faaafa1e321f2680db726240f0d30677b075dc7`. Fixture SHA256 is unchanged at `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`.

Four complete official Arena runs for this exact tuple passed all 15/15 comparisons. Candidate jobs **79102**, **79169**, and **79308** measured respectively `80,940 / 39,733 / 216,394`, `79,328 / 40,280 / 217,662`, and `80,166 / 40,347 / 215,456` (K1/K2/K3 medians). Two adjacent P005 control/candidate pairs also passed 15/15: **79166** `86,065 / 40,072 / 216,647` → **79169** `79,328 / 40,280 / 217,662`, and **79307** `87,000 / 39,808 / 216,231` → **79308** `80,166 / 40,347 / 215,456`. K1 fell by **7.83%** and **7.86%** in the two direct pairs; whole-package geometric runtime improved **2.42%** and **2.43%**. K2/K3 shifts were small and inconsistent, as expected for unchanged kernels. Earlier P005 controls 78726, 78756, 78861, 78871, and 78962 also put K1 medians between 86,349 and 88,324.

The evidence points in different directions: qkv69382 is **2,858 static cycles slower** and has 53 more instructions than head-local P005, while two adjacent Arena pairs show an almost identical K1 win and package-score gain. Source inspection gives a plausible mechanism: this path keeps weighted input RMSNorm but uses one FP8 activation term instead of P005's two-term dynamic quantization, reducing projection work while adding normalization/remapping work. The one-term path satisfies every official output tolerance in four full runs. Since Arena cycles are the graded metric, select qkv69382 as the **current score-leading isolated K1 choice**, while treating its worse static schedule as a real model discrepancy—not a claimed schedule improvement. Keep the active workspace and MOA submission unchanged.

The older arena log copies in `P005-K1-081-K2K3-P005-final` predate that directory's rebuilt binary and are not evidence for this tuple. The authoritative logs are `arena-79102.log` and `arena-79169.log` in that package and `arena-79166.log` in `P005-fixed-rings`.

## Follow-up screening after P005

P006 removes the unused Sub warm-up in K3; arithmetic, data mappings, official harness and fixture are unchanged. Its K3 source SHA256 is `6e745a4ccd27b173bbeeba0b8edf5a9f4100bebbbd755cc2238ff64d26edfb1b`. Static schedule is **108,253 / 167**, nine cycles slower than P005 (**108,244 / 169**).

Two adjacent P005/P006 Arena pairs all passed the full 15/15 public checks:

| Pair | P005 K1/K2/K3 medians | P006 K1/K2/K3 medians | Package geometric runtime change |
|---|---|---|---:|
| P005 job 78861 → P006 job 78864 | 88,031 / 39,734 / 217,642 | 86,867 / 39,653 / 215,879 | P006 +0.79% |
| P006 job 78868 → P005 job 78871 | 90,642 / 40,046 / 216,403 | 88,324 / 39,772 / 216,503 | P006 -1.07% |

Across these two small pairs, the K3-only geometric median shifts suggest about **0.43% fewer cycles** without the warm-up, but one pair is nearly flat and the whole-package result changes sign. This is within the observed noise and the compiler schedule is slightly slower. P006 is therefore recorded as an experiment and **not promoted**; P005 remains the tested source in the main workspace. Raw logs and parsed medians are in the P005/P006 package directories.

Third pair, P006 job 78959 → P005 job 78962: P006 medians `87,095 / 39,990 / 217,009`; P005 medians `87,610 / 39,779 / 215,886`; both 15/15 PASS. P006 is **0.15% slower** on the package geometric mean and 0.52% slower in K3. Across all three pairs the per-pair K3 effects alternate in size and the package log-mean indicates no gain; keep P005 selected.

P010 explored using only the leading FP8 activation term in K3 down projection. Static schedule was **108,247 / 157**, only three cycles slower than P005 despite twelve fewer instructions. The frozen binary hash was `213c40e003f3eec8b8db705277de967c160c143f7b5f66461e598a76fc0a418e`; K3 source hash was `1b9a34314da3d10de8486b34bee998c7397d4d42825b7859c3d517d9485715ac`. Arena job 78942 failed K3 on all three seeds (15/15 checks not passed; total 12/15). The control job 78938 passed. P010 is rejected, and its cycle measurements are not treated as a valid performance result.

P011 explored passing the existing BF16 GeGLU result as one activation term. SDK 0.8.1 does not provide direct `f4e2m1 -> bf16` table lookup for down weights (`TableLookupCast<bf16>` is not implemented); adapting it would require another materialization/conversion pass, so this candidate did not reach a valid schedule or Arena test. No source from P006–P011 was promoted; the main source stays at P005.

P007 moves the independent `gate_factor_vrf` preparation before projections; P009 moves down-scale loading before GeGLU. Both compile, but the emitted per-instruction timing sequence matches P005 exactly (108,244 / 169); neither has Arena evidence and neither justifies a runtime claim. P008 combines early gate preparation with no warm-up and matches P006's 108,253 / 167 timing. These source-only reorder probes remain isolated from the main workspace.

## K1 P016: native weighted RMSNorm screen

P016 keeps `input_rms_weight` live and performs the same weighted RMSNorm on the native QKV shard mapping, removing the normalized-vector HBM round trip. The reduction and accumulation association changes, so Arena correctness was required. Static K1 schedule improved from **42,556 / 189** to **41,260 / 182**. Binary SHA256 `fc230521f2e71757c8b77ebf1bcdf9258e8fd1a03a60e23f21d3aaee2d8eb127`; changed K1 source SHA256 `a9c19f6f38163fd3035dc7b06eb11f40652cbe8aeFD7BBD5E1298807FB46E504` (case-insensitive); K2/K3 sources remained `a4803e59…bad416` / `b75a1f55…b1211`.

Both P016 public Arena jobs passed 15/15: job **79239** measured `79,255 / 39,652 / 216,987`, and job **79247** measured `80,322 / 39,693 / 218,147`. Their qkv69382 controls, also 15/15 PASS, were job **79169** at `79,328 / 40,280 / 217,662` and job **79245** at `79,657 / 40,312 / 217,068`. The paired directions disagree on K1 (P016 0.09% faster in one, 0.83% slower in the other); across both pair medians P016 is about 0.37% slower on K1. The lower static schedule therefore did not yield a repeatable hardware gain. Keep qkv69382 as the current score-leading K1 choice; P016 is a correctness-passing but non-promoted experiment.

K3 P020 changed only the down-scale lane mapping to `Interleaved`, but after full integration the binary SHA stayed byte-identical to qkv69382 + P005 K2/K3 (`76cd70d4…a913a`). Do not submit a duplicate binary. Further P021/P022 scale-fold mapping probes either failed packet layout or retained the same schedule and were not Arena-tested.

## Continued screening from the user-selected P005-final parent

The user-selected best parent is `packages/P005-K1-081-K2K3-P005-final`, binary `76cd70d40f541b1caf63f73c88ad166e39bddd09b6cb90d3c664d395116a913a`. The following packages copy its source and public harness directly. The independent read-only P025/P028 audit found no dropped input or invalid consumed slice: full runtime RoPE rows are broadcast to eight initialized query-head copies per cluster; the KV mapping consumes initialized Gs=0 copies. `input_rms_weight` and all projection code remain live. P028's FP32 RMS storage change preserves its arithmetic and BF16 boundaries.

- P024 direct head-RMS Main-to-VRF: **42,141 / 186** versus parent **42,556 / 189**. Source `d3609b3708a3488aa27538fa7e0ac8f8e32dc0372524d869ac0d2910a0cb5696`. Static-only, combined into P028 for hardware screening.
- P025 full-table live-head RoPE: **41,075 / 131**. Source `c057126fb50b5d0ec36d1a06cbddd18fa1c2c97e3654aa93c3fad23639cac740`; binary `998b66b2b4c7b5937f4fdd1a411fbd34ab6b8775fc48a57cbe59120a35a7d54b`. Job **79397** passed 15/15 at `80,002 / 39,862 / 216,033`. The preceding parent **79392** passed at `80,090 / 39,823 / 216,061`. A 0.11% K1 shift is not a repeatable gain; no promotion.
- P028 combines P024 and P025: **40,314 / 128**, 5.27% below parent static. Source `5dc7331b87b9358e13782e5f51b617d873209c0b0f06ce2e8e4021ba7b948e59`; binary `db39b1943d430edd1e36703986cf3db1a5eda0728b379d623ac5204764eeaa8c`. Jobs **79402** and **79424**, both 15/15 PASS, measured `80,056 / 39,871 / 216,623` and `79,508 / 39,788 / 218,167`. Following parent controls **79406** and **79444** were `79,533 / 40,250 / 216,844` and `79,375 / 40,525 / 215,687`, also 15/15 PASS. K1 is 0.66% and 0.17% slower in these pairs. Static improvement has not produced a repeatable runtime win; no promotion.
- P030 contiguous per-slice projection rows removes FP32 staging and three row transposes, but changes HBM ownership. Static **42,372 / 186**; source `02c21bf30b3e75c0227e48c6514d960f9408cf3fb60efa1b1a4cbeb1573fa4ad`; binary `c2258209cbb3fdc5bf477294c9a0b680ba344e6df1a2b3fa4e9d99673f2d36f4`. Job **79451** passed 15/15 at `92,769 / 39,964 / 216,955`, decisively slower K1 than nearby parent **79444**. Reject. This demonstrates that the hardware cost of changing row ownership can dominate a small static improvement; retain striping and test narrower distribution changes instead.
- K2 P026 packet2 and P027 Interleaved-lane probes were rejected by the compiler's contraction packet/layout checks. No schedule, binary or Arena evidence exists for these changes.
- K3 P027 compact-partial attempt changed `Pw` without changing the producer's padded extent, causing reshape rejection. P031 changes producer extent, alias and consumer stride consistently, and compiles at **112,111 / 169** versus **108,244 / 169**. Source `c1816d923f7a193cdb5dd3cd07e633e86db0d3b3e648c8d49b34695c4feb117c`. Its partial HBM store takes 4,382 modeled cycles; reduced 128-byte row strides are slower despite less traffic. Reject before full build/Arena.

The official public leaderboard snapshot fetched from its API is preserved in `leaderboard-20260923-current.json`. At retrieval, rank 1 was IA -ARIA ON THE PLANETES-, score 9.155112220304128, K1/K2/K3 `73,366 / 33,020 / 201,948`, SDK 0.8.1. This is a target-setting snapshot, not a controlled comparison with Arena. No active workspace source or MOA submission was changed by this screening batch.
