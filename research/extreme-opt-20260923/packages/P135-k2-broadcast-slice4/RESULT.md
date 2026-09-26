# K2 architecture family: tail 120 on P041

Parent: P041-k2-direct-tail-dma. Only `src/device/sliding/output31.rs` changed. Full fixture and entrypoint copied from the parent; fixture SHA256 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`. P078 binary SHA256 `37622e7935a9e9fd730f8485c0c495e24685ce67f2fd8081fb434c2b05488da1`.

Hypothesis: map the post-projection tail as 120 rows per group across 32 groups (8 x 32 physical slices), reducing the final per-slice vector pass while preserving direct on-chip projection redistribution. Mathematical dependencies and all inputs are retained. The group reduction has more partials.

Cost model (whole K2 compiler schedule): P041 21,290 cycles / 42 instructions; P078 21,337 / 42. The P078 final Main pass ends 44 cycles sooner (19,955 vs 19,999), but additional reduction/layout overhead makes total 47 cycles worse. Reduced final-pass duration is an intrinsic potential gain. Larger gather and mapping cost is intrinsic to this particular group geometry. No added HBM traffic is intended. Hardware impact must be measured.

Compiler gate: PASS under furiosa-opt-std 0.8.1. Full release build: PASS. Public Arena jobs 91905, 91914, 91916 all PASS 15/15.

Hardware K2 medians (cycles): P078 91905 = 37,916 [37,154, 37,916, 38,012]; adjacent P041 control 91914 = 37,810 [37,578, 37,851, 37,810]; P078 91916 = 37,872 [38,058, 37,872, 37,631]. P078 loses the adjacent pair by 62 cycles (0.16%). This is below measurement noise and does not support promotion. The earlier 91905 to historical P041 83329 comparison is not paired and is not promotion evidence.

Decision: retain P041 as K2 control. The 120-row family is legal and fixture-correct, but current implementation has no repeatable hardware win. A further reduction of the final pass would have to exceed the gather/layout cost and hardware noise; pure Rust command reordering (P061) already emitted a byte-identical binary. No official submission made.

Follow-up probe: removing `vector_widen_pad` after the cross-slice reduction was rejected by SDK 0.8.1 stage typing; `vector_final` cannot follow `IntraSliceReduce` directly. The required padding is compiler/API debt for this layout and remains in P078.

## Epsilon hoist follow-up

The source was then changed to add `EPS_SCALED * H` to each squared element before the H-way reduction and remove the scalar epsilon add after the mean. This is algebraically equivalent before the division by H and produces a distinct binary `40B1F097D638F876E4FBA6D2E4E3A33EB9AA4804D3B353C96D3E8E7EA2A75BF4`.

The exact schedule remained 21,337 cycles / 42 instructions. Full Arena correctness passed 15/15 in jobs 92013, 92021, and 92037. K2 medians were 37,776; 37,378; and 37,666 cycles. Adjacent P041 controls 92022 and 92038 were 37,793 and 37,828; the earlier paired control 91914 was 37,810 against candidate 91916 at 37,872. Across the three paired comparisons, epsilon-hoist won two and lost one; mean candidate minus control was approximately -171 cycles (-0.45%).

Decision: retain epsilon-hoist as the active K2 frontier for further refinement, with P041 as the control. The hardware signal is promising but still close to noise; no main-workspace replacement or official submission has been made.

Direct-VRF follow-up: SDK source shows `vector_final().to_vrf()` is legal, but the next `Context::begin` accepts only `DmTensorView`, not a VRF view. The attempted direct `mean` VRF handoff therefore failed type checking at the consumer boundary and was restored. This confirms the remaining mean DM materialization is an API/dataflow boundary in SDK 0.8.1, not an overlooked method spelling.

## `sw` on Sub follow-up

The `sw = scale * rms_weight` pass was moved from `Main` to `Sub`. The compiler accepted it and emitted the same static schedule, 21,337 cycles / 42 instructions, but a distinct release binary `390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6`.

Arena job 92076 passed 15/15 with K2 median 37,659 cycles `[37,669, 37,659, 37,622]`. The adjacent P041 control job 92086 passed 15/15 with K2 median 38,130 cycles `[38,130, 38,032, 38,539]`. A second `sw` sample, job 92087, passed 15/15 at 37,887 cycles `[37,887, 37,480, 38,155]`. The two `sw` samples average 37,773 versus the adjacent control 38,130, a 357-cycle / 0.94% difference; the first sample is favorable, so this remains a promising frontier rather than a final promotion claim.

The change keeps all inputs and arithmetic live. It changes only which TU performs the existing scale-times-RMS-weight pass; the equal static schedule shows the gain is a hardware scheduling/issuer effect.

Third paired check: P041 job 92098 passed 15/15 at K2 38,292 cycles `[38,292, 37,865, 38,508]`; `sw`-Sub job 92097 passed 15/15 at K2 38,216 cycles `[38,023, 38,216, 38,322]`. The candidate won this pair by 76 cycles. The `sw`-Sub family remains the active frontier and is still being measured against P041 because the gain is small relative to Arena noise.

Negative probe: moving the residual fetch/cast/collect from Sub to Main compiled with the same 21,337-cycle static schedule, but Arena job 92121 deadlocked at the start of `sliding_attention_output` (`rank 0 left request 6 unanswered for 7s`). It is rejected and the source is restored to the verified Sub residual path.

Negative static probe: moving the primary `scale` fetch/cast/collect to Main required a Main-to-Sub VRF handoff and increased the exact schedule to 21,638 cycles / 42 instructions. It was rejected before hardware testing; `scale` remains on Sub.

Prescale family: replacing the RMS `/H` with `scale / sqrt(H)` was tested in two forms. Normalizing the scale VRF compiled but failed K2 correctness in Arena job 92146 with max errors about 4.1--4.9. Moving the `1/H` multiply after the two existing multiplies was rejected by the SDK lowerer because the vector ALU operand was unavailable. The source is restored to epsilon-hoist plus the original `/H`; the verified `sw`-Sub source compiles again.

Clean artifact refresh: removed two unused prescale constants, recompiled the exact `sw`-Sub source with SDK 0.8.1, and rebuilt `test_runtime-sw-sub-clean`. The exact schedule remains 21,337 cycles / 42 instructions; schedule SHA256 is `B4E3A74105D0C380469D782B15F4F923800AB5D9789B61705ACAE1A96EF1816F`; binary SHA256 is `BC3748BD27224C394F9267103B0987EB9CBE3609229179F40D76DA1A1FA28EF3`.

Architecture probe: recompiling the P043 local RMS partial/reduction path under the same SDK produced 23,310 cycles / 51 instructions. It is materially worse than the active P078 path and is rejected before hardware measurement.

API inventory: SDK 0.8.1 exposes only `FpUnaryOp::Sqrt` in the repository's FP unary paths; no `Rsqrt`, reciprocal, or inverse-square-root primitive is available to replace the current `Sqrt + Div` construction.

Runtime provenance correction: job 92241 used a stale prescale binary (`BC3748...`) despite the restored source and failed K2 correctness with max errors 4.12--4.92; it is rejected as an artifact mismatch. After `cargo clean`, a full release rebuild produced binary SHA256 `390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6`, matching the previously verified sw-Sub binary. Arena job 92265 then passed all 15/15 checks; K2 median was 37,530 cycles `[37,771, 36,909, 37,530]`.

Paired control: fresh P041 job 92269 also passed 15/15 with K2 median 37,869 cycles `[37,641, 37,869, 38,226]`. Against candidate 92265, P078 wins by 339 cycles (0.895%) in this adjacent pair. This is stronger evidence for the sw-Sub frontier, but it remains one pair and is not yet a promotion decision.

Reversed-order pair: P041 job 92274 passed 15/15 at 37,687 cycles `[37,687, 37,132, 38,601]`; P078 job 92279 passed 15/15 at 37,975 cycles `[37,975, 37,076, 38,341]`. P078 loses this pair by 288 cycles. Across the two fresh pairs, candidate mean is 37,752.5 versus control mean 37,778, a negligible 25-cycle / 0.066% aggregate edge; retain P078 as frontier but do not claim a stable speedup.

Third fresh pair: P078 job 92383 passed 15/15 at 37,423 cycles `[37,031, 37,423, 38,368]`; P041 job 92386 passed 15/15 at 37,892 cycles `[37,892, 37,730, 38,006]`. P078 wins by 469 cycles (1.24%). Across all three fresh pairs, P078 mean is 37,642.7 versus P041 mean 37,816, an aggregate 173.3-cycle / 0.46% edge. The pair variance is still material, so this supports continued investigation rather than promotion.

Fourth reversed-order pair: P041 job 92413 passed 15/15 at 37,942 cycles `[37,636, 37,942, 38,171]`; P078 job 92419 passed 15/15 at 37,637 cycles `[37,637, 37,413, 38,444]`. P078 wins by 305 cycles (0.80%). Across four fresh pairs, P078 mean is 37,641.3 versus P041 mean 37,847.5, an aggregate 206.3-cycle / 0.54% edge. Keep the candidate as the measured frontier; the distribution still does not justify official promotion.

Fifth pair: P078 job 92521 passed 15/15 at 37,862 cycles `[37,105, 37,862, 38,024]`; P041 job 92523 passed 15/15 at 37,911 cycles `[38,056, 37,737, 37,911]`. P078 wins by 49 cycles. Across five fresh pairs, P078 mean is 37,685.4 versus P041 mean 37,860.2, an aggregate 174.8-cycle / 0.46% edge; continue treating this as a measured frontier, not a proven promotion.

Gather mapping probe: changing the tail-120 cross-slice switch from `Broadcast1 slice1: 32` to `16` was rejected by the mapper (`Switch OutSlice mismatch`, expected `(H / 1920 # 16, 16)`, got `Vr # 256`). The 32-wide broadcast is required by this ownership/layout; source restored and compile verified.
