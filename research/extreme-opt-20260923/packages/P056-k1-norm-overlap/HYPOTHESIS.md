P056 attempted to remove the HBM store/reload round trip in `qkv69382.rs::normalize_native_input`
(`normalized0.view().to_hbm_view(...)` then `normalized_hbm.to_dm(...)`), replacing it with a direct
`unsafe { normalized0.reshape() }` into the `Shards` mapping.

RESULT: BUILD FAILED (does not compile).
  error: furiosa-opt: visa: while lowering LowLevelReshape
         caused by: a reshape may change a tensor's Cluster extent only as a pad or an unpad
         caused by: pad: the requested view type is not the padded view.
                    Padding Qs / 2048 % 2 to 2 produces Qs / 2048, but the requested view type is 1 # 2
  at src/device/sliding/qkv69382.rs:883

Root cause: `normalized0` has cluster extent `m![Qs/2048 % 2]` (= 2, the QKV dual-query cluster)
while the target `normalized` needs `m![1 # 2]` (the base 2-cluster split). These are two DIFFERENT
2-cluster decompositions; the hardware/compiler only permits cluster-extent changes via pad/unpad,
not via reshape. The HBM store/reload is therefore MANDATORY to cross the cluster mapping — it is
not a removable redundancy. Source was reverted to the P041 baseline; the package is a dead end.

LESSON: inter-cluster remapping in K1 requires an HBM handoff; the round trip is structural, not
a scheduling artifact. Do NOT retry this axis. The remaining K1 opportunity (if any) must come from
reducing the arithmetic inside `normalize` or `quantize`, not from eliminating the DM/HBM/DM handoff.

STATUS: REVERTED. No submission. P056 documents a hardware constraint, not a tuning result.
