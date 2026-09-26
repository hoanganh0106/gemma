# P083 sw widen-pad probe

Parent: P078-k2-tail120. Only the `sw` producer's `vector_widen_concat` was replaced with a same-shape `vector_widen_pad` attempt.

Compiler gate: rejected at `to_vrf()` with `pruning valid value is not allowed`. Padding cannot preserve the valid H/120 lane mapping for this product tensor; no binary or runtime measurement was made.
