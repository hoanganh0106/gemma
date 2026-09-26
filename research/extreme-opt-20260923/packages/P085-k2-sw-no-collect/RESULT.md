# P085 sw no-collect probe

Parent: P078-k2-tail120. Removed only the `collect::<H/8%15, H%8>()` relayout before the `sw` vector pipeline.

Compiler gate: rejected. The fetched/cast position cannot transition directly to `vector_init`; SDK 0.8.1 requires the collect relayout to establish a vector-compatible position. No binary or runtime measurement was made.
