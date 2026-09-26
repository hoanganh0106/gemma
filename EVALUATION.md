# Stage 1 Evaluation Rules

This file records the current Round 1 evaluation update. The evaluator runs
`src/bin/test_kernels.rs` for each submission.

## Correctness gate

Each of the three kernels is tested with three independently seeded input cases. All nine
checks must pass. A required correctness failure makes the submission invalid and removes it
from the leaderboard, regardless of its cycle count.

## Performance and score

For each kernel, sort the three RNGD cycle counts and use the middle value (the median).
The Stage 1 score is the geometric mean of the three kernel speedups:

```text
score = ((B1 / S1) * (B2 / S2) * (B3 / S3)) ** (1/3)
```

`Bi` is the baseline median and `Si` is the submission median for kernel `i`. Compare only
within the same `furiosa-opt-std` version. Both `0.6.0` and `0.8.1` are supported; `0.8.1`
is recommended for its runtime improvements and stability.

Do not use a single run, a best-of-N value, or medians from a different number of input cases
as an official score. Do not combine the fastest K1, K2, and K3 values from different jobs.

## Round 2

Approximately 70% of participating teams are expected to advance. The current expected cutoff
is in the 3x-5x range; this is an advancement estimate, not a change to the score formula.
