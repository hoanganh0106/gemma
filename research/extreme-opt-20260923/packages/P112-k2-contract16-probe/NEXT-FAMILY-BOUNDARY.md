# K2 next family boundary — 2026-09-25

## Evidence

- P078: **21,337 cycles / 42 instructions**.
- P109 declaration reorder: **21,337 / 42**; source order does not expose
  overlap.
- P106 H-row split: **22,895 / 56**; two weight DMA spans plus duplicated
  contract work increase the makespan.
- P108 column half primitives compile, but no runtime graph result is
  attributable while the frozen P078 control itself fails graph preparation.

## Decision

The next candidate must change the weight packet/layout or reduce the number
of bytes issued on the existing 120-row path. More source-order changes and
H-row duplication are ruled out by the static schedule.

## Gate for the next probe

1. Preserve one full H/120 output owner and one contract consumer.
2. Change only weight packet/layout or an SDK-supported view that reduces DMA
   duration.
3. Require static makespan below 21,337 before any runtime attempt.
4. Revalidate against a clean P078 control once graph preparation is healthy.
