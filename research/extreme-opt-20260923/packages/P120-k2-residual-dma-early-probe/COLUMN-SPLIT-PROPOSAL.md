# Column-split contract family

## Hypothesis

Row-splitting the H/120 weight tile cannot preserve the ring32 consumer mapping because SDK 0.8.1 requires the committed slice size to remain unchanged. Split along `Qs` instead: each weight half keeps all H/120 rows and half of the Qs columns; each x half uses the matching TRF columns; two contracts accumulate into the same full-size `z` mapping.

## Why this is a distinct family

- The output slice remains H/120, so the ring32 gather sees the original ownership.
- The reduction is over Qs, so the two contracts need an explicit BF16 accumulation path.
- TDMA can potentially load the second weight half while Main contracts the first half.
- The cost debt is extra x materialization, one extra contract, and partial accumulation.

## Cost model

| Component | Expected effect | Classification |
|---|---:|---|
| Weight DMA overlap | hide part of the 13,383-cycle load | potential |
| Second contract | extra Main/vector work | intrinsic cost |
| x half extraction | extra layout/materialization | temporary or unknown |
| z accumulation | extra BF16/f32 combine | intrinsic or unknown |
| Ring32 tail consumer | unchanged mapping | preserved invariant |

## Gate sequence

1. Prove a half-Qs `XTrf` and `WeightColumnHalf` can be represented with packet size 32 bytes.
2. Prove two contracts can accumulate into a full-size `Z` without changing its slice mapping.
3. Generate static schedule and compare the added contract cost with the DMA overlap floor.
4. Only then run numerical correctness and paired Arena hardware tests.

The first microprobe should be the accumulation API/type-state, before copying the full P078 package.
