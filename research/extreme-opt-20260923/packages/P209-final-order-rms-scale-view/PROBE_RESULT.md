# P209 hardware probe

## Mechanism

- Base: P208 final multiply order plus explicit RMS destination view.
- Add an explicit destination allocation and `to_dm_view` for `weight_scale`.
- Binary SHA256:
  `bb671de725119ab6cd2a19e0b879045b90ee6394ea59a5047d15fa3cf0afce81`.
- Fixture SHA256:
  `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`.

## Arena evidence

- Job 102190: PASS 15/15; K2 37424, samples `[37199, 37424, 37514]`.
- Reverse P166 control job 102195: PASS 15/15; K2 37625, samples
  `[37657, 37552, 37625]`.
- Confirmation job 102201: PASS 15/15; K2 37874, samples
  `[38163, 37482, 37874]`.
- Earlier P166 control job 102139: PASS 15/15; K2 37668, samples
  `[37321, 37668, 38188]`.

## Decision

Do not promote. P209 won the first reverse pair by 201 cycles but lost on the
confirmation run. The mean of the two P209 medians is 37649; the mean of the two
P166 control medians is 37646.5. These observations do not establish a hardware
gain. Two job medians per binary are insufficient to establish equivalence or
attribute the difference to noise. P166 remains the measured frontier; P209 is
an open development base, not a rejected architecture family.

## Follow-up mechanism investigation

P209 retains P166's contraction, redistribution, and mean DM materialization.
Its three source changes are explicit scale and RMS destination allocations,
and swapped final multiply operands. A changed executable hash alone does not
prove a changed device schedule, lower traffic, or lower tail latency.

D003-p209-device-profile preserves all P209 device sources and instruments only
the host trace collector. Compare per-cluster dominant DMA duration and the
remaining task window against D002. Nested Cluster spans are not additive costs.
Check redistribution/reuse synchronization before attempting another view tweak.

Potential removable debt: intermediate mean DM commit/fetch, and any proven
allocation-induced wait. Intrinsic dependencies: the global RMS needs all row
partials, and final normalization needs that scalar. Ownership/broadcast repair
cost is unknown. P076's fused gather expression failed SDK stage typing; P129 and
P130 failed consumer geometry. Those failures constrain implementations, not the
entire family. No numeric ideal floor is established yet.
