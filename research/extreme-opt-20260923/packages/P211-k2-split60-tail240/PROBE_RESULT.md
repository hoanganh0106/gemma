# P211 hardware probe

## Mechanism

- Base: P166 packet32 coupled Kernel 2.
- Split the 120-row weight into two 60-row DMA/contract passes.
- Keep the P077 tail240/ring16 consumer so the two contract outputs can feed the
  consumer directly without P210's canonical Z copy.
- Goal: overlap the first contract with the second weight DMA while avoiding the
  ownership-repair copy.

## Build evidence

- Static schedule: 22820 cycles (P166: 21341).
- First weight DMA: 1936..8902.
- First contract: 8902..9800.
- Second weight DMA: 8902..15868.
- Compiler-model overlap: 898 cycles; not a measured hardware overlap.
- Binary SHA256:
  `863e92bb494d0160a3da6c4bbe551efcc2a7c97007f9cb3f5700632d083b9f9b`.

## Arena evidence

- Job: 102123.
- Correctness: PASS 15/15.
- K1: 79523 cycles.
- K2: 39522 cycles, samples `[39522, 39505, 40067]`.
- K3: 216202 cycles.
- Raw result: `arena-102123.result.txt`.

## Decision

Reject as a performance candidate. It is about 2000 cycles slower than the P166
hardware controls (~37546-37697). Removing P210's canonical copy recovered about
1100 cycles, but the 898-cycle overlap does not repay the split DMA/contract and
tail-layout overhead. P166 remains the frontier.

The aggregate hardware regression does not separately measure these overheads.
It rejects this candidate for promotion, not every split-weight architecture.
A device trace is needed to attribute the regression and bound removable debt.

## Device trace follow-up

Diagnostic job 102320 passed all checks. Its device source is byte-identical to
this package. See `../D004-p211-device-profile/PROBE_RESULT.md` and its raw trace.
The two weight DMAs are serial on hardware; the first contract overlaps the
second DMA by 1624–2446 cycles. The two transfers sum to 26013–27161 cycles,
and the task still runs 9944–12196 cycles after the second DMA ends. This
confirms a real but insufficient overlap mechanism. A new producer/consumer
dataflow must move local tail work ahead of the final weight transfer if this
family is pursued toward 30000 cycles.
