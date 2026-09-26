# P116 K2 contract lane interleaved probe

## Exploration

The SDK does expose `LaneMode::Interleaved`. The first attempt exposed the
required mapping changes:

- `OutPacket` must be `m![Lq # 8]`;
- `OutTime` must be `m![H % 120]`.

After applying those mappings, the compiler progressed past MIR but the VISA
lowerer rejected the downstream reduction:

```text
visa: while lowering TensorUnit
caused by: reduce axis should not have padding
```

## Decision

Interleaved changes the contraction ownership enough that P078's padded
`Lq`/H reduction sequence is no longer legal. It needs a new reduction and
commit layout; this is a separate architecture family, not a local lane-mode
swap. No schedule or runtime result was produced.

Removing the pre-reduction `vector_narrow_trim` does not provide a local fix:
the following `vector_intra_slice_reduce` is then unavailable at that vector
stage. Interleaved therefore needs a redesigned vector stage and output
mapping.

The repository's legal Interleaved examples confirm the distinction: they
commit directly after `contract_lane` and do not perform the K2 `Lq` intra-slice
reduction. K2 has `Lq=2` and must reduce those two lanes before the H/120
commit, which is the source of the padding incompatibility. Reusing the
Interleaved pattern would change the arithmetic ownership, so it is not a
drop-in optimization for K2.
## Exact Lq=2 reduction probe

- Kept `LaneMode::Interleaved` and changed the pre-reduction trim/output from `1 # 4` to `1 # 2`.
- Clean compiler reached VISA lowering but still rejected `sliding_attention_output` with `reduce axis should not have padding`.
- This rules out the local trim width as the cause; interleaved requires a redesigned ownership/reduction layout.
- No schedule, binary, or Arena submission was produced.
