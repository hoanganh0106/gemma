# P122 K2 direct z reshape probe

## Hypothesis

Reuse the projection result allocation directly as the tail layout with an
`unsafe reshape`, removing the explicit `z.to_dm` redistribution.

## Static result

The Furiosa compiler accepted the reshape and emitted:

```text
20,495 cycles / 37 instructions
```

P078 is **21,337 cycles / 42 instructions**. The static reduction is **842
cycles (3.95%)** and removes the 446-cycle `z.to_dm` span plus associated
instructions. The remaining long spans are the weight DMA
`1936..15319` and contract `15319..16693`.

## Risk and next gate

This is only a static result. The reshape crosses the projection's
`OutputClusters/Rows` ownership into the tail's `Vc/Tail` ownership, so the
unsafe mapping must pass graph construction and all-three correctness before
any performance claim. Do not promote from the schedule alone.

The release test binary built successfully, but runtime graph construction
currently fails with the same baseline error seen in rebuilt P078:

```text
gather src residue has no live target axis
```

Therefore P122 has no correctness or hardware result yet; the 20,495-cycle
figure remains static screening evidence only.

## Recompile confirmation (2026-09-25)

The isolated source was recompiled with SDK 0.8.1 using:

```text
CARGO_INCREMENTAL=0 cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedule-p144.json
```

The compiler emitted **20,495 cycles / 37 instructions**. This confirms that
the direct `z.reshape()` path is stable in the current isolated package. It is
not a new optimization over P122.

## Artifact recheck (2026-09-25)

The package directory also contains three completed Arena summary artifacts
that pass the full public harness (15/15 checks each):

```text
arena-91914.summary.json  K2 median 37810
arena-91916.summary.json  K2 median 37872
arena-91905.summary.json  K2 median 37916
```

These are hardware evidence for the submitted package artifacts and are kept
separate from the local rebuilt-harness failure above. They do not prove that
the current local rebuild is identical to every submitted artifact, so P122
remains the hardware frontier rather than a promoted source result.
