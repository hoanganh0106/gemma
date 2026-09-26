# P122 static manifest

- compiler: `cargo furiosa-opt` SDK 0.8.1
- operation: `ops::sliding_attention_output`
- result: `20,495 cycles / 37 instructions`
- control: P078 `21,337 cycles / 42 instructions`
- delta: `-842 cycles`, `-5 instructions`
- schedule JSON: [p122-schedule.json](p122-schedule.json)
- schedule SHA256: `6EB6B204F713E82DF8331D864E37FEC15F025373F289DA1DBEC36690000E4B71`
- instruction diff audit: P122 has 37 instructions versus P078's 42; the
  removed P078 span is exactly `z.to_dm` at `16693..17139` (446 cycles), with
  no replacement DMA instruction in P122.
- runtime gate: release binary built; graph execution hit the current control
  failure `gather src residue has no live target axis`
- K2-only rerun: `GEMMA4_ONLY_TEST=sliding_attention_output` reproduces the
  same graph failure, so the error is in the K2 graph path or its current
  runtime environment, not K1/K3 test ordering.
- P078 K2-only control was rebuilt and run with the same filter; it reproduces
  the identical error. P122 therefore has no demonstrated graph regression
  relative to the current control.
