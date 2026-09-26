# P122 static manifest

- compiler: `cargo furiosa-opt` SDK 0.8.1
- operation: `ops::sliding_attention_output`
- result: `20,495 cycles / 37 instructions`
- control: P078 `21,337 cycles / 42 instructions`
- delta: `-842 cycles`, `-5 instructions`
- schedule JSON: emitted at compile time in WSL `/tmp/p122-z-reshape.json`; the
  temporary file was not persisted after the WSL session ended
- runtime gate: release binary built; graph execution hit the current control
  failure `gather src residue has no live target axis`
