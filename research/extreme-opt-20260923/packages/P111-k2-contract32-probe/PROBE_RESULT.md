# P111 K2 contract32 probe

## Hypothesis

Reduce the contract outer grouping from `Qs%64` to `Qs%32` while retaining the
single full H/120 weight DMA and output owner.

## Result

Static compile passed with **21,341 cycles / 42 instructions**. The long spans
are:

```text
weight DMA       1936..15319  (13383)
contract         15319..16697 (1378)
tail work        17759..18943 (1184)
```

Compared with P078 at 21,337 cycles / 42 instructions, the smaller grouping
adds 4 cycles to the contract and 4 cycles to the total. It provides no cost
advantage and is rejected before runtime.
