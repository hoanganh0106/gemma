# P136 K2 custom broadcast32 probe

## Hypothesis

Replace P122's `Broadcast1 { slice1: 32, slice0: 1 }` mean gather with the
SDK's `CustomBroadcast { ring_size: 32 }` while keeping the same logical output
shape.

## Static result

The custom mapping is legal and compiles:

```text
21,289 cycles / 38 instructions
```

This is 794 cycles and one instruction worse than P122's 20,495 / 37. No
runtime or Arena test was run.

## Interpretation

CustomBroadcast32 is a valid alternate implementation of the gather family,
but its mapper/resource schedule has higher cost. It remains useful as a legal
architecture reference; it is not a new frontier.
