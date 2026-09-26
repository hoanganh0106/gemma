# P112 K2 contract16 probe

The `Qs%16` outer grouping was rejected by the SDK before scheduling:

```text
mir: OutPacket must be 1 or 2 flits (32 bytes each), got 16 bytes
```

The output packet cannot be smaller than one flit for the existing H/120
commit mapping. Together with P110 (`Qs%128`, 128 bytes) and P111 (`Qs%32`,
21,341 cycles), this closes the simple Qs grouping sweep around P078's
`Qs%64` choice. No runtime test was warranted.
