# P114 K2 weight fetch64 probe

The fetch/collect layout was changed to `Qs/64%4, Qs%64` so it matched the
P078 contract outer grouping directly. The SDK rejected it before scheduling:

```text
Collect output packet must be exactly 32 bytes (one flit): 64 elements = 64 bytes
```

The weight fetch packet is constrained to 32 bytes even though the contract
outer grouping is 64 elements. The P078 two-level fetch/collect layout is
therefore required for this output mapping; no runtime test was warranted.
