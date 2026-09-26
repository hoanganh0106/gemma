# P120 K2 residual DMA early probe

The residual HBM-to-DM allocation was issued before `contract_tile`, while
keeping its fetch on Sub and all mappings unchanged.

Static compile passed with **21,337 cycles / 42 instructions**, identical to
P078. The compiler did not expose additional overlap; the weight DMA,
contract, and tail spans remain unchanged. This is a neutral issuer-order
probe and has no reason to proceed to runtime.
