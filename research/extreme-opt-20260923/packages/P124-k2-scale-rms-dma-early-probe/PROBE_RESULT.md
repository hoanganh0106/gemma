# P124 K2 scale/RMS DMA early probe

Starting from P122, the scale and RMS-weight DM allocations were issued before
the projection contract while their Sub fetches stayed in the original order.

Static compile passed with **20,495 cycles / 37 instructions**, identical to
P122. The TDMA dependency chain does not expose extra overlap; P122 remains
the static frontier.
