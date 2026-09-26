# P121 K2 explicit z DM view probe

The projection result was materialized into a preallocated `TailDm` with
`to_dm_view` instead of the direct `z.to_dm` helper.

Static compile passed with **21,337 cycles / 42 instructions**, identical to
P078. The allocation spelling does not change the DMA lifetime or scheduler
graph. No runtime test was warranted.
