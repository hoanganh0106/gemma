# P210 split60 with canonical Z

- Mechanism: split the P166 weight transfer into two 60-row slabs, overlap the second DMA with the first packet32 contract, then materialize a same-layout canonical `Z` owner before the ring32 consumer.
- Clean binary SHA256: `275096C9F8811297CC8A4D8D7EEACC27E18921BD94A320D86937D5B56E9885F4`.
- Compiler schedule: `22899` cycles. The second weight DMA overlaps the first contract for about `898` cycles; the canonical `Z` copy and split command overhead increase the complete schedule.
- Arena job `102080` passed all 15 correctness checks.
- K1 median `81892`; K2 median `40619` (`[40788,40329,40619]`); K3 median `217638`.
- Ownership is solved, but hardware performance regresses by about 3K cycles versus the P166 control range. Do not promote.
- Result artifact: `arena-102080.result.txt`.
