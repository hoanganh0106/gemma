## P208 final multiply order plus explicit RMS view - Arena job 101812

- Source change: P205 final multiply order combined with P199 explicit `rms_dm` view materialization.
- Binary SHA256: `1F8C1F8B47A87036A319C3C3859A8CF78F4541D8CD4EE618A8F745B97714D512`.
- All 15 correctness checks passed.
- K1 median `80528`; K2 median `37635` (`[37635,37454,38513]`); K3 median `216997`.
- P208 is 89 cycles slower than adjacent P166 control job `96843` at K2 `37546`; retain as near-frontier but do not promote.

## 2026-09-26 recheck

- P166 control job 102139: PASS 15/15; K2 37668, samples
  `[37321, 37668, 38188]`.
- P208 job 102145: PASS 15/15; K2 38191, samples
  `[38191, 37544, 38814]`.
- P208 lost the adjacent recheck by 523 cycles. Do not promote.
- Result artifact: `arena-101812.result.txt`.
