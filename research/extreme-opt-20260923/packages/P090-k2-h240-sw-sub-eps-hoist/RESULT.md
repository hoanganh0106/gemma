# P090 H/240 sw-Sub plus epsilon-hoist

Parent: P041-k2-direct-tail-dma. Changes: move `sw` producer to Sub and move `EPS_SCALED` before the H-way sum, adding `EPS_SCALED * H_F32` before reduction and retaining `/H` afterward. All inputs, projection boundary, RMS arithmetic, residual, and gather topology are preserved.

Static gate: PASS, 21,290 cycles / 42 instructions. Schedule SHA256: `8DCEF840DE0F5805590EFF6D7142CFFE04A4D053E575A71C464DBEDE8A2A9A45`. Binary SHA256: `4D6DE74C69F15B882047DCC601A70FE7A441A025CB68463FFBA72A0D596E4735`.

Arena 92584 passed 15/15; K2 median 37,798 cycles `[37,675, 37,962, 37,798]`. Adjacent P041 control 92589 passed 15/15; K2 median 37,979 cycles `[37,930, 37,979, 38,343]`. P090 wins this pair by 181 cycles (0.48%); one pair is insufficient for promotion.

Reversed-order pair: P041 job 92599 passed 15/15 at 38,180 cycles `[38,180, 38,534, 37,666]`; P090 job 92607 passed 15/15 at 37,755 cycles `[38,377, 37,110, 37,755]`. P090 wins by 425 cycles (1.11%). Across both P090 pairs, candidate mean is 37,776.5 versus control mean 38,079.5, a 303-cycle / 0.80% edge. This is a stronger H/240 signal than the first pair, but more interleaving is still needed before promotion.

Third pair: P090 job 92614 passed 15/15 at 38,128 cycles `[38,128, 37,927, 38,288]`; P041 job 92617 passed 15/15 at 37,872 cycles `[38,014, 37,835, 37,872]`. P090 loses this pair by 256 cycles (0.68%). Across all three pairs, P090 mean is 37,893.7 versus P041 mean 38,010.3, an aggregate edge of 116.7 cycles / 0.31% for P090. The signal remains positive but small and noisy; keep P090 as a candidate, not a promoted best.

Direct pair against P078: P090 job 92618 passed 15/15 at 38,009 cycles `[38,165, 38,009, 37,996]`; P078 job 92619 passed 15/15 at 37,747 cycles `[37,747, 37,628, 37,782]`. P078 wins by 262 cycles (0.69%). This indicates the tail/120 topology currently has the stronger runtime signal than the H/240 epsilon-hoist variant.
## Fresh rebuild and Arena validation — job 95152

- Source was rebuilt cleanly with SDK 0.8.1 after the stale manifest was detected.
- Binary SHA256: `4D6DE74C69F15B882047DCC601A70FE7A441A025CB68463FFBA72A0D596E4735`.
- Arena result: `SUCCEEDED`, all 15 checks passed.
- K1 median: `81275` cycles.
- K2 median: `37935` cycles, samples `[37935, 37718, 38397]`.
- K3 median: `216358` cycles.
- Result artifact: `arena-95152.result.txt`.

The fresh H/240 measurement remains slower than the P158 H/120 frontier; no promotion.
