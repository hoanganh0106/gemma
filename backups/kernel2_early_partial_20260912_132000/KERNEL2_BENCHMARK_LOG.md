# Kernel 2 benchmark log

Workspace: D:\Project\furiosa-opt-gemma4-12B-main

## E5 early partial add

- Source change: compute `p01` immediately after `p0`, `p1`; then compute `p2`, `p3`, `p23`.
- Arena jobs: 23903 = 339376; 23921 = 332864; 23922 = 336175 cycles.
- Median: 336175 cycles.
- Correctness: all 3 kernels PASS.
- Status: candidate retained for further comparison; backup baseline remains in `backups/kernel2_current_20260912_131556`.

