# P126 Integrated Build Recheck

Date: 2026-09-25

The integrated P005 K1/K3 plus P122 K2 package was rebuilt with Furiosa SDK
0.8.1:

```text
CARGO_INCREMENTAL=0 cargo furiosa-opt build --release --locked --bin test_kernels
```

The release binary is 3,567,920 bytes with SHA256:

```text
76CD70D40F541B1CAF63F73C88AD166E39BDDD09B6CB90D3C664D395116A913A
```

The runtime retry reached device initialization but stopped at
`Device(Topology("No such file or directory (os error 2)"))`, before graph
execution. This artifact is therefore build provenance only, not numerical
correctness evidence.
