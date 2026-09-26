# P078 static recheck — 2026-09-25

The current P078 source was compiled with:

```text
CARGO_INCREMENTAL=0 cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule /tmp/p078-live-schedule.json
```

The compiler reported:

```text
Compiling [1/1] ops::sliding_attention_output
Finished 1 compiled, 14 not matching ops::sliding_attention_output
```

This separates the static compiler path from the runtime graph failure. The
schedule file was emitted in the WSL temporary directory; its cycle summary
still needs to be copied out after the transient WSL access error clears.
