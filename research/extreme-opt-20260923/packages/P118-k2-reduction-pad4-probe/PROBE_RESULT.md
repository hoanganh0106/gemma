# P118 K2 reduction pad4 probe

The probe reduced the RMS partial reduction padding from `1#8` to `1#4` and
updated its commit trim accordingly. The SDK passed MIR but VISA rejected the
commit shape:

```text
commit: input does not match pipeline
declared InSlice ... = 4
pipeline InSlice ... = 8
```

The downstream ring32 consumer requires the original `1#8` shape. The pad is
therefore an ownership/layout requirement, not removable local overhead. No
schedule or runtime result was produced.
