# P082 tail160 topology probe

Parent: P078-k2-tail120. The tail row group was changed from 120 to 160 across projection, RMS, and gather mappings while preserving arithmetic and inputs.

Compiler gate: rejected. SDK 0.8.1 reports `DmTensor T27's slice extent 192 does not match the device config (slice = 256)` at the residual HBM transfer. The H/160 topology is incompatible with the fixed 256-slice device configuration; no binary or runtime submission was made.
