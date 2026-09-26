# P115 K2 contract lane parallel probe

The probe attempted to replace `LaneMode::Sequential` with
`LaneMode::Parallel` for the H/120 contract. SDK 0.8.1 rejected it at Rust
compile time:

```text
no variant, associated function, or constant named `Parallel` for LaneMode
```

The SDK exposes no parallel lane mode for this operation. No schedule or
runtime result exists; P078's sequential lane mode is the only legal enum
choice in the current toolchain.
