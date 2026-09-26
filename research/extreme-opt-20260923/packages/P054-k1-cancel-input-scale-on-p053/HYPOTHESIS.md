# P054 hypothesis

Base: P053-k1-native-scale-on-p051.

The weighted input RMSNorm remains intact: normalize_native_input still divides x by its RMS and multiplies by input_rms_weight before FP8 quantization. The candidate removes only the post-contraction multiplication by the positive common input quantization scale. Q and K are immediately head-RMS-normalized, and V is immediately RMS-normalized, so a common positive scalar should cancel at that stage apart from finite-precision/EPS effects. The quantization scale is still used to encode the two FP8 input components.

This is experimental until fixture correctness and Arena hardware results pass.
