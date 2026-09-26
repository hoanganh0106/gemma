P064 isolates the quantization logic from P059. It keeps the weighted RMS
normalization and BF16 materialization exactly as P059, but removes only
vector_logic(BitAnd, -0.5) from the second FP8 quantization pass.

Goal: determine whether BF16 rounding alone is sufficient for correctness and
whether removing the logic op reduces hardware latency. input_rms_weight remains
live through weight.to_dm -> weight_vrf -> MulF before quantization.
