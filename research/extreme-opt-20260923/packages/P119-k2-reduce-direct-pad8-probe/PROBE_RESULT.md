# P119 K2 direct reduction pad8 probe

The probe tried to make `vector_intra_slice_reduce` produce the required
`1#8` output directly, removing `vector_widen_pad`. The SDK did not provide a
legal reduction stage after the existing `1#4` narrow mapping; compilation
failed with unavailable/ambiguous `vector_intra_slice_reduce` implementations
at that stage.

This confirms the pad is a required stage transition in the current API. P118
also showed that simply changing the trim shape breaks the downstream
pipeline. No schedule or runtime result was produced.
