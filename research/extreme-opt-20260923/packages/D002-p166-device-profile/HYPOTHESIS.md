# P166 device profile

Diagnostic only. Device source, inputs, references, and output comparisons are unchanged from P166. The host collector logs each NPU span's name, cluster, begin cycle, and end cycle with trace profiling enabled.

Trace instrumentation may perturb timing, so its cycle counts are used to locate waits and overlap. Normal Arena runs remain the performance evidence.
