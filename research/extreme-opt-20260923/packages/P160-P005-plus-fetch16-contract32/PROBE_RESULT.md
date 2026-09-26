# P160 fetch16 plus contract32

## Result

The combined probe was rejected by SDK 0.8.1 during MIR validation:

```text
furiosa-opt: ... sliding_attention_output: mir: `contract_outer`: Time does not divide OutTime
```

The contract32 consumer cannot absorb the finer `Qs / 16 % 16` fetch layout.
No schedule or Arena binary was produced.
