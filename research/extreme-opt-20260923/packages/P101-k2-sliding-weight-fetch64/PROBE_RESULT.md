# P101 K2 weight fetch 64 probe

P101 is the first correctly scoped K2 probe after the source audit. It changes `sliding/output31.rs` weight fetch/collect from `Qs / 32 % 8, Qs % 32` to `Qs / 64 % 4, Qs % 64`.

SDK 0.8.1 rejects the collect at compile time: `Packet size must be exactly one flit (32 bytes)`. The 64-element packet violates the hardware packet constraint. No schedule or Arena job was produced.
