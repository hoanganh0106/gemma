from pathlib import Path
p=Path('research/extreme-opt-20260923/k3/v26-f8e5m2-down/ffn7.rs')
s=p.read_text()
a=s.index('fn quantize_gather('); b=s.index('fn down_scales_bf16(',a)
part=s[a:b].replace('f8e4m3','f8e5m2')
s=s[:a]+part+s[b:]
a=s.index('macro_rules! down_tile {'); b=s.index('pub(crate) fn feedforward(',a)
part=s[a:b].replace('TrfTensor<f8e4m3,','TrfTensor<f8e5m2,').replace('fetch_table_lookup::<f8e4m3>()','fetch_table_lookup::<f8e5m2>()')
s=s[:a]+part+s[b:]
p.write_text(s)
