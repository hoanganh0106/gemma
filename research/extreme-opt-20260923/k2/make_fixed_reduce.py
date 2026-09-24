from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C15-ring16-tail240/output31.rs').read_text()
assert s.count('SwitchConfig::CustomBroadcast { ring_size: 16 }')==1
s=s.replace('SwitchConfig::CustomBroadcast { ring_size: 16 }','SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 }')
d=root/'candidates/C22-fixed-rms-ring16';d.mkdir(parents=True,exist_ok=True)
(d/'output31.rs').write_text(s)
(d/'intent.txt').write_text('C15 with fixed Broadcast1 slice1=16,slice0=1 for all-gather of sixteen scalar partials. Based on verified K1 Broadcast1 scalar-gather mechanism; identical reduction and scalar arithmetic.')
