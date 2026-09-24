from pathlib import Path
root=Path(__file__).resolve().parent
for old,new in [('C18-x-fused-broadcast','C20-x-fused-fixed'),('C19-x-bf16-broadcast','C21-x-bf16-fixed')]:
 s=(root/'candidates'/old/'output31.rs').read_text()
 assert 'SwitchConfig::CustomBroadcast { ring_size: 256 }' in s
 s=s.replace('SwitchConfig::CustomBroadcast { ring_size: 256 }','SwitchConfig::Broadcast1 { slice1: 16, slice0: 16 }')
 d=root/'candidates'/new;d.mkdir(parents=True,exist_ok=True)
 (d/'output31.rs').write_text(s)
 (d/'intent.txt').write_text(old+' with fixed Broadcast1 16x16 topology instead of generic CustomBroadcast for x replication; identical initialized source slices and mathematical operations.')
