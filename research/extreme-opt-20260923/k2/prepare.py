from pathlib import Path
import shutil,json
root=Path(__file__).resolve().parent
baseline=(root.parents[1]/'furiosa-score-20260923/probes/k2-direct-vrf/src/device/sliding/output31.rs').read_text()
def sw(s,sub=False):
    start=s.index('    let sw_dm:')
    end=s.index('    let z_tail:', start)
    part=s[start:end]
    part=part[:part.index('        .commit_trim')] + ('        .to_vrf();\n\n' if sub else '        .to_vrf(&mut device.sub);\n\n')
    part=part.replace('let sw_dm: TailDm<f32>','let sw: TailVrf')
    if sub: part=part.replace('.main\n', '.sub\n')
    return s[:start]+part+s[end:]
def clean(s):
    return s.replace('    let out: DmTensor<bf16, Chip, m![1 # 2], Tail, m![H % 480]> = unsafe { out.reshape() };\n','')
def invsub(s):
    start=s.index('    let inv_rms: VrfTensor')
    end=s.index('    // Same physical',start)
    part=s[start:end].replace('.main\n','.sub\n').replace('.to_vrf(&mut device.sub)', '.to_vrf()')
    return s[:start]+part+s[end:]
def tail(s,n):
    s=s.replace('Vr = 8','Vr = '+str(3840//n))
    s=s.replace('m![1 # 32, H / 480]',f'm![1 # {n//15}, H / {n}]')
    s=s.replace('m![1 # 32, Vr]',f'm![1 # {n//15}, Vr]')
    s=s.replace('H % 480',f'H % {n}')
    s=s.replace('H / 8 % 60',f'H / 8 % {n//8}')
    s=s.replace('H / 4 % 120',f'H / 4 % {n//4}')
    return s
candidates={
 'C01-invrms-vrf':(baseline,'Port exact prior direct VRF inverse RMS.'),
 'C02-sw-vrf':(sw(baseline),'Direct Main SW producer into VRF; preserve multiplication.'),
 'C03-sw-vrf-clean':(clean(sw(baseline)),'C02 plus remove identity output reshape.'),
 'C04-sw-sub':(clean(sw(baseline,True)),'Compute SW using Sub vector context directly into VRF.'),
 'C05-inv-sub':(invsub(clean(sw(baseline,True))),'C04 plus inverse RMS on Sub.'),
 'C06-tail240':(tail(clean(sw(baseline,True)),240),'C04 with 16 groups of 240 values; same global H sum.'),
 'C07-tail960':(tail(clean(sw(baseline,True)),960),'C04 with 4 groups of 960 values; same global H sum.'),
}
for name,(s,note) in candidates.items():
    d=root/'candidates'/name;d.mkdir(parents=True,exist_ok=True)
    (d/'output31.rs').write_text(s)
    (d/'intent.txt').write_text(note)
names=list(candidates)
(root/'names.json').write_text(json.dumps(names))
