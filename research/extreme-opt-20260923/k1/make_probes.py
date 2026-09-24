from pathlib import Path
import shutil, json

root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
base=(baseline/'src/device/sliding/qkv_head_local.rs').read_text()
def save(name,s,ops=None):
    p=root/name
    p.mkdir(exist_ok=True)
    shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
    for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'): shutil.copy2(baseline/f,p/f)
    (p/'src/device/sliding/qkv_head_local.rs').write_text(s)
    if ops: (p/'src/ops.rs').write_text(ops)
    return s

def rms_direct(s):
    a=s.index('    let rms: DmTensor',s.index('fn root_mean_square'))
    b=s.index('\n}',a)
    section=s[a:b]
    section=section.replace('    let rms: DmTensor<f32, Chip, HeadCluster, S, m![1 # 8]> = device.main','    device.main')
    c=section.index('        .vector_final()')
    section=section[:c]+ '        .vector_final()\n        .to_vrf(&mut device.sub)'
    return s[:a]+section+s[b:]

def rope_direct(s):
    a=s.index('    let sin_product: DmTensor')
    b=s.index('    // The cosine product',a)
    section=s[a:b].replace('let sin_product: DmTensor','let sin_product_vrf: VrfTensor')
    c=section.index('        .vector_final()')
    section=section[:c]+ '        .vector_final()\n        .to_vrf(&mut device.sub);\n'
    return s[:a]+section+s[b:]

def input_direct(s):
    a=s.index('    let rms: DmTensor',s.index('pub(crate) fn normalize_native_input'))
    b=s.index('    let weight_vrf:',a)
    section=s[a:b].replace('let rms: DmTensor','let rms: VrfTensor',1)
    c=section.index('        .vector_final()')
    section=section[:c]+ '''        .vector_final()
        .to_vrf(&mut device.sub);
    let rms_vrf: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> =
        unsafe { rms.reshape() };
'''
    return s[:a]+section+s[b:]

def rope_dm(s):
    a=s.index('    let cos_row: HbmTensor')
    b=s.index('    let q = rope_one',a)
    return s[:a]+'''    let q_cos: QueryHeadTensor = cos_row.to_dm(&mut device.tdma);
    let q_sin: QueryHeadTensor = sin_row.to_dm(&mut device.tdma);
'''+s[b:]

def trf_direct(s):
    a=s.index('    // Restore H order')
    b=s.index('    let trf: NativeTrf',a)
    return (s[:a]+s[b:]).replace('.begin(dense.view())','.begin(shared.view())')

save('p01-rms-vrf',rms_direct(base))
save('p02-rope-vrf',rope_direct(base))
save('p03-input-rms-vrf',input_direct(base))
save('p04-rope-dm',rope_dm(base))
save('p05-input-trf',trf_direct(base))
save('p06-all-vrf',input_direct(rope_direct(rms_direct(base))))

runner='''#!/usr/bin/env bash
set -u
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target
ROOT=/mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k1
for name in "$@"; do
 cd "$ROOT/$name"
 cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedule.json > compile.log 2>&1
 echo "$name exit=$?"
done
'''
(root/'compile.sh').write_text(runner)
