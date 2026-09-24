from pathlib import Path
import shutil, json

root=Path(__file__).resolve().parent
base=root.parent/'baseline'
work=root/'work'
work.mkdir(exist_ok=True)
if not (work/'src').exists(): shutil.copytree(base/'src',work/'src')
for name in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']: shutil.copy2(base/name,work/name)
source=(base/'src/device/shared/ffn7.rs').read_text()

def norm(s, post=False):
    start=s.index('pub(crate) fn normalize_add_gate_in_place(') if post else s.index('fn normalize_quantize(')
    end=len(s) if post else s.index('macro_rules! block_sums_all')
    q=s[start:end]
    first=q.index('    let rms: DmTensor')
    stop=q.index('    let weight_dm:', first)
    fragment=q[first:stop].replace('let rms: DmTensor','let rms: VrfTensor').replace('.commit_trim::<m![1 # 8]>()\n        .commit();','.to_vrf(&mut device.sub);')
    fragment=fragment.replace('let rms: VrfTensor<f32, Chip, Cluster, ReducingSlices','let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices') if post else fragment.replace('let rms: VrfTensor<f32, Chip, UpGateClusters, Pieces','let rms_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces')
    q=q[:first]+fragment+q[stop:]
    old_start=q.index('    let rms_vrf: VrfTensor',q.index('    let weight_dm:'))
    old_end=q.index('        .to_vrf();',old_start)+len('        .to_vrf();\n')
    q=q[:old_start]+q[old_end:]
    return s[:start]+q+s[end:]

def geglu(s):
    start=s.index('    let gelu: DmTensor')
    end=s.index('    device.main\n        .begin(up.view())',start)
    q=s[start:end]
    q=q[:q.index('    let gelu_vrf:')]
    q=q.replace('let gelu: DmTensor','let gelu_vrf: VrfTensor').replace('.commit_trim::<m![L % 8]>()\n        .commit();','.to_vrf();')
    return s[:start]+q+s[end:]

def tile(s,n):
    start=s.index('    // SDK 0.8.1 requires this DMA row factor')
    end=s.index('\n    // Two neighbouring slices',start)
    q=''.join(f'    down_tile!(device, down_weight_packed, x_trf, down_s, partial, {i}, {n});\n' for i in range(0,15,n))
    return s[:start]+q+s[end:]

variants={
 'v01-pre-rms-vrf':(norm(source),'Pre RMS Main-to-VRF, same arithmetic/mapping.'),
 'v02-post-rms-vrf':(norm(source,True),'Post RMS Main-to-VRF, same arithmetic/mapping.'),
 'v03-geglu-vrf':(geglu(source),'GeGLU Sub-to-VRF, same arithmetic/mapping.'),
 'v04-all-vrf':(geglu(norm(norm(source),True)),'Combine three direct VRF eliminations.'),
 'v05-down15':(tile(source,15),'One 15-row down tile instead of three5; less LUT provisioning, larger DM/TRF.'),
 'v06-down3':(tile(source,3),'Five3-row down tiles; tests memory/overlap tradeoff.'),
}
for name,(s,reason) in variants.items():
    d=root/name; d.mkdir(exist_ok=True)
    (d/'ffn7.rs').write_text(s)
    (d/'intent.txt').write_text(reason)
(root/'compile.sh').write_text('''#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k3
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target
for name in "$@"; do
  cp "$name/ffn7.rs" work/src/device/shared/ffn7.rs
  (cd work && cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule "../$name/schedule.json") > "$name/compile.log" 2>&1
  code=$?
  echo "$name exit=$code"
  if [ "$code" != 0 ]; then tail -20 "$name/compile.log"; fi
done
''')
print('\n'.join(variants))
