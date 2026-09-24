"""Isolate a direct head-RMS VRF probe from the measured qkv69382 parent."""
from pathlib import Path
import hashlib
import json
import re
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P024-k1-head-rms-vrf'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/device/sliding/qkv69382.rs'
s = p.read_text()
pattern = re.compile(r'    let rms: DmTensor<f32, Chip, HeadCluster, (S|KvHeads), m!\[1 # 8\]> = ctx(.*?)        \.vector_final\(\)\n        \.commit_trim::<m!\[1 # 8\]>\(\)\n        \.commit\(\);\n(?:    let rms_vrf: VrfTensor<f32, Chip, HeadCluster, (?:S|KvHeads), m!\[1 # 8\]> = ctx\n        \.sub|    ctx\.sub)\n        \.begin\(rms\.view\(\)\)\n        \.fetch::<m!\[1\], m!\[1 # 8\]>\(\)\n        \.collect::<m!\[1\], m!\[1 # 8\]>\(\)\n        \.to_vrf\(\)(;?)', re.S)
def replace(m):
    if m.group(3):
        return f'    let rms_vrf: VrfTensor<f32, Chip, HeadCluster, {m.group(1)}, m![1 # 8]> = ctx{m.group(2)}        .vector_final()\n        .to_vrf(&mut ctx.sub);'
    return f'    ctx{m.group(2)}        .vector_final()\n        .to_vrf(&mut ctx.sub)'
s, n = pattern.subn(replace, s)
assert n == 3, n
p.write_text(s)
hashes = {str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dest / 'src').rglob('*')) if f.is_file()}
(dest / 'frozen-source.json').write_text(json.dumps(hashes, indent=2))
(dest / 'HYPOTHESIS.md').write_text('Replace three FP32 head-RMS DM commits plus Sub reloads with Main-to-VRF delivery. Preserve arithmetic, sqrt, EPS, input_rms_weight, all projection and RoPE inputs, BF16 boundaries and physical mappings. Parent: P005-K1-081-K2K3-P005-final. Static parent: 42556 cycles / 189 instructions. Official harness unchanged.\n')
print(dest)
