from pathlib import Path
import hashlib, json
root=Path('research/extreme-opt-20260923/packages')
base=root/'P005-K1-081-K2K3-P005-final'
cand=root/'P020-K1-081-K2K3-P005-K3-interleaved'
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def files(p): return {str(x.relative_to(p)):h(x) for x in (p/'src').rglob('*') if x.is_file()}
a,b=files(base),files(cand)
print('source file set identical:', set(a)==set(b))
print('source files differing:', [(k,a.get(k),b.get(k)) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)])
for p in ['src/device/sliding/qkv_head_local.rs','src/device/sliding/output31.rs','src/device/shared/ffn7.rs']:
 print(p, h(base/p), h(cand/p))
for p in ['fixtures.safetensors','remote_entrypoint.sh','Cargo.toml','Cargo.lock','rust-toolchain.toml']:
 print(p, 'identical=',h(base/p)==h(cand/p),'sha256=',h(cand/p))
m=json.loads((cand/'manifest.json').read_text())
print('binary',m['files']['test_runtime'])
print('fresh artifact',not m['artifact'].get('fresh',True),'manifest match',m['artifact']['manifest_path']==str(cand.resolve()/'Cargo.toml'))
