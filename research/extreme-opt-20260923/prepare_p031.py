"""Correct every extent in the compact K3 partial-buffer hypothesis."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P031-k3-compact-partial-consistent'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/device/shared/ffn7.rs'
s = p.read_text()
for before, after in [('Pw = 64', 'Pw = 32'), ('m![H % 30 # 64]', 'm![H % 30 # 32]'), ('Pw = 30 # 64', 'Pw = 30 # 32')]:
    assert s.count(before) == 1, (before, s.count(before))
    s = s.replace(before, after)
p.write_text(s)
(dest / 'frozen-source.json').write_text(json.dumps({str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dest / 'src').rglob('*')) if f.is_file()}, indent=2))
(dest / 'HYPOTHESIS.md').write_text('P027 changed the Pw alias but left the actual producer at H%30#64; its reshape rejection does not test a consistently compact buffer. P031 changes producer extent, Pw and the consumer stride together. Preserve all 30 live FP32 values per row and drop only padding. Expected HBM traffic halves, but 128-byte partial writes may regress transaction costs. Compiler legality and full schedule decide whether hardware screening is justified.\n')
print(dest)
