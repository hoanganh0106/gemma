"""Port the exact full-table live-head RoPE path onto the best K1 tuple."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P025-k1-live-head-rope'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/device/sliding/qkv69382.rs'
s = p.read_text()
donor = (root / 'packages/P005-fixed-rings/src/device/sliding/qkv_head_local.rs').read_text()
start = 'fn rope_one<S: M>('
end = 'pub(crate) fn apply_rope('
rope = donor[donor.index(start):donor.index(end)]
s = s[:s.index(start)] + rope + s[s.index('fn rope_one_pair_fixed<S: M>('):]
end = 'fn scale_head<S: M>('
apply = donor[donor.index('pub(crate) fn apply_rope('):donor.index(end)]
s = s[:s.index('pub(crate) fn apply_rope(')] + apply + s[s.index(end):]
p.write_text(s)
hashes = {str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dest / 'src').rglob('*')) if f.is_file()}
(dest / 'frozen-source.json').write_text(json.dumps(hashes, indent=2))
(dest / 'HYPOTHESIS.md').write_text('Use full runtime cos/sin table rows with true broadcast to initialized live head slices (2 clusters x 8 query-head slices), then preserve the official half-swap arithmetic. Remove packed-half HBM round trip and redundant sine product DM reload. Keep native weighted input norm and all Q/K/V projection code byte-identical to parent. This also removes the parent assumption of repeated/signed table halves. Donor RoPE was public-harness tested in P005-fixed-rings; this new integrated tuple still needs its own schedule, hashes and Arena checks.\n')
print(dest)
