"""Combine separately screened head RMS and full-table RoPE transformations."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P028-k1-rope-rms-vrf'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
relative = 'src/device/sliding/qkv69382.rs'
rms = (root / 'packages/P024-k1-head-rms-vrf' / relative).read_text()
rope = (root / 'packages/P025-k1-live-head-rope' / relative).read_text()
split = 'fn rope_one<S: M>('
(dest / relative).write_text(rms[:rms.index(split)] + rope[rope.index(split):])
(dest / 'HYPOTHESIS.md').write_text('Combine P024 direct head-RMS Main-to-VRF with P025 full-table live-head RoPE. Shared-resource interactions require a new schedule; individual gains are not additive assumptions. All input normalization and projection functions remain unchanged from the canonical parent, except the exact FP32 head-RMS storage path.\n')
print(dest)
