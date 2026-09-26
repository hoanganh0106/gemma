#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/furiosa-score-20260923/probes/k2-direct-vrf
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target
cargo furiosa-opt build --release --locked --bin test_kernels --message-format=json > build-artifacts.jsonl 2> public-build.log
python3 - <<'PY'
import hashlib
import json
from pathlib import Path
import shutil

root = Path.cwd()
artifacts = []
for line in (root / 'build-artifacts.jsonl').read_text().splitlines():
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        continue
    if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == 'test_kernels' and item.get('executable'):
        artifacts.append(item)
if len(artifacts) != 1:
    raise RuntimeError(f'Expected exactly one test_kernels executable, found {len(artifacts)}')
selected = artifacts[0]
shutil.copy2(selected['executable'], root / 'test_runtime')
fixture = root.parents[1] / 'fixtures.safetensors'
fixture_sha = hashlib.sha256(fixture.read_bytes()).hexdigest()
expected_fixture_sha = '338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359'
if fixture_sha != expected_fixture_sha:
    raise RuntimeError(f'Unexpected fixture hash: {fixture_sha}')
shutil.copy2(fixture, root / 'fixtures.safetensors')
shutil.copy2(root.parents[1] / 'remote_entrypoint.sh', root / 'remote_entrypoint.sh')
manifest = {
    'candidate': 'k2-direct-vrf-sdk081',
    'reference_arena_job': 78017,
    'artifact': selected,
    'hashes': {},
    'source_hashes': {},
}
for name in ('test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh', 'k2.json', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'):
    path = root / name
    manifest['hashes'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
for path in sorted((root / 'src').rglob('*')):
    if path.is_file():
        manifest['source_hashes'][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'runtime-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
for name, digest in manifest['hashes'].items():
    print(f'{digest}  {name}')
PY
