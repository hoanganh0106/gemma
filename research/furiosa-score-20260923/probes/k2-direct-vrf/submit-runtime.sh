#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/furiosa-score-20260923/probes/k2-direct-vrf
python3 - <<'PY'
from pathlib import Path
import hashlib
import json

manifest = json.loads(Path('runtime-manifest.json').read_text())
for name in ('remote_entrypoint.sh', 'test_runtime', 'fixtures.safetensors'):
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != manifest['hashes'][name]:
        raise RuntimeError(f'Changed staged file: {name}')
wrapper = Path('remote_entrypoint.sh').read_bytes()
assert b'\r' not in wrapper, 'Wrapper has CRLF'
assert b'FURIOSA_OPT_PROFILE=info' in wrapper
assert b'taskset' not in wrapper and b'numactl' not in wrapper
assert not Path('arena-submit.json').exists(), 'Submission already attempted; inspect existing evidence, do not duplicate'
PY
furiosa-arena submit --name research-k2-direct-vrf-sdk081-three-seeds --entrypoint remote_entrypoint.sh --timeout 70 remote_entrypoint.sh test_runtime fixtures.safetensors > arena-submit.json
cat arena-submit.json
