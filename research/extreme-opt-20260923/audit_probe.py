"""Record the exact isolated source and static schedule being screened."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('package')
args = parser.parse_args()
root = Path(__file__).resolve().parent
package = root / 'packages' / args.package
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(package)): sha(p) for p in sorted((package / 'src').rglob('*')) if p.is_file()}
changed = {name: {'parent': sha(parent / name), 'candidate': digest} for name, digest in hashes.items() if sha(parent / name) != digest}
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh', 'src/bin/test_kernels.rs']:
    assert sha(package / name) == sha(parent / name), name
assert set(changed) == {'src/device/sliding/qkv69382.rs'}, changed
schedule = json.loads((package / 'schedule-k1.json').read_text())
instructions = schedule['instructions']
result = {'package': args.package, 'parent': parent.name, 'changed': changed,
          'makespan': max(i['lifetime']['end'] for i in instructions),
          'instruction_count': len(instructions),
          'schedule_sha256': sha(package / 'schedule-k1.json'),
          'fixture_sha256': sha(package / 'fixtures.safetensors'),
          'correctness': 'NOT_YET_RUN', 'arena': 'NOT_YET_RUN'}
(package / 'frozen-source.json').write_text(json.dumps(hashes, indent=2))
(package / 'static-screen.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
