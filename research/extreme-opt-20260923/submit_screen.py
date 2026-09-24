"""Submit one immutable official-harness Arena package, with no implicit retry."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('package')
parser.add_argument('label')
parser.add_argument('--diagnostic', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
package = root / 'packages' / args.package
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
manifest = json.loads((package / 'manifest.json').read_text())
assert manifest.get('diagnostic') is args.diagnostic
for name, digest in manifest['source_hashes'].items():
    assert sha(package / name) == digest, name
for name, digest in manifest['files'].items():
    assert sha(package / name) == digest, name
same_files = ['fixtures.safetensors'] if args.diagnostic else ['src/bin/test_kernels.rs', 'fixtures.safetensors', 'remote_entrypoint.sh']
for name in same_files:
    assert sha(package / name) == sha(parent / name), name
if args.diagnostic:
    assert args.label.startswith('diagnostic-')
    for name in manifest['source_hashes']:
        if name != 'src/bin/test_kernels.rs':
            assert sha(package / name) == sha(parent / name), name
    assert 'DIAGNOSTIC_ONLY' in (package / 'src/bin/test_kernels.rs').read_text()
    assert (package / 'remote_entrypoint.sh').read_text().replace('PROFILE=trace', 'PROFILE=info') == (parent / 'remote_entrypoint.sh').read_text()
assert b'taskset' not in (package / 'remote_entrypoint.sh').read_bytes()
record = package / ('submission-' + args.label + '.json')
assert not record.exists(), 'Do not duplicate an existing submission label'
entry = {'package': package.name, 'label': args.label, 'diagnostic': args.diagnostic, 'binary_sha256': sha(package / 'test_runtime'),
         'fixture_sha256': sha(package / 'fixtures.safetensors'), 'state': 'DISPATCHING'}
record.write_text(json.dumps(entry, indent=2))
command = ['furiosa-arena', 'submit', '--name', args.label, '--entrypoint', 'remote_entrypoint.sh', '--timeout', '70',
           'remote_entrypoint.sh', 'test_runtime', 'fixtures.safetensors']
result = subprocess.run(command, cwd=package, capture_output=True, text=True)
entry.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
match = re.search(r'submitted job (\d+)', result.stdout)
entry['state'] = 'SUBMITTED' if match else 'DISPATCH_UNCERTAIN'
if match:
    entry['job_id'] = int(match.group(1))
record.write_text(json.dumps(entry, indent=2))
print(json.dumps(entry))
assert result.returncode == 0 and match, 'Inspect exact dispatch record before any retry'
