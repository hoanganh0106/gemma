"""Pair a frozen candidate with the same independently generated diagnostic data."""
import json
from pathlib import Path
import shutil
import sys
from campaign import sha, source_hashes

here = Path(__file__).resolve().parent
source, dest = (Path(s).resolve() for s in sys.argv[1:3])
reference = here / 'stress-extended-baseline'
assert not (dest / 'manifest.json').exists(), 'Refusing to modify a built package'
if not dest.exists():
    dest.mkdir(parents=True)
    shutil.copytree(source / 'src', dest / 'src')
    for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'remote_entrypoint.sh'):
        shutil.copy2(source / name, dest / name)
    shutil.copy2(reference / 'src/bin/test_kernels.rs', dest / 'src/bin/test_kernels.rs')
    shutil.copy2(reference / 'fixtures.safetensors', dest / 'fixtures.safetensors')
before, after = source_hashes(source), source_hashes(dest)
changed = [name.replace('\\', '/') for name in before if before[name] != after[name]]
assert changed == ['src/bin/test_kernels.rs'], changed
assert sha(dest / 'src/bin/test_kernels.rs') == sha(reference / 'src/bin/test_kernels.rs')
assert sha(dest / 'fixtures.safetensors') == sha(reference / 'fixtures.safetensors')
(dest / 'diagnostic-origin.json').write_text(json.dumps({
    'candidate': str(source), 'diagnostic_reference': str(reference),
    'changed_source_files': changed,
    'candidate_source_hashes': before,
    'fixture_sha256': sha(dest / 'fixtures.safetensors'),
}, indent=2))
print(dest)
