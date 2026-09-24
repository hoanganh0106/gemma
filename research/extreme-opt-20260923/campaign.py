"""Freeze/build packages and parse full Arena logs for this bounded campaign."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / 'research/furiosa-score-20260923'
FIXTURE_HASH = '338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_hashes(path):
    return {str(p.relative_to(path)): sha(p) for p in sorted((path / 'src').rglob('*')) if p.is_file()}

def freeze(args):
    dest = HERE / 'packages' / args.name
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite frozen package: {dest}')
    shutil.copytree(HERE / 'baseline', dest)
    for field, target in [('k1', 'src/device/sliding/qkv_head_local.rs'), ('k2', 'src/device/sliding/output31.rs'), ('k3', 'src/device/shared/ffn7.rs'), ('ops', 'src/ops.rs')]:
        source = getattr(args, field)
        if source:
            shutil.copy2(source, dest / target)
    shutil.copy2(REFERENCE / 'fixtures.safetensors', dest / 'fixtures.safetensors')
    shutil.copy2(REFERENCE / 'remote_entrypoint.sh', dest / 'remote_entrypoint.sh')
    assert sha(dest / 'fixtures.safetensors') == FIXTURE_HASH
    (dest / 'frozen-source.json').write_text(json.dumps(source_hashes(dest), indent=2))
    print(dest)

def build(args):
    import os
    import fcntl
    dest = Path(args.package).resolve()
    if (dest / 'SUPERSEDED.txt').exists():
        print(json.dumps({'name': dest.name, 'not_built': True, 'reason': (dest / 'SUPERSEDED.txt').read_text()}))
        return
    before = source_hashes(dest)
    target_dir = Path(args.target_dir).resolve() if args.target_dir else ROOT / 'target'
    env = dict(os.environ, CARGO_INCREMENTAL='0', CARGO_NET_OFFLINE='true', CARGO_TARGET_DIR=str(target_dir))
    # Cargo can reuse this crate across copied manifests with preserved mtimes.
    # Keep dependencies, force this package to rebuild, and retain the lock until
    # the executable has been copied out of the shared target directory.
    # Every release build shares this lock, including default and explicit forms
    # of the same target path. Static probes use a separate Linux target/lock.
    lock_path = '/tmp/furiosa-extreme-full-build.lock'
    lock = open(lock_path, 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    identity = {
        'source': {k: v for k, v in before.items() if k != 'src/bin/test_kernels.rs'},
        'config': {name: sha(dest / name) for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']},
        'flags': {name: env.get(name) for name in ['RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_PROFILE_RELEASE_OPT_LEVEL', 'CARGO_PROFILE_RELEASE_CODEGEN_UNITS']},
    }
    state_path = target_dir / '.extreme-campaign-verified-library.json'
    previous = json.loads(state_path.read_text()) if state_path.exists() else {}
    reuse = previous.get('identity') == identity and bool(previous.get('library_files')) and all(
        Path(path).is_file() and sha(path) == digest for path, digest in previous.get('library_files', {}).items())
    if reuse:
        # Only the test binary differs. Reuse is allowed solely when both all
        # library source/config bytes and the actual library artifacts match a
        # completed, verified build held under this same target lock.
        os.utime(dest / 'src/bin/test_kernels.rs', None)
    else:
        subprocess.run(['cargo', 'clean', '--release', '-p', 'furiosa-opt-gemma4'], cwd=dest, env=env, check=True)
    command = ['cargo', 'furiosa-opt', 'build', '--release', '--locked', '--bin', 'test_kernels', '--message-format=json']
    with (dest / 'build-artifacts.jsonl').open('w') as out, (dest / 'build.log').open('w') as err:
        subprocess.run(command, cwd=dest, env=env, stdout=out, stderr=err, check=True)
    artifacts = []
    libraries = []
    for line in (dest / 'build-artifacts.jsonl').read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == 'test_kernels' and item.get('executable'):
            artifacts.append(item)
        if item.get('reason') == 'compiler-artifact' and item.get('target', {}).get('name') == 'furiosa_opt_gemma4':
            libraries.append(item)
    assert len(artifacts) == 1, artifacts
    assert len(libraries) == 1 and (not libraries[0].get('fresh') or reuse), 'Refusing a possibly stale crate library'
    assert not artifacts[0].get('fresh'), 'Refusing a possibly stale shared-target binary'
    assert Path(artifacts[0]['manifest_path']).resolve() == dest / 'Cargo.toml'
    assert before == source_hashes(dest), 'Source changed during build'
    shutil.copy2(artifacts[0]['executable'], dest / 'test_runtime')
    manifest = {'name': dest.name, 'diagnostic': args.diagnostic, 'command': command, 'target_dir': str(target_dir), 'lock_path': lock_path, 'forced_crate_rebuild': not reuse, 'verified_library_reused_from': previous.get('package') if reuse else None, 'artifact': artifacts[0], 'library_artifact': libraries[0], 'source_hashes': before,
                'files': {name: sha(dest / name) for name in ['test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh', 'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml']}}
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    state_path.write_text(json.dumps({'identity': identity, 'package': str(dest),
        'library_files': {path: sha(path) for path in libraries[0]['filenames']}}, indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    print(json.dumps({'name': dest.name, 'binary_sha256': manifest['files']['test_runtime'], 'fixture_sha256': manifest['files']['fixtures.safetensors']}))

def parse(args):
    path = Path(args.log)
    raw = path.read_text()
    matches = re.findall(r'median cycles=(\d+) \(of 3 runs: \[([^]]+)\]\)', raw)
    samples = [[int(v.strip()) for v in values.split(',')] for _, values in matches]
    medians = [int(value) for value, _ in matches]
    passed = raw.count('-> PASS')
    valid = passed == 15 and 'all 3 tests passed' in raw and len(medians) == 3
    assert all(len(s) == 3 and sorted(s)[1] == m for m, s in zip(medians, samples))
    diagnostic = 'diagnostic max tolerance ratio=' in raw or 'DIAGNOSTIC_ONLY' in raw
    out = {'log': str(path), 'valid_full_public_harness': valid and not diagnostic, 'all_checks_pass': valid, 'diagnostic': diagnostic, 'pass_count': passed, 'medians': medians, 'samples': samples}
    if valid and not diagnostic:
        out['illustrative_score_snapshot_baseline'] = math.prod(b / s for b, s in zip([250514,404633,3703473], medians)) ** (1 / 3)
    path.with_suffix('.summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('freeze'); p.add_argument('name')
    for field in ['k1', 'k2', 'k3', 'ops']: p.add_argument('--' + field)
    p.set_defaults(action=freeze)
    p = commands.add_parser('build'); p.add_argument('package'); p.add_argument('--diagnostic', action='store_true'); p.add_argument('--target-dir'); p.set_defaults(action=build)
    p = commands.add_parser('parse'); p.add_argument('log'); p.set_defaults(action=parse)
    args = parser.parse_args(); args.action(args)
