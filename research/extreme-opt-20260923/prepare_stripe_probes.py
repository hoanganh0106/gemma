"""Two ownership-preserving row-stripe experiments with explicit inverse maps."""
from pathlib import Path
import shutil
import sys

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
mode = sys.argv[1]
names = {'kv': 'P032-k1-kv-stripe4', 'q': 'P033-k1-q-stripe8x2'}
dest = root / 'packages' / names[mode]
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/device/sliding/qkv69382.rs'
s = p.read_text()
if mode == 'kv':
    # Row index before switch: 16*A + 4*T + B.
    # Switch swaps the 4-way B slice axis and T time axis; output rows are canonical.
    start, end = s.index('fn project_one_kv_matrix_raw('), s.index('fn apply_projection_scale_kv(')
    part = s[start:end]
    part = part.replace('m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2]', 'm![Ns % 4, Ds / 16, Ds % 4]')
    part = part.replace('m![Ns % 4, Ds / 32, Ds / 8 % 4, Ds / 4 % 2]', 'm![Ns % 4, Ds / 16, Ds / 4 % 4]')
    part = part.replace('Ds / 8 % 4', 'Ds / 4 % 4').replace('slice0: 2', 'slice0: 1')
    part = part.replace('8-way row ring', '4-way row ring')
else:
    # Row index before switch: 128*A + 16*T + 8*C + B.
    # Switch swaps B and T with C retained at slice0=2, giving canonical groups of 8.
    start, end = s.index('pub(crate) fn project_query('), s.index('pub(crate) fn copy_query(')
    part = s[start:end]
    part = part.replace('m![Ns % 4, Gs, Ds / 64, Ds % 8]', 'm![Ns % 4, Gs, Ds / 128, Ds % 8, Ds / 8 % 2]')
    part = part.replace('m![Ns % 4, Gs, Ds / 64, Ds / 8 % 8]', 'm![Ns % 4, Gs, Ds / 128, Ds / 16 % 8, Ds / 8 % 2]')
    part = part.replace('Ds / 8 % 8', 'Ds / 16 % 8').replace('slice0: 1', 'slice0: 2', 1)
s = s[:start] + part + s[end:]
p.write_text(s)
(dest / 'HYPOTHESIS.md').write_text('Preserve all runtime input, contraction arithmetic, BF16 boundary and head ownership. Change only the weight-row stripe and its matching inverse InterTranspose. No row is dropped or duplicated in output. P030 demonstrated hardware sensitivity to row ownership despite a near-neutral static model; this probe tests a narrower DMA distribution change. Mode: ' + mode + '\n')
print(dest)
