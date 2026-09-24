"""Probe contiguous projection rows while retaining the best K1 arithmetic."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P030-k1-contiguous-projection'
assert not (dest / 'HYPOTHESIS.md').exists(), dest
if dest.exists():
    assert (dest / 'src/device/sliding/qkv69382.rs').read_bytes() == (parent / 'src/device/sliding/qkv69382.rs').read_bytes()
dest.mkdir(exist_ok=True)
shutil.copytree(parent / 'src', dest / 'src', dirs_exist_ok=True)
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
relative = 'src/device/sliding/qkv69382.rs'
s = (parent / relative).read_text()
donor = (root / 'packages/P005-fixed-rings/src/device/sliding/qkv_head_local.rs').read_text()
q = donor[donor.index('pub(crate) fn project_query('):donor.index('fn project_one_kv_matrix(')]
kv = donor[donor.index('fn project_one_kv_matrix('):donor.index('pub(crate) fn project_key_value(')]
kv = kv[:kv.index('    let weight_scale: HbmTensorView')]
oldraw = s[s.index('fn project_one_kv_matrix_raw('):s.index('fn apply_projection_scale_kv(')]
kv += oldraw[oldraw.index('    // Each 64-slice group'):]
kv = kv.replace('fn project_one_kv_matrix(', 'fn project_one_kv_matrix_raw(')
kv = kv.replace('    weight_scale: &HbmTensor<bf16, Chip, m![Ps]>,\n', '')
kv = kv.replace(' = ctx\n', ' = device\n')
def single_term(text, axis):
    text = text.replace('    x_scale: &NativeScale,\n', '')
    text = text.replace(f'm![{axis}, Dummy2]', f'm![{axis}]')
    text = text.replace('        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale)\n', '')
    text = text.replace(f'        .vector_intra_slice_reduce::<Dummy2, m![{axis}], m![1 # 4]>(IntraSliceReduceOpF32::Add)\n', '')
    assert 'Dummy2]' not in text and 'x_scale' not in text
    return text
q = single_term(q, 'Qs % 8')
kv = single_term(kv, 'Ps % 4')
s = s[:s.index('pub(crate) fn project_query(')] + q + s[s.index('pub(crate) fn copy_query('):]
s = s[:s.index('fn project_one_kv_matrix_raw(')] + kv + s[s.index('fn apply_projection_scale_kv('):]
(dest / relative).write_text(s)
(dest / 'HYPOTHESIS.md').write_text('Use contiguous eight Q rows / four KV rows per slice instead of striped row ownership, and write contraction output directly as BF16 before head broadcast. Removes the FP32 DM output plus row InterTranspose stage. One-term FP8 input and all weighted normalization/scales remain unchanged from canonical parent. BF16 rounding moves across a data permutation only, never across arithmetic. Weight DMA can worsen despite fewer commands; compare full schedule and hardware. Parent RoPE unchanged.\n')
print(dest)
