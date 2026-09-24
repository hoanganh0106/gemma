"""Separate diagnostic harness: generic K1 activation, unchanged device kernels."""
from pathlib import Path
import inspect
import shutil
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
dest = here / 'stress-baseline'
if not dest.exists():
    shutil.copytree(here / 'baseline', dest)
shutil.copy2(root / 'research/furiosa-score-20260923/remote_entrypoint.sh', dest / 'remote_entrypoint.sh')
harness = (here / 'baseline/src/bin/test_kernels.rs').read_text()
old = 'let x: HbmTensor<bf16, Chip, m![H]> = exact_rmsnorm_input(device, &s).await;'
new = 'let span = [(-1.0, 1.0), (-0.001, 0.001), (-3.0, 3.0)][run];\n    let x: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "x_generic", span).await;'
assert harness.count(old) == 1
harness = harness.replace(old, new)
harness = harness.replace('let mut within = 0usize;', 'let mut within = 0usize;\n    let mut max_tolerance_ratio = 0.0f32;')
harness = harness.replace('let diff = (want - got).abs();', 'let diff = (want - got).abs();\n        max_tolerance_ratio = max_tolerance_ratio.max(diff / (atol + rtol * want.abs()));')
harness = harness.replace('    let count = expected.len();', '    println!("[{label:34}] diagnostic max tolerance ratio={max_tolerance_ratio:.6}");\n    let count = expected.len();')
(dest / 'src/bin/test_kernels.rs').write_text(harness)

sys.path.insert(0, str(root / 'scripts'))
import generate_references as ref
import torch
from safetensors.torch import load_file, save_file
torch.set_grad_enabled(False)
code = inspect.getsource(ref.gen_sliding_project_qkv)
old = 'signs = s.signs("x_signs", (H,))\n    x = s.derived("x_exact", (signs.float() / input_rms_weight.float()).to(torch.bfloat16))'
new = 'span = [(-1.0, 1.0), (-0.001, 0.001), (-3.0, 3.0)][run]\n    x = s.bf16("x_generic", (H,), span)'
assert old in code
exec(code.replace(old, new), ref.__dict__)
entries = load_file(str(root / 'research/furiosa-score-20260923/fixtures.safetensors'))
for run in range(3):
    prefix = f'run{run}.sliding_project_qkv.'
    entries = {k: v for k, v in entries.items() if not k.startswith(prefix)}
    outputs, checks = ref.gen_sliding_project_qkv(run)
    for key, value in outputs.items(): entries[prefix + key] = value.detach().float().contiguous()
    for key, value in checks.items(): entries[prefix + 'check.' + key] = torch.tensor([value], dtype=torch.int64)
    print('Generated generic weighted QKV case', run, flush=True)
save_file(entries, str(dest / 'fixtures.safetensors'))
print('Diagnostic fixture:', dest / 'fixtures.safetensors', flush=True)
