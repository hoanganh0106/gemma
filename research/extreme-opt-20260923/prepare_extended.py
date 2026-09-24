"""Create a separate diagnostic package; never modify an official candidate."""
from pathlib import Path
import inspect
import shutil
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
source, dest = (Path(arg).resolve() for arg in sys.argv[1:3])
assert not dest.exists(), f'Refusing to overwrite {dest}'
dest.mkdir(parents=True)
shutil.copytree(source / 'src', dest / 'src')
for name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'):
    shutil.copy2(source / name, dest / name)
shutil.copy2(root / 'research/furiosa-score-20260923/remote_entrypoint.sh', dest / 'remote_entrypoint.sh')

def change(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

p = dest / 'src/bin/test_kernels.rs'
h = p.read_text()
h = change(h, 'let x: HbmTensor<bf16, Chip, m![H]> = exact_rmsnorm_input(device, &s).await;',
           'let x: HbmTensor<bf16, Chip, m![H]> = s.bf16(device, "x_generic", [(-1.0,1.0),(-0.001,0.001),(-3.0,3.0)][run]).await;')
h = change(h, 'let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.bf16(device, "x", ACTIVATION).await;',
           'let x: HbmTensor<bf16, Chip, m![Ns, Gs, Ds]> = s.bf16(device, "x", [(-0.001,0.001),(-1.0,1.0),(-3.0,3.0)][run]).await;')
start = h.index('async fn decoder_feedforward(')
tail = h[start:]
tail = change(tail, 's.bf16(device, "residual", UNIT).await', 's.bf16(device, "residual", ACTIVATION).await')
tail = change(tail, '&[LAYER_SCALAR; 8]', '&[[0.125f32, 0.375, 0.875][run]; 8]')
h = h[:start] + tail
start = h.index('fn global_scale(')
end = h.index('\n}', start) + 2
h = h[:start] + '''fn global_scale(run: usize, name: &str) -> f32 {
    let index = match name { "up" => 0, "gate" => 1, "down" => 2, _ => panic!("unknown matrix") };
    [2048.0, 8192.0, 16384.0][(run + index) % 3]
}''' + h[end:]
h = change(h, 'let mut within = 0usize;', 'let mut within = 0usize;\n    let mut max_tolerance_ratio = 0.0f32;')
h = change(h, 'let diff = (want - got).abs();', 'let diff = (want - got).abs();\n        max_tolerance_ratio = max_tolerance_ratio.max(diff / (atol + rtol * want.abs()));')
h = change(h, '    let count = expected.len();', '    println!("[{label:34}] diagnostic max tolerance ratio={max_tolerance_ratio:.6}");\n    let count = expected.len();')
p.write_text(h)

sys.path.insert(0, str(root / 'scripts'))
import generate_references as ref
import numpy as np
import torch
from safetensors.torch import save_file
torch.set_grad_enabled(False)

def rewrite(fn, replacements):
    code = inspect.getsource(fn)
    for old, new in replacements: code = change(code, old, new)
    exec(code, ref.__dict__)

rewrite(ref.gen_sliding_project_qkv, [('signs = s.signs("x_signs", (H,))\n    x = s.derived("x_exact", (signs.float() / input_rms_weight.float()).to(torch.bfloat16))', 'x = s.bf16("x_generic", (H,), [(-1.0,1.0),(-0.001,0.001),(-3.0,3.0)][run])')])
rewrite(ref.gen_sliding_attention_output, [('s.bf16("x", (NS, GS, DS), ACTIVATION)', 's.bf16("x", (NS, GS, DS), [(-0.001,0.001),(-1.0,1.0),(-3.0,3.0)][run])')])
rewrite(ref.gen_decoder_feedforward, [('s.bf16("residual", (H,), UNIT)', 's.bf16("residual", (H,), ACTIVATION)'), ('np.full(8, LAYER_SCALAR, dtype=np.float32)', 'np.full(8, [0.125,0.375,0.875][run], dtype=np.float32)')])
original = ref.prng.f32_uniform
def endpoint_scales(name, count, lo, hi, offset=0):
    if name.endswith('_global_scale'):
        run = int(name.split('.')[0][3:]); matrix = name.split('.')[-1].split('_')[0]
        return np.array([[2048.0,8192.0,16384.0][(run + ['up','gate','down'].index(matrix)) % 3]], dtype=np.float32)
    return original(name, count, lo, hi, offset)
ref.prng.f32_uniform = endpoint_scales
entries = {}
for run in range(3):
    for name in ('sliding_project_qkv','sliding_attention_output','decoder_feedforward'):
        outputs, checks = getattr(ref, 'gen_' + name)(run)
        prefix = f'run{run}.{name}.'
        for key,value in outputs.items(): entries[prefix+key] = value.detach().float().contiguous()
        for key,value in checks.items(): entries[prefix+'check.'+key] = torch.tensor([value], dtype=torch.int64)
        print('generated', run, name, flush=True)
save_file(entries, str(dest / 'fixtures.safetensors'))
print('EXTENDED_FIXTURE_READY', dest, flush=True)
