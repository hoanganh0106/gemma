from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
base=root.parent/'baseline'
src=root/'p18-rope-kv-stripe'
s=(src/'src/device/sliding/qkv_head_local.rs').read_text()
a=s.index('fn project_one_kv_matrix(');b=s.index('\n}',a)+2
f=s[a:b].replace('    x_scale: &NativeScale,','    x_scale: &VrfTensor<f32, Chip, HeadCluster, m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2], m![1 # 8]>,',1)
f=f.replace('''    // The input and calibration scale are identical at every physical slice.
    let x_scale: &VrfTensor<f32, Chip, HeadCluster, KvStripe, m![1 # 8]> =
        unsafe { std::mem::transmute(x_scale) };
''','')
s=s[:a]+f+s[b:]
a=s.index('pub(crate) fn project_key_value(');b=s.index('\n}',a)+2
f=s[a:b].replace('    x_scale: &NativeScale,','    x_scale: NativeScale,',1)
f=f.replace('    let k = project_one_kv_matrix(','''    // Consuming the handle changes only its logical labels; all physical units
    // already hold the same globally calibrated scale. Use the SDK reshape primitive.
    let x_scale: VrfTensor<f32, Chip, HeadCluster, m![Ns % 4, Ds / 32, Ds % 4, Ds / 4 % 2], m![1 # 8]> =
        unsafe { x_scale.reshape() };
    let k = project_one_kv_matrix(''',1)
f=f.replace('(device, x_trf, x_scale,','(device, x_trf, &x_scale,')
s=s[:a]+f+s[b:]
ops=(base/'src/ops.rs').read_text().replace('device, &x, &x_scale, k_weight, v_weight, k_weight_scale, v_weight_scale,','device, &x, x_scale, k_weight, v_weight, k_weight_scale, v_weight_scale,',1)
p=root/'p18b-rope-kv-stripe';p.mkdir(exist_ok=True)
shutil.copytree(base/'src',p/'src',dirs_exist_ok=True)
for fn in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(base/fn,p/fn)
(p/'src/device/sliding/qkv_head_local.rs').write_text(s)
(p/'src/ops.rs').write_text(ops)
