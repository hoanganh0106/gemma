from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
s=(root/'p14-rope-head-broadcast/src/device/sliding/qkv_head_local.rs').read_text()
old=(root.parents[2]/'experiments/auto_k1_current_38723/src/device/sliding/qkv69382.rs').read_text()
a=old.index('fn project_one_kv_matrix_scaled(')
b=old.index('    // Apply the row scale',a)
body=old[a:b]
body=body.replace('fn project_one_kv_matrix_scaled(', 'fn project_one_kv_matrix(',1)
body=body.replace('ctx: &mut Context,','ctx: &mut Device,',1)
body=body.replace('    x_trf: &NativeTrf,','    x_trf: &NativeTrf,\n    x_scale: &NativeScale,',1)
body=body.replace(') -> (KvHeadTensor, KvMeanSquare) {',') -> KvHeadTensor {',1)
body=body.replace('KvStripe, m![1], m![H]>','KvStripe, m![Dummy2], m![H]>',1)
anchor='        unsafe { std::mem::transmute(x_trf) };'
body=body.replace(anchor,anchor+'''\n    // The input and calibration scale are identical at every physical slice.
    let x_scale: &VrfTensor<f32, Chip, HeadCluster, KvStripe, m![1 # 8]> =
        unsafe { std::mem::transmute(x_scale) };''',1)
body=body.replace('.contract_lane::<m![Ds / 8 % 4], m![1 # 8]>(LaneMode::Interleaved)',
                  '.contract_lane::<m![Ds / 8 % 4, Dummy2], m![1 # 8]>(LaneMode::Sequential)',1)
anchor='        .vector_narrow_trim::<m![1 # 4]>()'
body=body.replace(anchor,anchor+'''\n        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), x_scale)
        .vector_intra_slice_reduce::<Dummy2, m![Ds / 8 % 4], m![1 # 4]>(IntraSliceReduceOpF32::Add)''',1)
a=s.index('fn project_one_kv_matrix(')
b=s.index('    // Each 64-slice group',a)
s=s[:a]+body.replace('ctx.', 'device.').replace('ctx: &mut Device','device: &mut Device')+s[b:]
p=root/'p18-rope-kv-stripe';p.mkdir(exist_ok=True)
shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
(p/'src/device/sliding/qkv_head_local.rs').write_text(s)
