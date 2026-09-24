from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
base=(baseline/'src/device/sliding/qkv_head_local.rs').read_text()
opsbase=(baseline/'src/ops.rs').read_text()
def save(name,s,ops=opsbase):
 p=root/name;p.mkdir(exist_ok=True)
 shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
 for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
 (p/'src/device/sliding/qkv_head_local.rs').write_text(s)
 (p/'src/ops.rs').write_text(ops)

# Share the one calibrated scalar VRF across both quantization and projection.
a=base.index('    let scale: DmTensor',base.index('pub(crate) fn normalize_native_input'))
b=base.index('    let scaled: DmTensor',a)
block=base[a:b].replace('let scale: DmTensor','let scale: VrfTensor',1)
c=block.index('        .vector_final()')
block=block[:c]+'''        .vector_final()
        .to_vrf(&mut device.sub);
    let quant_scale: VrfTensor<f32, Chip, QkvDualQueryCluster, Shards, m![1 # 8]> =
        unsafe { scale.clone().reshape() };
'''
s=base[:a]+block+base[b:]
a=s.index('    let scale: DmTensor',s.index('    let trf: NativeTrf'))
b=s.index('    (trf, scale_vrf)',a)
s=s[:a]+'''    let scale_vrf: NativeScale = unsafe { scale.reshape() };
'''+s[b:]
save('p07-quant-scale-vrf',s)

# Keep the all-gather's H wire order in TRF and let contraction address that mapping.
s=base.replace('m![Dummy2], m![H]>;','m![Dummy2], m![H / 8 % 15, H / 120, H % 8]>;',1)
s=s.replace('type NativeSlices = m![Dummy256];','type NativeSlices = m![Dummy256 / 32, Dummy256 % 32];')
a=s.index('    let shared: DmTensor<f8e4m3',s.index('// Broadcast aligned'))
b=s.index('    let scale: DmTensor',a)
block='''    let trf: TrfTensor<f8e4m3, Chip, QkvDualQueryCluster, Copies, m![Dummy2], m![H / 8 % 15, H / 120, H % 8]> = device.main
        .begin(parts.view())
        .fetch::<m![Dummy2, H / 8 % 15], m![H % 8]>()
        .switch::<Copies, m![Dummy2, H / 8 % 15, H / 120]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Dummy2, H / 8 % 15, H / 120], m![H % 8 # 32]>()
        .to_trf();
    let trf: NativeTrf = trf;
'''
save('p08-switch-trf',s[:a]+block+s[b:])

# Exact BF16 projection scale is applied while values remain partitioned across rows.
# Broadcast the rounded rows for output while Sub reduces exactly those BF16 rows.
# The sum association changes from 256 serial values to 64 groups of four, but no
# projection/input weight or rounding boundary is dropped.
start=base.index('fn project_one_kv_matrix(')
end=base.index('\n}\n',start)+3
f=base[start:end]
f=f.replace('fn project_one_kv_matrix(', 'fn project_value_normalized(',1)
a=f.index('    // Each 64-slice group')
f=f[:a]+'''    let scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let scale: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4]> = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4 # 8]> = device.sub
        .begin(scale.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ds % 4 # 8]>()
        .to_vrf();
    let scaled: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![Ds % 4]> = device.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    let partial: DmTensor<f32, Chip, HeadCluster, m![Ns % 4, Ds / 4], m![1 # 8]> = device.sub
        .begin(scaled.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let mean: DmTensor<f32, Chip, HeadCluster, m![Ns % 4, Dummy256 % 64], m![1 # 8]> = device.sub
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns % 4, Dummy256 % 64], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let mean: DmTensor<f32, Chip, HeadCluster, KvHeads, m![1 # 8]> = unsafe { mean.reshape() };
    let heads: DmTensor<bf16, Chip, HeadCluster, m![Ns % 4, Dummy256 % 64], m![Ds]> = device.main
        .begin(scaled.view())
        .fetch::<m![1], m![Ds % 4]>()
        .switch::<m![Ns % 4, Dummy256 % 64], m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
    let heads: KvHeadTensor = unsafe { heads.reshape() };
    let rms_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![1 # 8]> = device.main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    device.main
        .begin(heads.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_div(&rms_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()
}
'''
s=base+'\n'+f
s=s.replace('let v = project_one_kv_matrix(device, x_trf, x_scale, v_weight, v_weight_scale);','let v = project_value_normalized(device, x_trf, x_scale, v_weight, v_weight_scale);')
ops=opsbase.replace('    let v = sliding::qkv_head_local::normalize_value(device, &v);\n','')
save('p09-parallel-v-exact',s,ops)

# A010's actual dependency split, but retain weighted input norm + dual FP8.
# Statistics use FP32 scaled values before the BF16 output rounding. This explicit
# numerical delta is an optional candidate, never treated as semantics-identical.
f=base[start:end].replace('fn project_one_kv_matrix(', 'fn project_value_normalized(',1)
a=f.index('    // Each 64-slice group')
f=f[:a]+'''    let scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let scale: KvHeadTensor = scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![Ds]> = device.sub
        .begin(scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    let mean: DmTensor<f32, Chip, HeadCluster, KvHeads, m![1 # 8]> = device.sub
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<KvHeads, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(const { Ds::SIZE as f32 })
        .vector_widen_pad::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, EPS)
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let heads: KvHeadTensor = device.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 4]>()
        .fetch_cast::<f32>()
        .switch::<KvHeads, m![Ds / 4]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 1 })
        .collect::<m![Ds / 4], m![Ds % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_pad::<m![Ds % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 4 # 8 # 16]>()
        .commit_trim::<m![Ds % 4]>()
        .commit();
'''
p09=(root/'p09-parallel-v-exact/src/device/sliding/qkv_head_local.rs').read_text()
tail=p09[p09.index('    let rms_vrf:',p09.index('fn project_value_normalized')):]
f+=tail
s=base+'\n'+f
s=s.replace('let v = project_one_kv_matrix(device, x_trf, x_scale, v_weight, v_weight_scale);','let v = project_value_normalized(device, x_trf, x_scale, v_weight, v_weight_scale);')
save('p10-a010-parallel-v',s,ops)
