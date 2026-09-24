from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
base=(baseline/'src/device/sliding/qkv_head_local.rs').read_text()
def fuse(s):
 a=s.index('    // Each 32-slice group')
 b=s.index('\n}',a)
 s=s[:a]+'''    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Gs, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_scale: QueryHeadTensor = weight_scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, QueryHeads, m![Ds]> = device.sub
        .begin(weight_scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    // Fuse head assembly and channel scaling; keep the existing BF16 boundary.
    device.main
        .begin(rows.view())
        .fetch::<m![1], m![Ds % 8]>()
        .fetch_cast::<f32>()
        .switch::<QueryHeads, m![Ds / 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 1 })
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Ds / 4], m![Ds % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale_vrf)
        .vector_widen_concat::<m![Ds / 8], m![Ds % 8]>()
        .vector_final()
        .cast::<bf16, m![Ds % 8 # 16]>()
        .commit_trim::<m![Ds % 8]>()
        .commit()'''+s[b:]
 a=s.index('    // Each 64-slice group')
 b=s.index('\n}',a)
 s=s[:a]+'''    let weight_scale: HbmTensorView<'_, bf16, Chip, m![Ns, Ds]> =
        unsafe { weight_scale.view().reshape() };
    let weight_scale: KvHeadTensor = weight_scale.to_dm(&mut device.tdma);
    let scale_vrf: VrfTensor<f32, Chip, HeadCluster, KvHeads, m![Ds]> = device.sub
        .begin(weight_scale.view())
        .fetch::<m![Ds / 16], m![Ds % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![Ds / 8], m![Ds % 8]>()
        .to_vrf();
    // Every head replica receives the same scaled row values as the unfused path.
    device.main
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
        .commit()'''+s[b:]
 return s
def save(name,s):
 p=root/name;p.mkdir(exist_ok=True)
 shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
 for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
 (p/'src/device/sliding/qkv_head_local.rs').write_text(s)
save('p11-fused-gather-scale',fuse(base))
save('p12-fused-gather-rms-vrf',fuse((root/'p01-rms-vrf/src/device/sliding/qkv_head_local.rs').read_text()))
