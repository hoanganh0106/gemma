from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
def save(name,s,ops=None):
 p=root/name;p.mkdir(exist_ok=True)
 shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
 for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
 (p/'src/device/sliding/qkv_head_local.rs').write_text(s)
 if ops:(p/'src/ops.rs').write_text(ops)

s=(root/'p14-rope-head-broadcast/src/device/sliding/qkv_head_local.rs').read_text()
for op in ('Add','Max'):
 old=f'''        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Copies, m![1]>(InterSliceReduceOpF32::{op})
        .vector_intra_slice_tag(TagMode::Zero)'''
 new=f'''        .fetch::<m![1], m![1 # 8]>()
        .switch::<Copies, m![H / 120]>(SwitchConfig::CustomBroadcast {{ ring_size: 32 }})
        .collect::<m![H / 120], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::{op})
        .vector_widen_pad::<m![1 # 8]>()'''
 assert s.count(old)==1
 s=s.replace(old,new)
save('p16-rope-input-ring32',s)

s=(root/'p09-parallel-v-exact/src/device/sliding/qkv_head_local.rs').read_text()
old='''        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Ns % 4, Dummy256 % 64], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)'''
new='''        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![Ns % 4, Dummy256 % 64], m![Ds / 4]>(SwitchConfig::CustomBroadcast { ring_size: 64 })
        .collect::<m![Ds / 4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Ds, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()'''
assert s.count(old)==1
s=s.replace(old,new)
save('p17-parallel-v-exact-ring64',s,(root/'p09-parallel-v-exact/src/ops.rs').read_text())

s=(root/'p07-quant-scale-vrf/src/device/sliding/qkv_head_local.rs').read_text()
s=s.replace('unsafe { scale.clone().reshape() }','unsafe { scale.reshape() }')
s=s.replace('let scale_vrf: NativeScale = unsafe { scale.reshape() };','let scale_vrf: NativeScale = unsafe { quant_scale.reshape() };')
save('p07b-quant-scale-vrf',s)
