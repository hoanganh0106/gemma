from pathlib import Path
root=Path(__file__).resolve().parent
base=(root/'candidates/C15-ring16-tail240/output31.rs').read_text()
decl='type InputSlices = m![1 # 16, Qs / 256];\n'
base=base.replace('type Levels =',decl+'\ntype Levels =',1)
old='''    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };'''
seed='''    // HBM only initializes the first 16 slices with the 16 x blocks on each cluster.
    let x: DmTensor<bf16, Chip, OutputClusters, m![1 # 16, Ns, Gs], m![Ds]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, InputSlices, m![Qs % 256]> = unsafe { x.reshape() };'''
assert old in base
fused=base.replace(old,seed)
start=fused.index('pub(crate) fn quantise_x(');end=fused.index('pub(crate) fn project_normalize_add(',start)
fn=fused[start:end].replace('x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>','x: &DmTensor<bf16, Chip, OutputClusters, InputSlices, m![Qs % 256]>')
part='''        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()'''
assert fn.count(part)==2
fn=fn.replace(part,part+'\n        .switch::<SlidingOutputColumns, m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })')
fused=fused[:start]+fn+fused[end:]
separate=base.replace(old,seed+'''
    // Broadcast BF16 once while the independent weight DMA is running.
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = device.main
        .begin(x.view()).fetch::<m![1],m![Qs % 256]>()
        .switch::<SlidingOutputColumns,m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![Qs / 16 % 16],m![Qs % 16]>()
        .commit_trim::<m![Qs % 16]>().commit();''')
for name,source,note in [
 ('C18-x-fused-broadcast',fused,'C15 with x loaded once per cluster then explicit F32 switch broadcast fused into both existing quantization passes; every actual x block initialized in first16 slices.'),
 ('C19-x-bf16-broadcast',separate,'C15 with x loaded once per cluster then one separate BF16 switch broadcast, overlappable with weight DMA; original quantization function retained.')]:
 d=root/'candidates'/name;d.mkdir(parents=True,exist_ok=True)
 (d/'output31.rs').write_text(source);(d/'intent.txt').write_text(note)
