from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parent/'baseline/src/device/shared/ffn7.rs').read_text()
s=s.replace('Sb = 240, Sd = 480','Sb = 240, Sd = 480, UG2 = 2')
start=s.index('macro_rules! block_sums_all {')
end=s.index('/// The block scales',start)
old=s[start:end]
new=old.replace('block_sums_all','block_sums_pair').replace('$w:ident, $x_trf:ident','$u:ident, $g:ident, $x_trf:ident')
first=new.index('        let packed:')
second=new.index('        let packed:',first+1)
new=new[:first]+'''        let mut packed: DmTensor<f4e2m1, Chip, UpGateClusters, RowSlices, m![UG2, L % 30, H]> = DmTensor::new();
        let u: HbmTensorView<f4e2m1, Chip, m![UG2 = 1, L, H]> = unsafe { $u.view().reshape() };
        let g: HbmTensorView<f4e2m1, Chip, m![UG2 = 1, L, H]> = unsafe { $g.view().reshape() };
        u.to_dm_view(&mut $device.tdma, packed.view_mut().tile::<m![UG2], 1, m![UG2 = 1 #{!} 2, L % 30, H]>(0));
        g.to_dm_view(&mut $device.tdma, packed.view_mut().tile::<m![UG2], 1, m![UG2 = 1 #{!} 2, L % 30, H]>(1));
'''+new[second:]
# Every row/time shape carries the matrix discriminator, preserving separate outputs.
new=new.replace('m![L % 30','m![UG2, L % 30')
s=s[:start]+new+s[end:]
s=s.replace('z: &DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]>','z: DmTensorView<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]>')
a=s.index('fn apply_scales(');b=s.index('/// A cluster',a)
s=s[:a]+s[a:b].replace('.begin(z.view())','.begin(z)')+s[b:]
s=s.replace('''    let up_z = block_sums_all!(device, up_weight_packed, x_trf);''','''    let pair_z = block_sums_pair!(device, up_weight_packed, gate_weight_packed, x_trf);
    let up_z: DmTensorView<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]> = unsafe { pair_z.view().tile::<m![UG2], 1, m![UG2 = 1 # 2, L % 30, Sb]>(0).reshape() };
    let gate_z: DmTensorView<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]> = unsafe { pair_z.view().tile::<m![UG2], 1, m![UG2 = 1 # 2, L % 30, Sb]>(1).reshape() };''')
s=s.replace('    let gate_z = block_sums_all!(device, gate_weight_packed, x_trf);\n','')
s=s.replace('apply_scales(device, &up_z,','apply_scales(device, up_z,').replace('apply_scales(device, &gate_z,','apply_scales(device, gate_z,')
d=root/'v12-pair-upgate';d.mkdir(exist_ok=True)
(d/'ffn7.rs').write_text(s,newline='\n')
(d/'intent.txt').write_text('Up/gate packed tensors loaded into separate slices of an extra local matrix axis, then one LUT/contraction pass. Both matrix weights/scales preserved and same accumulation order per output. Saves one decode setup, increases packed live allocation, delays first contraction until both matrices loaded.',newline='\n')
