from pathlib import Path
p=Path(r'research/extreme-opt-20260923/packages/P023-k2-q-half-double-buffer/src/device/sliding/output31.rs')
s=p.read_text()
s=s.replace('pub(crate) type HalfPartial = DmTensor<f32, Chip, OutputClusters, HalfRows, m![H % 120]>;', 'pub(crate) type HalfPartialVrf = VrfTensor<f32, Chip, OutputClusters, HalfRows, m![H % 120]>;')
a=s.index('pub(crate) fn contract_half(')
b=s.index('/// x as two exact f8 levels', a)
old=s[a:b]
base=old.replace('pub(crate) fn contract_half(device: &mut Device, weight: &HalfWeight, x_trf: &HalfXTrf) -> HalfPartial {', 'pub(crate) fn contract_half(device: &mut Device, weight: &HalfWeight, x_trf: &HalfXTrf) -> HalfPartialVrf {')
base=base.replace('        .vector_final()\n        .cast::<f32, m![1 # 8]>()\n        .commit()', '        .vector_final()\n        .to_vrf(&mut device.sub)')
# Replace final code that never had linebreak escaping.
base=base.replace('        .vector_final()\n        .commit()', '        .vector_final()\n        .to_vrf(&mut device.sub)')
# Regenerate helper from the common first-partial projection body.
assert '.to_vrf(&mut device.sub)' in base
add=base.replace('pub(crate) fn contract_half(', 'pub(crate) fn contract_half_add(')
add=add.replace('x_trf: &HalfXTrf) -> HalfPartialVrf {', 'x_trf: &HalfXTrf, first: &HalfPartialVrf, z: &mut Z) {')
add=add.replace('        .vector_inter_slice_reduce::<HalfRows, m![H % 120]>(InterSliceReduceOpF32::Add)\n        .vector_final()\n        .to_vrf(&mut device.sub)', '''        .vector_inter_slice_reduce::<HalfRows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_fp_binary(FpBinaryOp::AddF, first)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 120 % 4 # 16]>()
        .commit_trim::<m![H % 120 % 4]>()
        .commit_view(z.view_mut())''')
assert '.vector_fp_binary(FpBinaryOp::AddF, first)' in add
s=s[:a]+base+'\n'+add+s[b:]
# Replace project setup after first partial through the old two-partial BF16 merge.
start=s.index('    // Load and quantize the first 2048 Q values')
end=s.index('    // Load each tail operand into an independent allocation', start)
new='''    // Load the first Q half and keep its FP32 projection partial in VRF.
    let x_q: HbmTensorView<'_, bf16, Chip, m![Qh]> = unsafe { x.view().reshape() };
    let weight_q: HbmTensorView<'_, f8e4m3, Chip, m![H, Qh]> = unsafe { weight.view().reshape() };
    let x_half0_hbm = x_q.tile::<m![Qh], 2048, m![Qh = 2048 # 4096]>(0);
    let x_half0: DmTensor<bf16, Chip, OutputClusters, HalfColumns, m![Qh = 2048 % 128]> =
        x_half0_hbm.to_dm(&mut device.tdma);
    let x_half0_trf = quantise_x_half(device, &x_half0);
    let weight_half0_hbm = weight_q.tile::<m![Qh], 2048, m![H, Qh = 2048 # 4096]>(0);
    let weight_half0: HalfWeight = weight_half0_hbm.to_dm(&mut device.tdma);
    let partial0: HalfPartialVrf = contract_half(device, &weight_half0, &x_half0_trf);

    // Fetch and contract the second half. Its DMA can overlap the first projection;
    // the second FP32 contraction adds the first partial before the sole BF16 cast.
    let x_half1_hbm = x_q.tile::<m![Qh], 2048, m![Qh = 2048 # 4096]>(2048);
    let x_half1: DmTensor<bf16, Chip, OutputClusters, HalfColumns, m![Qh = 2048 % 128]> =
        x_half1_hbm.to_dm(&mut device.tdma);
    let x_half1_trf = quantise_x_half(device, &x_half1);
    let weight_half1_hbm = weight_q.tile::<m![Qh], 2048, m![H, Qh = 2048 # 4096]>(2048);
    let weight_half1: HalfWeight = weight_half1_hbm.to_dm(&mut device.tdma);
    let mut z: Z = DmTensor::new();
    contract_half_add(device, &weight_half1, &x_half1_trf, &partial0, &mut z);

'''
s=s[:start]+new+s[end:]
p.write_text(s)
