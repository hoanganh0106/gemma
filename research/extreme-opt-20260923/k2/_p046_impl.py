from pathlib import Path
p=Path(r'research/extreme-opt-20260923/packages/P046-k2-qhalf-fp32-merge/src/device/sliding/output31.rs')
s=p.read_text()
# Add the per-half mapping aliases after the current full-width WeightTile alias.
needle='pub(crate) type WeightTile = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]>;\n'
insert='''pub(crate) type WeightTile = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]>;

// A 2048-column weight half uses 16 row groups x 16 128-column groups = 256 slices.
type HalfColumns = m![H / 120 % 16, Qs / 128];
type HalfRows = m![H / 120 % 16, 1 # 16];
type HalfLevels = DmTensor<f8e4m3, Chip, OutputClusters, HalfColumns, m![Lq, Qs % 128]>;
pub(crate) type HalfXTrf = TrfTensor<f8e4m3, Chip, OutputClusters, HalfColumns, m![Lq], m![Qs % 128]>;
pub(crate) type HalfWeight = DmTensor<f8e4m3, Chip, OutputClusters, HalfColumns, m![H % 120, Qs % 128]>;
pub(crate) type HalfPartial = DmTensor<f32, Chip, OutputClusters, HalfRows, m![H % 120]>;
'''
assert needle in s
s=s.replace(needle,insert,1)
# Duplicate the exact two-level activation quantizer at a 128-column local extent.
a=s.index('pub(crate) fn quantise_x(')
b=s.index('\npub(crate) fn project_normalize_add(',a)
q=s[a:b]
qh=q.replace('pub(crate) fn quantise_x(', 'pub(crate) fn quantise_x_half(')
qh=qh.replace('x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>', 'x: &DmTensor<bf16, Chip, OutputClusters, HalfColumns, m![Qs % 128]>')
qh=qh.replace('-> XTrf {','-> HalfXTrf {').replace('let mut levels: Levels', 'let mut levels: HalfLevels')
qh=qh.replace('Qs % 256','Qs % 128').replace('Qs / 8 % 32','Qs / 8 % 16').replace('Qs / 4 % 64','Qs / 4 % 32').replace('Qs / 32 % 8','Qs / 32 % 4')
s=s[:b]+'\n'+qh+'\n'+s[b:]
# Add an FP32 half-contraction which keeps both activation levels and delays BF16 conversion.
needle='/// x as two exact f8 levels in the TRF: q0 = f8(16x), q1 = f8(16x - q0).\n'
helper='''/// Contract one contiguous Q half. Both Q halves are accumulated in FP32 before the
/// original projection BF16 boundary, so the only reassociation is FP32 addition grouping.
pub(crate) fn contract_half(device: &mut Device, weight: &HalfWeight, x_trf: &HalfXTrf, z: &mut HalfPartial) {
    device.main
        .begin(weight.view())
        .fetch::<m![H % 120, Qs / 32 % 4], m![Qs % 32]>()
        .collect::<m![H % 120, Qs / 32 % 4], m![Qs % 32]>()
        .contract_outer::<m![H % 120, Qs / 64 % 2], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<HalfRows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .transpose::<m![H % 120 / 4], m![H % 120 % 4 # 16]>()
        .commit_trim::<m![H % 120 % 4]>()
        .commit_view(z.view_mut());
}

'''
assert needle in s
s=s.replace(needle,helper+needle,1)
# Replace projection setup/projection with two Q-half stages and a final FP32 sum.
start=s.index('    // The whole weight in ONE DMA command: 120 rows per slice.')
end=s.index('    // Load each tail operand into an independent allocation',start)
new='''    // Load and quantize the first 2048 Q values, then contract it while the second
    // weight half is fetched into a separate full-device allocation.
    let x_q: HbmTensorView<'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };
    let x_half0_hbm = x_q.tile::<m![Qs], 2048, m![Qs = 2048 # 4096]>(0);
    let x_half0: DmTensor<bf16, Chip, OutputClusters, HalfColumns, m![Qs % 128]> =
        x_half0_hbm.to_dm(&mut device.tdma);
    let x_half0_trf = quantise_x_half(device, &x_half0);
    let weight_half0_hbm = weight.view().tile::<m![Qs], 2048, m![H, Qs = 2048 # 4096]>(0);
    let weight_half0: HalfWeight = weight_half0_hbm.to_dm(&mut device.tdma);
    let mut partial0: HalfPartial = DmTensor::new();
    contract_half(device, &weight_half0, &x_half0_trf, &mut partial0);

    // Second half uses the same 256-slice mapping. Its DMA is independent of the
    // first contraction and may overlap it in the static scheduler.
    let x_half1_hbm = x_q.tile::<m![Qs], 2048, m![Qs = 2048 # 4096]>(2048);
    let x_half1: DmTensor<bf16, Chip, OutputClusters, HalfColumns, m![Qs % 128]> =
        x_half1_hbm.to_dm(&mut device.tdma);
    let x_half1_trf = quantise_x_half(device, &x_half1);
    let weight_half1_hbm = weight.view().tile::<m![Qs], 2048, m![H, Qs = 2048 # 4096]>(2048);
    let weight_half1: HalfWeight = weight_half1_hbm.to_dm(&mut device.tdma);
    let mut partial1: HalfPartial = DmTensor::new();
    contract_half(device, &weight_half1, &x_half1_trf, &mut partial1);

    // Rejoin in FP32 and cross the same BF16 projection boundary only once.
    let partial1_vrf: VrfTensor<f32, Chip, OutputClusters, HalfRows, m![H % 120]> = device
        .sub.begin(partial1.view())
        .fetch::<m![1], m![H % 120]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .to_vrf();
    let mut z: Z = DmTensor::new();
    device.main
        .begin(partial0.view())
        .fetch::<m![1], m![H % 120]>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &partial1_vrf)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 120 % 4 # 16]>()
        .commit_trim::<m![H % 120 % 4]>()
        .commit_view(z.view_mut());

    // The 3,840 projected values now enter the unchanged RMS/residual tail.
    let mut hop: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    let z_gathered: DmTensor<bf16, Chip, OutputClusters, m![1 # 256], m![H % 1920]> =
        z.to_dm(&mut device.tdma);
    z_gathered.view().to_hbm_view(&mut device.tdma, hop.view_mut());

'''
s=s[:start]+new+s[end:]
p.write_text(s)

