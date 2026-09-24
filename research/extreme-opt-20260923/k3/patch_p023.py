from pathlib import Path
p=Path('research/extreme-opt-20260923/k3/v28-down-bf16-exact/ffn7.rs')
s=p.read_text()
start=s.index('fn quantize_gather(')
end=s.index('/// The down projection\'s block scales',start)
new='''fn quantize_gather(
    device: &mut Device,
    x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![L % 120]>,
) -> TrfTensor<bf16, Chip, UpGateClusters, DownSlices, m![1], m![Dt, Xp]> {
    // GeGLU already produces BF16. Replicate those exact values over the down-projection
    // slices instead of quantizing them into two FP8 components.
    let x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![T2 = 1, Xg]> = unsafe { x.reshape() };
    let x: DmTensor<bf16, Chip, UpGateClusters, GroupsX, m![T2 = 1, Xd % 120]> = unsafe { x.reshape() };
    let gathered: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![T2 = 1, Xd]> = device
        .main
        .begin(x.view())
        .fetch::<m![T2 = 1, Xd / 24 % 5], m![Xd % 24]>()
        .switch::<DownSlices, m![T2 = 1, Xd / 24 % 5, Xd / 120]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![T2 = 1, Xd / 24 % 5, Xd / 120], m![Xd % 24 # 32]>()
        .commit_trim::<m![Xd % 24]>()
        .commit();
    let gathered: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![T2 = 1, Dt, Xp]> =
        unsafe { gathered.reshape() };
    device.sub
        .begin(gathered.view())
        .fetch::<m![T2 = 1, Dt, Xp / 16], m![Xp % 16]>()
        .collect::<m![T2 = 1, Dt, Xp / 16], m![Xp % 16]>()
        .to_trf()
}

'''
s=s[:start]+new+s[end:]
# Replace down tile's one-step f4->f8 streaming contraction with exact two-stage decode to BF16 DM, then BF16 contraction.
old='''        let z: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16]> = $device
            .main
            .begin(packed.view())
            .fetch::<m![H / 2 % 15 = $len, Dt], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![H / 2 % 15 = $len, Dt, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![H / 2 % 15 = $len, Dt]>()
            .contract_lane::<m![H / 2 % 15 = $len, Dt], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .cast::<bf16, m![Xp / 16 # 16]>()
            .commit_trim::<m![Xp / 16]>()
            .commit();'''
new='''        let weight_f8: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]> = $device
            .main
            .begin(packed.view())
            .fetch::<m![H / 2 % 15 = $len, Dt], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]>()
            .commit_trim::<m![Xp % 32]>()
            .commit();
        let weight_bf16: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16], m![Xp % 16]> = $device
            .main
            .begin(weight_f8.view())
            .fetch::<m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]>()
            .fetch_table_lookup::<bf16>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 16], m![Xp % 16]>()
            .commit_trim::<m![Xp % 16]>()
            .commit();
        let z: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16]> = $device
            .main
            .begin(weight_bf16.view())
            .fetch::<m![H / 2 % 15 = $len, Dt, Xp / 16], m![Xp % 16]>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 16], m![Xp % 16]>()
            .contract_outer::<m![H / 2 % 15 = $len, Dt], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![H / 2 % 15 = $len, Dt]>()
            .contract_lane::<m![H / 2 % 15 = $len, Dt], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .cast::<bf16, m![Xp / 16 # 16]>()
            .commit_trim::<m![Xp / 16]>()
            .commit();'''
assert old in s, 'down contraction block not found'
s=s.replace(old,new,1)
# Remove down activation gain undo (activation remains full BF16 GeGLU value).
s=s.replace('''        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &up_global_scale_vrf)
        .vector_intra_slice_reduce::<C2, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_GAIN)
        .vector_widen_pad''','''        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &up_global_scale_vrf)
        .vector_intra_slice_reduce::<C2, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad''')
p.write_text(s)
