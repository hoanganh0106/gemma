from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C04-sw-sub/output31.rs').read_text()
prefix=s[:s.index('    // Load each tail operand')]
prefix=prefix.replace('Vr = 8','Vr = 16')
tail='''    // Keep the projected BF16 row partitions on the cluster that computed them.
    // Only the scalar mean-square crosses the cluster boundary.
    type LocalDm<D> = DmTensor<D, Chip, OutputClusters, Rows, m![H % 120]>;
    type LocalVrf = VrfTensor<f32, Chip, OutputClusters, Rows, m![H % 120]>;
    type MeanDm = DmTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]>;
    type MeanVrf = VrfTensor<f32, Chip, OutputClusters, Rows, m![1 # 8]>;

    let scale_dm: LocalDm<bf16> = weight_scale.to_dm(&mut device.tdma);
    let scale: LocalVrf = device.sub.begin(scale_dm.view())
        .fetch::<m![1], m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>().to_vrf();
    let rms_dm: LocalDm<bf16> = rms_weight.to_dm(&mut device.tdma);
    let rms: LocalVrf = device.sub.begin(rms_dm.view())
        .fetch::<m![1], m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>().to_vrf();
    let sw: LocalVrf = device.sub.begin(scale_dm.view())
        .fetch::<m![1], m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms)
        .vector_widen_concat::<m![H / 8 % 15], m![H % 8]>()
        .vector_final().to_vrf();
    let residual_dm: LocalDm<bf16> = residual_hbm.to_dm(&mut device.tdma);
    let residual: LocalVrf = device.sub.begin(residual_dm.view())
        .fetch::<m![1], m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>().to_vrf();

    // Divide by global H, so adding the peer gives the global mean square.
    let local_mean: DmTensor<f32, Chip, OutputClusters, m![Vr, 1 # 16], m![1 # 8]> = device.main.begin(z.view())
        .fetch::<m![1], m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15], m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<m![Vr, 1 # 16], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // Pure renaming: Vr occupies exactly the same 16 strided physical slices as Rows.
    let local_mean: MeanDm = unsafe { local_mean.reshape() };
    let peer_mean: MeanDm = local_mean.view().cluster_swap().to_dm(&mut device.tdma);
    let peer: MeanVrf = device.sub.begin(peer_mean.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS_SCALED)
        .vector_widen_pad::<m![1 # 8]>().vector_final().to_vrf();
    let inv: MeanVrf = device.main.begin(local_mean.view())
        .fetch::<m![1],m![1 # 8]>().collect::<m![1],m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, &peer)
        .vector_stash().vector_fp_unary(FpUnaryOp::Sqrt).vector_fp_div(Stash)
        .vector_widen_pad::<m![1 # 8]>().vector_final().to_vrf(&mut device.sub);
    let out: LocalDm<bf16> = device.main.begin(z.view())
        .fetch::<m![1],m![H % 120]>().fetch_cast::<f32>()
        .collect::<m![H / 8 % 15],m![H % 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 30],m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sw)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &inv)
        .vector_widen_concat::<m![H / 8 % 15],m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add,&residual)
        .vector_final().cast::<bf16,m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>().commit();
    out.view().to_hbm_view(&mut device.tdma,residual_hbm.view_mut());
}
'''
d=root/'candidates/C08-local-scalar';d.mkdir(parents=True,exist_ok=True)
(d/'output31.rs').write_text(prefix+tail)
(d/'intent.txt').write_text('Keep BF16 projections local; exchange only scalar mean-square via cluster_swap; preserve all scales and residual. Reduction tree changes but FP32 operations remain.')
