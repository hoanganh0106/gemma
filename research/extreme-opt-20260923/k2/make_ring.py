from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C03-sw-vrf-clean/output31.rs').read_text()
start=s.index('    // Ring of sixteen:')
end=s.index('    let inv_rms:',start)
s=s[:start]+'''    // Gather the eight group partials explicitly within each eight-slice ring.
    // Each active tail slice receives all eight H-group partials as time values.
    let mean: DmTensor<f32, Chip, Vc, m![1 # 32, Vr], m![1 # 8]> = device
        .main.begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![1 # 32, Vr], m![H / 480]>(SwitchConfig::CustomBroadcast { ring_size: 8 })
        .collect::<m![H / 480], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // The scalar arithmetic is unchanged; only the cross-slice sum is expressed explicitly.
''' + s[end:]
start=s.index('    let inv_rms:')
s=s[:start]+s[start:].replace('.begin(partial.view())','.begin(mean.view())',1).replace('        .vector_inter_slice_reduce::<m![1 # 32, Vr], m![1]>(InterSliceReduceOpF32::Add)\n','',1)
d=root/'candidates/C14-ring8';d.mkdir(parents=True,exist_ok=True)
(d/'output31.rs').write_text(s)
(d/'intent.txt').write_text('Replace global inter-slice reduction with explicit eight-slice switch gather and intra-slice sum. Scalar epsilon/sqrt/div unchanged; all eight partials participate, changed FP32 sum grouping.')
