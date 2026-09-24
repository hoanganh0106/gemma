from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C08-local-scalar/output31.rs').read_text()
s=s.replace('    let local_mean: DmTensor<f32, Chip, OutputClusters, m![Vr, 1 # 16], m![1 # 8]> = device.main.begin(z.view())','    let local_partial: MeanDm = device.main.begin(z.view())')
s=s.replace('        .vector_inter_slice_reduce::<m![Vr, 1 # 16], m![1]>(InterSliceReduceOpF32::Add)\n','')
at=s.index('    // Pure renaming:')
s=s[:at]+'''    // The row dimension is strided by 16 slices, so gather explicitly over
    // the 256-slice ring instead of requiring VRU reduction of a non-inner axis.
    let local_mean: DmTensor<f32, Chip, OutputClusters, m![Vr, 1 # 16], m![1 # 8]> = device.main
        .begin(local_partial.view()).fetch::<m![1], m![1 # 8]>()
        .switch::<m![Vr, 1 # 16], m![H / 120 % 16]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![H / 120 % 16], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
''' + s[at:]
d=root/'candidates/C17-local-ring';d.mkdir(parents=True,exist_ok=True)
(d/'output31.rs').write_text(s)
(d/'intent.txt').write_text('C08 topology with explicit 256-slice gather of the 16 strided local row partials; scalar-only cluster exchange and all weighted RMS semantics preserved.')
