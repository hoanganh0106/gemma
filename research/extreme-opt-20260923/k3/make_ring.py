from pathlib import Path
r=Path(__file__).resolve().parent
for name,parent in [('v16-ring8','v14-post-geglu-vrf'),('v17-down15-ring8','v15-down15-post-geglu')]:
    s=(r/parent/'ffn7.rs').read_text()
    for cluster,outmap in [('UpGateClusters','Gathered'),('Cluster','m![1 # 32, Dummy8]')]:
        needle=f'    let rms: '+('DmTensor' if cluster=='UpGateClusters' else 'VrfTensor')+f'<f32, Chip, {cluster}, {outmap}, m![1 # 8]> = device'
        a=s.index(needle)
        q=f'''    // Explicitly gather the eight H-group partials in each physical ring.
    // Every valid partial participates; only the cross-slice reduction implementation changes.
    let mean: DmTensor<f32, Chip, {cluster}, {outmap}, m![1 # 8]> = device
        .main.begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<{outmap}, m![H / 480]>(SwitchConfig::CustomBroadcast {{ ring_size: 8 }})
        .collect::<m![H / 480], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
'''
        b=s.index('    let weight_dm:',a)
        old=s[a:b].replace('.begin(mean_square.view())','.begin(mean.view())')
        old=old.replace(f'        .vector_inter_slice_reduce::<{outmap}, m![1]>(InterSliceReduceOpF32::Add)\n','')
        s=s[:a]+q+old+s[b:]
    d=r/name;d.mkdir(exist_ok=True)
    (d/'ffn7.rs').write_text(s,newline='\n')
    (d/'intent.txt').write_text('Explicit eight-slice switch gather plus intra-slice sum for both RMS reductions, preserving all eight partials and scalar arithmetic. FP32 addition association may change; requires stress correctness. Based on '+parent,newline='\n')

# Repair v12 compiler rejection: preserve tiled view through scale contraction,
# without an unsupported LowLevelReshape on that view.
s=(r/'v12-pair-upgate/ffn7.rs').read_text()
s=s.replace('z: DmTensorView<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]>','z: DmTensorView<bf16, Chip, UpGateClusters, Gathered, m![UG2 = 1 # 2, L % 30, Sb]>')
for label,index in [('up',0),('gate',1)]:
    a=s.index(f'    let {label}_z: DmTensorView')
    b=s.index('\n',a)
    s=s[:a]+f'    let {label}_z = pair_z.view().tile::<m![UG2], 1, m![UG2 = 1 # 2, L % 30, Sb]>({index});'+s[b:]
d=r/'v12b-pair-upgate-view';d.mkdir(exist_ok=True)
(d/'ffn7.rs').write_text(s,newline='\n')
(d/'intent.txt').write_text('Repair v12 unsupported tiled-view reshape by retaining its fixed/padded matrix axis in apply_scales input. Real arithmetic unchanged.',newline='\n')
