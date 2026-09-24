from pathlib import Path
r=Path(__file__).resolve().parent
s=(r.parent/'baseline/src/device/shared/ffn7.rs').read_text()
s=s.replace('Sb = 240, Sd = 480','Sb = 240, Sd = 480, ClusterCopies = 2')
a=s.index('    // The two clusters\' partial results meet in HBM')
b=s.index('\n    let down_global_scale:',a)
q='''    // Cluster 0 receives local contribution in C2=0 and its peer in C2=1.
    // Cluster 1 holds the reverse order and is discarded by the final padded cluster mapping.
    // All element moves below are real SRAM DMA; reshape only renames identical wire layouts.
    let partial: DmTensor<f32, Chip, m![ClusterCopies], m![Hg, 1 # 2], m![Pw]> = unsafe { partial.reshape() };
    let mut merged: DmTensor<f32, Chip, m![ClusterCopies], m![1 # 32, Hg / 16], m![C2, Hg % 16, Pw = 30]> = DmTensor::new();
    partial.view().tile::<m![Pw], 30, m![Pw = 30 # 64]>(0)
        .to_dm_view(&mut device.tdma, merged.view_mut().tile::<m![C2], 1, m![C2 = 1 #{!} 2, Hg % 16, Pw = 30]>(0));
    partial.view().tile::<m![Pw], 30, m![Pw = 30 # 64]>(0).cluster_swap()
        .to_dm_view(&mut device.tdma, merged.view_mut().tile::<m![C2], 1, m![C2 = 1 #{!} 2, Hg % 16, Pw = 30]>(1));
    let partial: DmTensor<f32, Chip, Cluster, ReducingSlices, m![C2, H % 480]> = unsafe { merged.reshape() };
'''
s=s[:a]+q+s[b:]
d=r/'v13-sram-partial';d.mkdir(exist_ok=True)
(d/'ffn7.rs').write_text(s,newline='\n')
(d/'intent.txt').write_text('Replace padded partial HBM store/load with local and peer SRAM DMA into two contribution slots; arithmetic unchanged. Cluster0 retains original C2 order; cluster1 reversed/discarded as padding. Needs compile and correctness confirmation.',newline='\n')
