from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'v08-down-full-dma-10-5/ffn7.rs').read_text()
a=s.index('        let packed:',s.index('macro_rules! down_tile'))
b=s.index('        let z:',a)
s=s[:a]+'''        let packed = $w.view()
            .tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, Dt, Xp]>($start);
'''+s[b:]
needle='    let down_weight_packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, L % 7680]> = down_weight_packed.to_dm(&mut device.tdma);'
s=s.replace(needle,needle+'''
    let down_weight_packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, Dt, Xp]> = unsafe { down_weight_packed.reshape() };''')
d=r/'v08b-full-dma-10-5-view';d.mkdir(exist_ok=True)
(d/'ffn7.rs').write_text(s,newline='\n')
(d/'intent.txt').write_text('Repair v08 unsupported tiled-view reshape: reshape the full allocated tensor first, then form padded row views for10+5compute. All input values and arithmetic retained.',newline='\n')
