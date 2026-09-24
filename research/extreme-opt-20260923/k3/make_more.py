from pathlib import Path
root=Path(__file__).resolve().parent
s=(root.parent/'baseline/src/device/shared/ffn7.rs').read_text()
def save(name,text,reason):
    d=root/name;d.mkdir(exist_ok=True)
    (d/'ffn7.rs').write_text(text,newline='\n')
    (d/'intent.txt').write_text(reason,newline='\n')

flat=s.replace('Sb = 240, Sd = 480','Sb = 240, Sd = 480, FlatScale = 3686400')
flat=flat.replace('    let s: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> = scale.to_dm(&mut device.tdma);', '''    let flat: HbmTensorView<f8e4m3, Chip, m![FlatScale]> = unsafe { scale.view().reshape() };
    let s: DmTensor<f8e4m3, Chip, m![FlatScale / 1843200], m![FlatScale / 7200 % 256], m![FlatScale % 7200]> = flat.to_dm(&mut device.tdma);''')
save('v07-flat-up-scales',flat,'Flatten up/gate scales contiguous 7200 bytes per physical slice before DMA; reshape identical wire order.')

prefix='''        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $device.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { packed.reshape() };'''
replacement='''        let packed: DmTensorView<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> = unsafe { $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, L % 7680]>($start)
            .reshape() };'''
assert prefix in s
dm=s.replace(prefix,replacement)
a=dm.index('macro_rules! down_tile')
b=dm.index('pub(crate) fn feedforward',a)
dm=dm[:a]+dm[a:b].replace('.begin(packed.view())','.begin(packed)')+dm[b:]
dm=dm.replace('    let down_s = down_scales_bf16(device, down_weight_scale);','''    let down_s = down_scales_bf16(device, down_weight_scale);
    let down_weight_packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, L % 7680]> = down_weight_packed.to_dm(&mut device.tdma);''')
start=dm.index('    // SDK 0.8.1 requires this DMA row factor')
end=dm.index('\n    // Two neighbouring slices',start)
for name,tiles in [('v08-down-full-dma-10-5',[(0,10),(10,5)]),('v09-down-full-dma-15',[(0,15)]),('v10-down-full-dma-5',[(0,5),(5,5),(10,5)])]:
    calls=''.join(f'    down_tile!(device, down_weight_packed, x_trf, down_s, partial, {i}, {n});\n' for i,n in tiles)
    save(name,dm[:start]+calls+dm[end:],'One full down DMA, compute tiled views '+str(tiles)+'. No arithmetic changes; tile view preserves physical per-slice rows.')

# Remove an unused read from uninitialized data memory. It was intended as a command warmup.
a=s.index('    // WARM-UP.')
b=s.index('    let x: DmTensor',a)
save('v11-no-warmup',s[:a]+s[b:],'Remove unused Sub warmup on uninitialized DM. Semantics unchanged; hardware issuer startup behavior may change.')
