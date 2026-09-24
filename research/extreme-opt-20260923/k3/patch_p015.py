from pathlib import Path
p=Path('research/extreme-opt-20260923/k3/v23-down-double-buffer/ffn7.rs')
s=p.read_text()
old='''macro_rules! down_tile {
    ($device:ident, $w:ident, $x_trf:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $device.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { packed.reshape() };'''
new='''macro_rules! down_tile {
    ($device:ident, $w:ident, $x_trf:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { $w.reshape() };'''
assert old in s
s=s.replace(old,new)
anchor='''    }};
}

pub(crate) fn feedforward('''
insert='''    }};
}

macro_rules! down_weight_tile {
    ($device:ident, $w:ident, $name:ident, $start:literal, $len:literal) => {
        let $name: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $device.tdma);
    };
}

pub(crate) fn feedforward('''
assert anchor in s
s=s.replace(anchor,insert,1)
old='''    // Three disjoint five-row tiles cover the same rows as the former 10+5 split.
    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 0, 5);
    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 5, 5);
    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 10, 5);'''
new='''    // Three disjoint five-row tiles cover the same rows as the former 10+5 split.
    down_weight_tile!(device, down_weight_packed, down_w0, 0, 5);
    down_weight_tile!(device, down_weight_packed, down_w1, 5, 5);
    down_tile!(device, down_w0, x_trf, down_s, partial, 0, 5);
    down_tile!(device, down_w1, x_trf, down_s, partial, 5, 5);
    down_weight_tile!(device, down_weight_packed, down_w2, 10, 5);
    down_tile!(device, down_w2, x_trf, down_s, partial, 10, 5);'''
assert old in s
s=s.replace(old,new)
p.write_text(s)
