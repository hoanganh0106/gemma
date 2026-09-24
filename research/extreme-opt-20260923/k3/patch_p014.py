from pathlib import Path
p=Path('research/extreme-opt-20260923/k3/v22-down-prefetch-first/ffn7.rs')
s=p.read_text()
old='''        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $device.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { packed.reshape() };'''
new='''        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { $w.reshape() };'''
assert old in s
s=s.replace(old,new)
# Define typed prefetch for row 0 after macro.
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
# Prefetch row tile before GeGLU output and keep value for later contraction.
old='''    let gate_factor = gate_factor_vrf(device, gate_global_scale);
    let x = geglu(device, &up, &gate, &gate_factor);'''
new='''    let gate_factor = gate_factor_vrf(device, gate_global_scale);
    // Weight traffic is activation independent; stage one down tile under GeGLU work.
    down_weight_tile!(device, down_weight_packed, down_w0, 0, 5);
    let x = geglu(device, &up, &gate, &gate_factor);'''
assert old in s
s=s.replace(old,new)
old='''    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 0, 5);
    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 5, 5);
    down_tile!(device, down_weight_packed, x_trf, down_s, partial, 10, 5);'''
new='''    down_tile!(device, down_w0, x_trf, down_s, partial, 0, 5);
    down_weight_tile!(device, down_weight_packed, down_w1, 5, 5);
    down_tile!(device, down_w1, x_trf, down_s, partial, 5, 5);
    down_weight_tile!(device, down_weight_packed, down_w2, 10, 5);
    down_tile!(device, down_w2, x_trf, down_s, partial, 10, 5);'''
assert old in s
s=s.replace(old,new)
p.write_text(s)
