from pathlib import Path
import shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
s=(root/'p06-all-vrf/src/device/sliding/qkv_head_local.rs').read_text()
a=s.index('    use crate::device::layout::{Cluster, Slice};')
b=s.index('    let q = rope_one',a)
s=s[:a]+'''    // These are true initialized broadcast distributions, not padded axes.
    // A scalar offset selects one row, requested at every cluster and slice.
    let cos_row: DmTensor<bf16, Chip, m![2], m![256], m![Ds]> =
        cos.dma_gather_scaled(rope_offset);
    let sin_row: DmTensor<bf16, Chip, m![2], m![256], m![Ds]> =
        sin.dma_gather_scaled(rope_offset);
    // Relabel already broadcast copies; no element order is changed.
    let q_cos: QueryHeadTensor = unsafe { cos_row.reshape() };
    let q_sin: QueryHeadTensor = unsafe { sin_row.reshape() };
'''+s[b:]
p=root/'p13-rope-broadcast'
p.mkdir(exist_ok=True)
shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
(p/'src/device/sliding/qkv_head_local.rs').write_text(s)
