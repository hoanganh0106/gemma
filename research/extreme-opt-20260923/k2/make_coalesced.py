from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C03-sw-vrf-clean/output31.rs').read_text()
s=s.replace('use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};','use crate::device::layout::OutputClusters;\ntype SlidingOutputColumns = m![H / 60 % 32, Qs / 512];\ntype SlidingOutputRows = m![H / 60 % 32, 1 # 8];')
s=s.replace('H % 120','H % 60').replace('Qs % 256','Qs % 512').replace('Qs / 32 % 8','Qs / 32 % 16').replace('Qs / 64 % 4','Qs / 64 % 8').replace('Qs / 8 % 32','Qs / 8 % 64').replace('Qs / 4 % 64','Qs / 4 % 128')
s=s.replace('m![H / 120 % 16, Ns, Gs], m![Ds]>', 'm![H / 60 % 32, Ns], m![Gs, Ds]>')
s=s.replace('120 rows per slice','60 rows per slice, with 512 contiguous weight bytes per row').replace('120 rows per row slice','60 rows per row slice')
d=root/'candidates/C13-weight512';d.mkdir(parents=True,exist_ok=True)
(d/'output31.rs').write_text(s)
(d/'intent.txt').write_text('Retile projection from 120 rows x 256 columns to 60 x 512 on each slice. Still all 256 slices and identical FP8 weight volume; doubles contiguous HBM segment, halves strided rows. Same two-level x quantization and BF16 projection boundary; changed FP32 reduction grouping.')
