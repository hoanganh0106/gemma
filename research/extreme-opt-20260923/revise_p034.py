"""SDK Fetch cannot convert FP8 directly to BF16; use its documented Cast stage."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
dest = root / 'packages/P034-k2-bf16-contraction'
p = dest / 'src/device/sliding/output31.rs'
s = p.read_text()
assert s.count('        .fetch_cast::<bf16>()') == 1
s = s.replace('        .fetch_cast::<bf16>()\n', '')
s = s.replace('        .fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()', '        .fetch::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()')
marker = 'pub(crate) fn contract_tile(device: &mut Device, weight: &WeightTile, x_trf: &XTrf, z: &mut Z) {\n'
assert s.count(marker) == 1
s = s.replace(marker, marker + '''    let weight: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]> = device.sub
        .begin(weight.view())
        .fetch::<m![H % 120, Qs / 8 % 32], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, Qs / 8 % 32], m![Qs % 8]>()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
''')
p.write_text(s)
(dest / 'frozen-source.json').write_text(json.dumps({str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dest / 'src').rglob('*')) if f.is_file()}, indent=2))
with (dest / 'HYPOTHESIS.md').open('a') as f:
    f.write('Revision: SDK 0.8.1 rejects FetchCast<bf16> for FP8. Convert FP8->FP32 in Fetch, FP32->BF16 exactly in Cast, then materialize a BF16 weight DM tile before contraction. This explicit extra Sub/DM pass is included in the schedule cost.\n')
