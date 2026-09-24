"""Test exact BF16 activation/weight contraction instead of two FP8 terms in K2."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/P034-k2-bf16-contraction'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/device/sliding/output31.rs'
s = p.read_text()
s = s.replace('pub(crate) type XTrf = TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>;', 'pub(crate) type XTrf = TrfTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![1], m![Qs % 256]>;')
start, end = s.index('pub(crate) fn contract_tile('), s.index('/// x as two exact f8 levels')
contract = s[start:end]
contract = contract.replace('.fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()', '.fetch::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()\n        .fetch_cast::<bf16>()')
contract = contract.replace('.collect::<m![H % 120, Qs / 32 % 8], m![Qs % 32]>()', '.collect::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()')
contract = contract.replace('m![H % 120, Qs / 64 % 4], m![Qs % 64]', 'm![H % 120, Qs / 32 % 8], m![Qs % 32]')
contract = contract.replace('m![H % 120, Lq]', 'm![H % 120]')
contract = contract.replace('        .vector_intra_slice_reduce::<Lq, m![H % 120], m![1 # 4]>(IntraSliceReduceOpF32::Add)\n', '')
s = s[:start] + contract + s[end:]
start, end = s.index('/// x as two exact f8 levels'), s.index('pub(crate) fn project_normalize_add(')
quant = '''/// Preserve the power-of-two gain and BF16 activation directly in the TRF.
pub(crate) fn quantise_x(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
) -> XTrf {
    let scaled: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), X_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    device.sub
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 256]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf()
}

'''
s = s[:start] + quant + s[end:]
p.write_text(s)
(dest / 'frozen-source.json').write_text(json.dumps({str(f.relative_to(dest)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((dest / 'src').rglob('*')) if f.is_file()}, indent=2))
(dest / 'HYPOTHESIS.md').write_text('Keep BF16 activation at the original power-of-two gain 16, convert each FP8 weight exactly to BF16 in Fetch, then contract one BF16 term. Preserve all weights/scales, projection BF16 boundary, EPS_SCALED, full RMS/residual tail and public entrypoint. Removes two-FP8 construction and term reduction without dropping the low component: BF16 represents the input directly. Numerical accumulation grouping changes and must pass official correctness. Weight DMA bytes stay FP8; on-chip contraction bandwidth doubles, which may offset the saved passes.\n')
print(dest)
