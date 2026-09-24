from pathlib import Path
import re
p=Path(r'research/extreme-opt-20260923/packages/P023-k2-q-half-double-buffer/src/device/sliding/output31.rs')
s=p.read_text().replace('type HalfRows = m![H / 120 % 16, 1 # 8];','type HalfRows = m![H / 120 % 16, 1 # 16];')
a=s.index('pub(crate) fn contract_half('); b=s.index('/// x as two exact f8 levels',a); q=s[a:b]
needle='.vector_intra_slice_reduce::<Lq, m![H % 120], m![1 # 4]>(IntraSliceReduceOpF32::Add)'
assert q.count(needle)==2
pos=q.index(needle)+len(needle); q=q[:pos]+'\n        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), 0.0625)'+q[pos:]
# On the second half, add the first half's reduced value at each of the 16 Q-group sites.
pos=q.index(needle, pos+100)+len(needle); q=q[:pos]+'\n        .vector_fp_binary(FpBinaryOp::AddF, first)'+q[pos:]
q=re.sub(r'\.vector_inter_slice_reduce::<HalfRows, m!\[H % 120\]>\(InterSliceReduceOpF32::Add\)\s*\.vector_fp_binary\(FpBinaryOp::AddF, first\)', '.vector_inter_slice_reduce::<HalfRows, m![H % 120]>(InterSliceReduceOpF32::Add)', q)
s=s[:a]+q+s[b:]; p.write_text(s)
