from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C04-sw-sub/output31.rs').read_text()
def tail(s,n):
    return s.replace('Vr = 8','Vr = '+str(3840//n)).replace('m![1 # 32, H / 480]',f'm![1 # {n//15}, H / {n}]').replace('m![1 # 32, Vr]',f'm![1 # {n//15}, Vr]').replace('H % 480',f'H % {n}').replace('H / 8 % 60',f'H / 8 % {n//8}').replace('H / 4 % 120',f'H / 4 % {n//4}')
candidates={
 'C09-tail120':(tail(s,120),'C04 with 32 groups of 120 values in tail.'),
 'C10-tail1920':(tail(s,1920),'C04 with two groups of 1920 values in tail; register pressure probe.'),
 'C11-sum-norm':(s.replace('        .vector_fp_div(H_F32)\n','').replace('FpBinaryOp::AddF, EPS_SCALED','FpBinaryOp::AddF, EPS_SCALED * H_F32').replace('        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms_weight)\n','        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms_weight)\n        // sqrt(H=3840), rounded once to FP32; this is a shape constant.\n        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), 61.967735290527344f32)\n'), 'Sum normalization: remove divide by H before reduction, scale epsilon by H and SW by sqrt(H); mathematically equivalent with changed FP32 reassociation.'),
 'C12-direct-reciprocal':(s.replace('        .vector_stash()\n        .vector_fp_unary(FpUnaryOp::Sqrt)\n        .vector_fp_div(Stash)','        .vector_fp_unary(FpUnaryOp::Sqrt)\n        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0)'), 'Compute 1/sqrt(t) directly using divider argument swapping rather than sqrt(t)/t; same real arithmetic with changed FP32 rounding.'),
}
for name,(source,note) in candidates.items():
    d=root/'candidates'/name;d.mkdir(parents=True,exist_ok=True)
    (d/'output31.rs').write_text(source)
    (d/'intent.txt').write_text(note)
