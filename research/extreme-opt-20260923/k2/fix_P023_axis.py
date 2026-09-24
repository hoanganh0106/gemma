from pathlib import Path
p=Path(r'research/extreme-opt-20260923/packages/P023-k2-q-half-double-buffer/src/device/sliding/output31.rs')
s=p.read_text()
s=s.replace('axes![Lq = 2, Vr = 16];','axes![Lq = 2, Vr = 16, Qh = 4096];',1)
s=s.replace('type HalfColumns = m![H / 120 % 16, Qs / 128];','type HalfColumns = m![H / 120 % 16, Qh = 2048 / 128];')
s=s.replace('m![Lq, Qs % 128]>;\npub(crate) type HalfXTrf', 'm![Lq, Qh = 2048 % 128]>;\npub(crate) type HalfXTrf',1)
s=s.replace('m![Lq], m![Qs % 128]>;\npub(crate) type HalfWeight', 'm![Lq], m![Qh = 2048 % 128]>;\npub(crate) type HalfWeight',1)
s=s.replace('m![H % 120, Qs % 128]>;\npub(crate) type HalfPartial', 'm![H % 120, Qh = 2048 % 128]>;\npub(crate) type HalfPartial',1)
# The two helpers operate on the local 2048-axis, not the full Qs extent.
a=s.index('pub(crate) fn contract_half('); b=s.index('/// x as two exact f8 levels',a)
s=s[:a]+s[a:b].replace('Qs','Qh')+s[b:]
a=s.index('pub(crate) fn quantise_x_half('); b=s.index('pub(crate) fn project_normalize_add(',a)
s=s[:a]+s[a:b].replace('Qs','Qh')+s[b:]
# Reshape each full source into Qh, then make the two explicit subviews.
s=s.replace('let x_q: HbmTensorView<\'_, bf16, Chip, m![Qs]> = unsafe { x.view().reshape() };', "let x_q: HbmTensorView<'_, bf16, Chip, m![Qh]> = unsafe { x.view().reshape() };\n    let weight_q: HbmTensorView<'_, f8e4m3, Chip, m![H, Qh]> = unsafe { weight.view().reshape() };")
s=s.replace('x_q.tile::<m![Qs], 2048, m![Qs = 2048 # 4096]>', 'x_q.tile::<m![Qh], 2048, m![Qh = 2048 # 4096]>')
s=s.replace('weight.view().tile::<m![Qs], 2048, m![H, Qs = 2048 # 4096]>', 'weight_q.tile::<m![Qh], 2048, m![H, Qh = 2048 # 4096]>')
s=s.replace('HalfColumns, m![Qs % 128]', 'HalfColumns, m![Qh = 2048 % 128]')
p.write_text(s)
