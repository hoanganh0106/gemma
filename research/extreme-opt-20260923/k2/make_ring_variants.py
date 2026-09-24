from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'candidates/C14-ring8/output31.rs').read_text()
for name,n in [('C15-ring16-tail240',240),('C16-ring32-tail120',120)]:
    out=s.replace('Vr = 8','Vr = '+str(3840//n)).replace('m![1 # 32, H / 480]',f'm![1 # {n//15}, H / {n}]').replace('m![1 # 32, Vr]',f'm![1 # {n//15}, Vr]').replace('H % 480',f'H % {n}').replace('H / 8 % 60',f'H / 8 % {n//8}').replace('H / 4 % 120',f'H / 4 % {n//4}').replace('m![H / 480]',f'm![H / {n}]').replace('ring_size: 8','ring_size: '+str(3840//n))
    out=out.replace('eight H-group',str(3840//n)+' H-group').replace('eight group',str(3840//n)+' group').replace('eight-slice',str(3840//n)+'-slice').replace('eight H-group',str(3840//n)+' H-group')
    d=root/'candidates'/name;d.mkdir(parents=True,exist_ok=True)
    (d/'output31.rs').write_text(out)
    (d/'intent.txt').write_text(f'C14 explicit ring gather with {3840//n} groups of {n} values; retain all H and all scalar arithmetic.')
