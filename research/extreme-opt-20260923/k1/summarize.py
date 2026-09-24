from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
rows=[]
for p in sorted(root.glob('p*')):
    if not p.is_dir():continue
    s=p/'schedule.json'; log=p/'compile.log'
    row={'probe':p.name,'source_sha256':hashlib.sha256((p/'src/device/sliding/qkv_head_local.rs').read_bytes()).hexdigest()}
    if s.exists():
        d=json.loads(s.read_text()); ins=d['instructions']
        row.update(cycles=max(i['lifetime']['end'] for i in ins),instructions=len(ins),schedule_sha256=hashlib.sha256(s.read_bytes()).hexdigest())
    elif log.exists():
        row['error']='\n'.join([l for l in log.read_text(errors='replace').splitlines() if 'error' in l or 'panic' in l][-8:])
    rows.append(row)
print(json.dumps(rows,indent=2))
(root/'summary.json').write_text(json.dumps(rows,indent=2))
