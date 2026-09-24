from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parent
rows=[]
for d in sorted((root/'candidates').iterdir()):
    log=(d/'compile.log').read_text(errors='replace') if (d/'compile.log').exists() else ''
    row={'candidate':d.name,'intent':(d/'intent.txt').read_text(),'source_sha256':hashlib.sha256((d/'output31.rs').read_bytes()).hexdigest()}
    p=d/'schedule.json'
    if p.exists():
        x=json.loads(p.read_text());i=x['instructions']
        row.update(status='COMPILE_PASS',cycles=max(v['lifetime']['end'] for v in i),instructions=len(i),schedule_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    else:
        row['status']='COMPILE_FAIL' if ('error' in log or 'panicked' in log) else 'PENDING_OR_INTERRUPTED'
        row['log_tail']=log[-1800:]
    rows.append(row)
(root/'ledger.json').write_text(json.dumps(rows,indent=2))
for r in rows: print(r['candidate'],r['status'],r.get('cycles'),r.get('instructions'))
