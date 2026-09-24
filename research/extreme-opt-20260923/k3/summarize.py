from pathlib import Path
import json, hashlib, difflib
root=Path(__file__).resolve().parent
rows=[]
for p in sorted(root.glob('v*')):
    if not p.is_dir():continue
    s=p/'ffn7.rs'; sch=p/'schedule.json'; log=p/'compile.log'
    row={'variant':p.name,'source_sha256':hashlib.sha256(s.read_bytes()).hexdigest(),'intent':(p/'intent.txt').read_text()}
    if sch.exists():
        j=json.loads(sch.read_text());ins=j['instructions'];row.update(status='COMPILE_PASS',cycles=max(i['lifetime']['end'] for i in ins),instructions=len(ins),schedule_sha256=hashlib.sha256(sch.read_bytes()).hexdigest())
        row['delta_cycles']=row['cycles']-110520
        row['duration_by_type']={}
        for i in ins:row['duration_by_type'][i['tpe']]=row['duration_by_type'].get(i['tpe'],0)+i['lifetime']['end']-i['lifetime']['begin']
    elif (p/'SKIPPED.txt').exists():
        row['status']='SKIPPED_KNOWN_LOWERING';row['reason']=(p/'SKIPPED.txt').read_text()
    elif log.exists() and ('error' in log.read_text(errors='replace').lower() or 'panicked' in log.read_text(errors='replace')):
        row['status']='COMPILE_FAIL';row['tail']=log.read_text(errors='replace')[-2500:]
    else:row['status']='PENDING'
    rows.append(row)
(root/'ledger.json').write_text(json.dumps(rows,indent=2),newline='\n')
best=root/'v20-post-ring8-fixed'
if (best/'ffn7.rs').exists():
    baseline=root.parent/'baseline/src/device/shared/ffn7.rs'
    (best/'source.diff').write_text(''.join(difflib.unified_diff(baseline.read_text().splitlines(True),(best/'ffn7.rs').read_text().splitlines(True),fromfile='baseline/src/device/shared/ffn7.rs',tofile='v20/src/device/shared/ffn7.rs')),newline='\n')
for r in rows:print(r['variant'],r['status'],r.get('cycles'),r.get('instructions'),r.get('delta_cycles'))
