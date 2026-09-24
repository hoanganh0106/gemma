from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
base=root.parent/'baseline'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
allowed={'src/device/sliding/qkv_head_local.rs','src/ops.rs'}
rows=[]
for p in sorted(root.glob('p*')):
 if not p.is_dir():continue
 changed=[];semantic=[];tree=[]
 for f in sorted(base.rglob('*')):
  if not f.is_file():continue
  rel=f.relative_to(base).as_posix();dst=p/rel
  if not dst.is_file():raise RuntimeError(f'Missing {p.name}/{rel}')
  h=digest(dst);tree.append(f'{rel}\0{h}\n')
  if h!=digest(f):
   changed.append(rel)
   if dst.read_text()!=f.read_text():semantic.append(rel)
 outside=set(changed)-allowed
 if outside:raise RuntimeError(f'Unauthorized candidate changes: {p.name}: {outside}')
 row={'probe':p.name,'changed_files':changed,'semantic_changed_files':semantic,'candidate_tree_sha256':hashlib.sha256(''.join(tree).encode()).hexdigest(),
      'cargo_lock_matches_baseline':digest(p/'Cargo.lock')==digest(base/'Cargo.lock'),
      'ops_sha256':digest(p/'src/ops.rs')}
 rows.append(row)
(root/'audit.json').write_text(json.dumps(rows,indent=2))
print(f'{len(rows)} candidate source trees audited; only allowed qkv helper/ops files differ; all lockfiles exact.')
for r in rows:
 if 'src/ops.rs' in r['semantic_changed_files']:print(r['probe'],'requires matching ops body',r['ops_sha256'])
