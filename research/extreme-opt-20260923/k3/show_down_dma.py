import json
from pathlib import Path
for pkg in ['P015-k3-down-double-buffer']:
 d=json.loads((Path('research/extreme-opt-20260923/packages')/pkg/'schedule.json').read_text())
 print('\n',pkg)
 for i in d['instructions']:
  desc=i.get('description','')
  if i['tpe']=='DmaLoad' and ('down_weight' in desc or 'to_dm' in desc or 'ffn7.rs:'):
   if 'view' in desc or 'to_dm' in desc or 'down_weight' in desc: print(i['index'],i['tpe'],i.get('lifetime'),desc[:110].replace('\n',' '))


