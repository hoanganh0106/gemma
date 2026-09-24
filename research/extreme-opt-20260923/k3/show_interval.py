import json
from pathlib import Path
p=Path('research/extreme-opt-20260923/k3/v20-post-ring8-fixed/schedule.json')
d=json.loads(p.read_text())
for i in d['instructions']:
 b=i['lifetime']['begin'];e=i['lifetime']['end']; desc=i.get('description','').replace('\n',' ')
 if 68000<=b<=99000 and i['tpe'] in ('Main','Sub','DmaLoad'):
  print(i['index'],i['tpe'],b,e,desc[:135])

