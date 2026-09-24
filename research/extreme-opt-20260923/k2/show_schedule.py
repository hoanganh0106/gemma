import json,sys
from pathlib import Path
p=Path(__file__).resolve().parent/'candidates'/sys.argv[1]/'schedule.json'
s=json.loads(p.read_text())
for i in sorted(s['instructions'],key=lambda x:x['lifetime']['begin']):
 print(i['index'],i['tpe'],i['lifetime']['begin'],i['lifetime']['end'],i['lifetime']['end']-i['lifetime']['begin'],i['contexts'],str(i.get('util')))
