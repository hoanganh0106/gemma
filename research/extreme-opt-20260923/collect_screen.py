"""Follow one known Arena job and archive its authoritative terminal result."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import campaign

parser = argparse.ArgumentParser()
parser.add_argument('package')
parser.add_argument('job_id', type=int)
args = parser.parse_args()
package = Path(__file__).resolve().parent / 'packages' / args.package
log = package / f'arena-{args.job_id}.log'
with log.open('w') as f:
    subprocess.run(['furiosa-arena', 'logs', '--follow', str(args.job_id)], stdout=f, check=True)
result = subprocess.run(['furiosa-arena', 'status', str(args.job_id)], capture_output=True, text=True, check=True)
status = json.loads(result.stdout)
(package / f'arena-{args.job_id}-status.json').write_text(json.dumps(status, indent=2))
print(json.dumps(status))
assert status['status'] in ['SUCCEEDED', 'FAILED', 'CANCELED'], 'Job still live; poll same job'
campaign.parse(SimpleNamespace(log=str(log)))
