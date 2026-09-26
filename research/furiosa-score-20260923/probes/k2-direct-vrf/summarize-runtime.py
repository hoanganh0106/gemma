import hashlib
import json
import math
from pathlib import Path
import re

probe = Path(__file__).resolve().parent
baseline = probe.parents[1]
candidate_log = (probe / 'arena-78085.log').read_text()
baseline_log = (baseline / 'arena-78017.log').read_text()
manifest = json.loads((probe / 'runtime-manifest.json').read_text())
status = json.loads((probe / 'arena-78085-status.json').read_text())

def parse_log(text):
    kernels = {}
    for section in text.split('==> ')[1:]:
        name = section.splitlines()[0].strip()
        match = re.search(r'median cycles=(\d+) \(of (\d+) runs: (\[[^\]]+\])\)', section)
        assert match, f'Missing median for {name}'
        errors = [line for line in section.splitlines() if line.startswith('[')]
        kernels[name] = {
            'median_cycles': int(match.group(1)),
            'sample_count': int(match.group(2)),
            'cycles': json.loads(match.group(3)),
            'output_checks': errors,
        }
    assert len(kernels) == 3
    return kernels

candidate = parse_log(candidate_log)
parent = parse_log(baseline_log)
checks = [line for item in candidate.values() for line in item['output_checks']]
assert len(checks) == 15 and all('within tol=100.00%' in line and '-> PASS' in line for line in checks)
assert 'all 3 tests passed' in candidate_log
assert status['status'] == 'SUCCEEDED' and status['exit_code'] == 0
for name in ('test_runtime', 'fixtures.safetensors', 'remote_entrypoint.sh'):
    actual = hashlib.sha256((probe / name).read_bytes()).hexdigest()
    assert actual == manifest['hashes'][name], f'Staged {name} changed'
result = {
    'status': status,
    'baseline_job': 78017,
    'baseline_runtime_sha256': hashlib.sha256((baseline / 'test_runtime').read_bytes()).hexdigest(),
    'candidate_runtime_sha256': manifest['hashes']['test_runtime'],
    'fixture_sha256': manifest['hashes']['fixtures.safetensors'],
    'source_sha256': manifest['source_hashes']['src/device/sliding/output31.rs'],
    'output_pass_count': len(checks),
    'total_output_count': 15,
    'kernels': candidate,
    'baseline_kernels': parent,
    'printed_error_lines_identical_to_baseline': all(candidate[name]['output_checks'] == parent[name]['output_checks'] for name in candidate),
    'k2_cycle_delta': candidate['sliding_attention_output']['median_cycles'] - parent['sliding_attention_output']['median_cycles'],
    'k2_cycle_delta_percent': 100 * (candidate['sliding_attention_output']['median_cycles'] / parent['sliding_attention_output']['median_cycles'] - 1),
    'decision': 'CORRECTNESS_PASS_PROMISING_ABBA_NOT_PROMOTED',
}
jobs = {}
for label, job_id, folder in [('A1', 78017, baseline), ('B1', 78085, probe), ('B2', 78108, probe), ('A2', 78115, probe)]:
    log = (folder / f'arena-{job_id}.log').read_text()
    job_status = json.loads((folder / f'arena-{job_id}-status.json').read_text())
    kernels = parse_log(log)
    output_lines = [line for item in kernels.values() for line in item['output_checks']]
    assert len(output_lines) == 15
    assert all('within tol=100.00%' in line and '-> PASS' in line for line in output_lines)
    assert 'all 3 tests passed' in log
    assert job_status['status'] == 'SUCCEEDED' and job_status['exit_code'] == 0
    assert all(kernels[name]['output_checks'] == parent[name]['output_checks'] for name in kernels)
    jobs[label] = {'job_id': job_id, 'status': job_status, 'kernels': kernels, 'output_pass_count': 15}
k2 = 'sliding_attention_output'
r1 = jobs['B1']['kernels'][k2]['median_cycles'] / jobs['A1']['kernels'][k2]['median_cycles']
r2 = jobs['B2']['kernels'][k2]['median_cycles'] / jobs['A2']['kernels'][k2]['median_cycles']
logmean = (math.log(r1) + math.log(r2)) / 2
geometric_ratio = math.exp(logmean)
order = ['A1', 'B1', 'B2', 'A2']
chronological = all(jobs[left]['status']['finished_at'] < jobs[right]['status']['started_at'] for left, right in zip(order, order[1:]))
assert chronological
abba = {
    'jobs': jobs,
    'output_pass_count': 60,
    'total_output_count': 60,
    'chronological_order_verified': chronological,
    'printed_error_lines_identical_across_all_jobs': True,
    'k2_B1_over_A1': r1,
    'k2_B2_over_A2': r2,
    'k2_logmean_ratio': logmean,
    'k2_geometric_ratio': geometric_ratio,
    'k2_cycle_change_percent': 100 * (geometric_ratio - 1),
    'k2_observed_speedup': 1 / geometric_ratio,
    'conditional_score_gain_percent_if_K1_and_K3_fixed': 100 * (geometric_ratio ** (-1/3) - 1),
    'decision': 'CORRECTNESS_PASS_PROMISING_ABBA_NOT_PROMOTED',
    'limitations': ['Two pairs only', 'All controller device_id fields are null', 'No stable performance confidence interval established', 'Printed error statistics do not prove bytewise output equality'],
}
result['abba'] = abba
(probe / 'abba-results.json').write_text(json.dumps(abba, indent=2, ensure_ascii=False) + '\n')
(probe / 'runtime-results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({key: value for key, value in abba.items() if key != 'jobs'}, indent=2))
