"""Summarize emitted schedules; these are static estimates, never hardware timing."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

root = Path(__file__).resolve().parent

def union_length(intervals):
    total, end = 0, 0
    for begin, finish in sorted(intervals):
        total += max(0, finish - max(begin, end))
        end = max(end, finish)
    return total

results = []
for path in sorted(root.glob('k?.schedule.json')):
    schedule = json.loads(path.read_text())
    instructions = schedule['instructions']
    makespan = max(i['lifetime']['end'] for i in instructions)
    by_context = defaultdict(list)
    for inst in instructions:
        for context in inst['contexts']:
            by_context[context].append((inst['lifetime']['begin'], inst['lifetime']['end']))
    top = []
    for inst in sorted(instructions, key=lambda x: x['lifetime']['end'] - x['lifetime']['begin'], reverse=True)[:16]:
        top.append({
            'index': inst['index'], 'type': inst['tpe'], 'contexts': inst['contexts'],
            'begin': inst['lifetime']['begin'], 'end': inst['lifetime']['end'],
            'duration': inst['lifetime']['end'] - inst['lifetime']['begin'],
            'util': inst.get('util'), 'description': inst.get('description'),
        })
    row = {
        'kernel': path.name.split('.')[0], 'schedule_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'makespan': makespan, 'instruction_count': len(instructions),
        'context_occupied_cycles': {key: union_length(value) for key, value in by_context.items()},
        'instruction_types': dict(Counter(i['tpe'] for i in instructions)), 'longest_instructions': top,
    }
    results.append(row)

(root / 'schedule-analysis.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
for result in results:
    print(result['kernel'], result['makespan'], result['instruction_count'], result['context_occupied_cycles'])
    for node in result['longest_instructions'][:6]:
        lines = (node['description'] or '').splitlines()
        print(node['index'], node['type'], node['begin'], node['end'], node['duration'], lines[0] if lines else '')
