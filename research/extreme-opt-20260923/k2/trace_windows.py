"""Summarize K2 device spans, without treating nested spans as additive costs."""
import json
import re
import sys
from pathlib import Path

def summarize(path):
    active = False
    spans = []
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith('==>'):
            active = line.strip() == '==> sliding_attention_output'
        if not active:
            continue
        match = re.fullmatch(r'PROFILE_SPAN\s+(\d+)\s+(\d+)\s+(\d+)\s+"([^"]+)"', line)
        if match:
            cluster, begin, end = map(int, match.groups()[:3])
            spans.append(dict(cluster=cluster, begin=begin, end=end, name=match[4]))
        run = re.search(r'\[sliding_attention_output run(\d+)', line)
        if run:
            for task in (s for s in spans if s['name'] == 'Task'):
                inside = [s for s in spans if s['cluster'] == task['cluster']
                          and s['name'] != 'Task' and task['begin'] <= s['begin']
                          and s['end'] <= task['end']]
                dma = max((s for s in inside if s['name'] == 'DMA'),
                          key=lambda s: s['end'] - s['begin'])
                rows.append(dict(run=int(run[1]), cluster=task['cluster'],
                    task_cycles=task['end']-task['begin'],
                    dominant_dma=dma['end']-dma['begin'],
                    before_dma=dma['begin']-task['begin'],
                    after_dma=task['end']-dma['end'],
                    large_dma=[dict(begin=s['begin']-task['begin'],
                        end=s['end']-task['begin'], duration=s['end']-s['begin'],
                        overlapping_tu=[dict(begin=max(s['begin'], t['begin'])-task['begin'],
                            end=min(s['end'], t['end'])-task['begin'],
                            cycles=min(s['end'], t['end'])-max(s['begin'], t['begin']))
                            for t in inside if t['name'] == 'Renegade::TuExec'
                            and min(s['end'], t['end']) > max(s['begin'], t['begin'])])
                        for s in sorted(inside, key=lambda s: s['begin'])
                        if s['name'] == 'DMA' and s['end']-s['begin'] >= 10000],
                    post_dma_spans=[dict(name=s['name'], begin=s['begin']-dma['end'],
                        end=s['end']-dma['end'], duration=s['end']-s['begin'])
                        for s in sorted(inside, key=lambda s: s['begin'])
                        if s['begin'] >= dma['end']]))
            spans = []
    if len(rows) != 6:
        raise ValueError(f'Expected 3 runs x 2 clusters, got {len(rows)}')
    return dict(source=str(path), diagnostic=True, rows=rows)

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        result = summarize(arg)
        Path(arg).with_suffix('.windows.json').write_text(json.dumps(result, indent=2))
        for row in result['rows']:
            print(Path(arg).parent.name, {k:v for k,v in row.items()
                  if k not in ('post_dma_spans', 'large_dma')}, 'large_dma=', row['large_dma'])
