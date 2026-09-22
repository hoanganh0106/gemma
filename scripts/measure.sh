#!/bin/bash
# Run the kernel test N times and report the best (lowest) cycle count per kernel.
# The on-device span window is noisy at the 20-35% level, so one run decides nothing.
set -u
export PATH=/home/jun/.local/fopt060/bin:/home/jun/.cargo/bin:/usr/local/bin:/usr/bin:/bin
export TUC_PROFILE_LEVEL=info
cd /home/jun/yik/furiosa-opt-gemma4-12B || exit 1
N="${1:-5}"
OUT=/home/jun/.claude/jobs/46bc5c7e/tmp/measure_raw.txt
: > "$OUT"
for run in $(seq 1 "$N"); do
    cargo furiosa-opt test --release --test test_kernels 2>&1 | grep -E "cycles=|FAIL" >> "$OUT"
done
/usr/bin/python3 - "$OUT" <<'PY'
import sys, math
names = ["sliding_project_qkv", "sliding_attention_output", "decoder_feedforward"]
base = [265101, 435551, 3720750]          # our own local baseline
grade = [250514, 404633, 3703473]         # the grading server's baseline, for reference
vals = [[] for _ in names]
fails = 0
i = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if "FAIL" in line:
        fails += 1
    elif line.startswith("cycles="):
        vals[i % 3].append(int(line.split("=")[1]))
        i += 1
if fails:
    print(f"!! {fails} FAILING comparisons -- correctness is a hard gate")
best = []
for n, v, b in zip(names, vals, base):
    if not v:
        continue
    print(f"{n:26} best {min(v):>9,}  median {sorted(v)[len(v)//2]:>9,}  worst {max(v):>9,}  (n={len(v)})")
    best.append(b / min(v))
if len(best) == 3:
    print(f"\nspeedups (best-of): {[round(x, 3) for x in best]}")
    print(f"score = geometric mean = {math.exp(sum(map(math.log, best)) / 3):.3f}")
PY
