#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target
for candidate in candidates/C0*; do
  cp "$candidate/output31.rs" work/src/device/sliding/output31.rs
  (cd work && flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule "../$candidate/schedule.json") > "$candidate/compile.log" 2>&1
  result=$?
  echo "$candidate exit=$result"
done
