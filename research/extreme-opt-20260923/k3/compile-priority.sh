#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k3
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/tmp/furiosa-extreme-20260923-target
for name in "$@"; do
  cp "$name/ffn7.rs" work-priority/src/device/shared/ffn7.rs
  (cd work-priority && flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::decoder_feedforward --exact --dump-schedule "../$name/schedule.json") > "$name/compile.log" 2>&1
  code=$?
  echo "$name exit=$code"
  if [ "$code" != 0 ]; then tail -20 "$name/compile.log"; fi
done
