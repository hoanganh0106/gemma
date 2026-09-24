#!/usr/bin/env bash
set -u
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/tmp/furiosa-extreme-20260923-target
ROOT=/mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k1
for name in "$@"; do
 cd "$ROOT/$name"
 flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedule.json > compile.log 2>&1
 echo "$name exit=$?"
done
