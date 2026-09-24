#!/usr/bin/env bash
set -uo pipefail
base=/mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
candidate="$base/candidates/$1"
cd "$candidate/isolated"
touch src/lib.rs src/ops.rs src/device/sliding/output31.rs
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/tmp/furiosa-extreme-20260923-target
flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule ../schedule.json > ../compile.log 2>&1
