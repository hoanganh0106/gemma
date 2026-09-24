#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2/candidates/C03-sw-vrf-clean/isolated
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/mnt/d/Project/furiosa-opt-gemma4-12B-main/target
flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule ../schedule.json > ../compile.log 2>&1
