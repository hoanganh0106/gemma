#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/tmp/furiosa-extreme-20260923-target
flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_attention_output --exact --dump-schedule schedule-k2-p043.json
