#!/usr/bin/env bash
set -euo pipefail
export CARGO_INCREMENTAL=0
export CARGO_TARGET_DIR=/tmp/furiosa-extreme-20260923-target
flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedule-k1-p052.json
