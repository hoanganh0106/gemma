#!/usr/bin/env bash
set -euo pipefail
if [[ -f "$HOME/.cargo/env" ]]; then
  source "$HOME/.cargo/env"
fi
export CARGO_INCREMENTAL=0
export CARGO_NET_OFFLINE=true
flock /tmp/furiosa-extreme-20260923.lock cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule schedule-k1-p067.json
