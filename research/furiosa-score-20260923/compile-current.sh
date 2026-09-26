#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main
export CARGO_INCREMENTAL=0
for spec in k1:sliding_project_qkv k2:sliding_attention_output k3:decoder_feedforward; do
  key=${spec%%:*}
  op=${spec#*:}
  cargo furiosa-opt compile "ops::$op" --exact --dump-schedule "research/furiosa-score-20260923/$key.schedule.json" > "research/furiosa-score-20260923/$key.compile.log" 2>&1
done
