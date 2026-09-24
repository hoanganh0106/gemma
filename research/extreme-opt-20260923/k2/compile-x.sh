#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
for name in C18-x-fused-broadcast C19-x-bf16-broadcast; do
  bash ./compile-special.sh "$name"
done
