#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
for name in C20-x-fused-fixed C21-x-bf16-fixed; do
  bash ./compile-special.sh "$name"
done
