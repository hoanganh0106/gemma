#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
while kill -0 1658 2>/dev/null; do sleep 2; done
for name in C09-tail120 C10-tail1920 C11-sum-norm C12-direct-reciprocal; do
  bash ./compile-one.sh "$name"
done
