#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
for name in C15-ring16-tail240 C16-ring32-tail120; do
  bash ./compile-special.sh "$name"
done
