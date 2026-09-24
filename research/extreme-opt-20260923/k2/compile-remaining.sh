#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/k2
# Existing C03 was launched before the shared full-compiler mutex was adopted.
while kill -0 1530 2>/dev/null; do sleep 2; done
for name in C04-sw-sub C05-inv-sub C06-tail240 C07-tail960 C08-local-scalar; do
  bash ./compile-one.sh "$name"
done
