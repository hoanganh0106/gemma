#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main
.venv/bin/python research/furiosa-score-20260923/generate-fixture.py > research/furiosa-score-20260923/fixture-generation.log 2>&1
CARGO_INCREMENTAL=0 cargo furiosa-opt build --release --bin test_kernels > research/furiosa-score-20260923/public-build.log 2>&1
cp target/release/test_kernels research/furiosa-score-20260923/test_runtime
sha256sum research/furiosa-score-20260923/test_runtime research/furiosa-score-20260923/fixtures.safetensors > research/furiosa-score-20260923/runtime-sha256.txt
