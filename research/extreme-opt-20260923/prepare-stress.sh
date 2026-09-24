#!/usr/bin/env bash
set -euo pipefail
cd /mnt/d/Project/furiosa-opt-gemma4-12B-main
.venv/bin/python research/extreme-opt-20260923/prepare_stress.py > research/extreme-opt-20260923/stress-generation.log 2>&1
python3 research/extreme-opt-20260923/campaign.py build research/extreme-opt-20260923/stress-baseline --diagnostic
