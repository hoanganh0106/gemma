#!/bin/sh
set -eu
cd "$(dirname "$0")"
chmod +x ./test_runtime
export FURIOSA_OPT_PROFILE=trace
export GEMMA4_FIXTURE=./fixtures.safetensors
exec ./test_runtime
