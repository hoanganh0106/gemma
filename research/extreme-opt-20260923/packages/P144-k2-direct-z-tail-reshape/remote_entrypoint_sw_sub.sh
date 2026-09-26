#!/bin/sh
set -eu
cd "$(dirname "$0")"
chmod +x ./test_runtime-sw-sub
export FURIOSA_OPT_PROFILE=info
export GEMMA4_FIXTURE=./fixtures.safetensors
exec ./test_runtime-sw-sub
