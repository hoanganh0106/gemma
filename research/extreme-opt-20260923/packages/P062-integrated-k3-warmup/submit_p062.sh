#!/bin/bash
export PATH=/home/hoanganh/.cargo/bin:$PATH
cd "$(dirname "$0")"

BINARY="target/release/test_kernels"
FIXTURE="fixtures.safetensors"
ENTRYPOINT="remote_entrypoint.sh"
JOB_NAME="${RNGD_JOB_NAME:-p062-k3-warmup-a1}"

if [ ! -f "$BINARY" ]; then echo "P062: binary missing, build first" >&2; exit 1; fi
if [ ! -f "$FIXTURE" ]; then echo "P062: fixture missing" >&2; exit 1; fi
if [ ! -f "$ENTRYPOINT" ]; then echo "P062: entrypoint missing" >&2; exit 1; fi

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
cp "$ENTRYPOINT" "$staging/remote_entrypoint.sh"
cp "$BINARY" "$staging/test_runtime"
cp "$FIXTURE" "$staging/fixtures.safetensors"
chmod +x "$staging/remote_entrypoint.sh" "$staging/test_runtime"

echo "==> submitting $JOB_NAME"
furiosa-arena submit \
    "$staging/remote_entrypoint.sh" \
    "$staging/test_runtime" \
    "$staging/fixtures.safetensors" \
    --name "$JOB_NAME" \
    --entrypoint remote_entrypoint.sh \
    --timeout "${RNGD_SUBMIT_TIMEOUT:-70}" 2>&1
