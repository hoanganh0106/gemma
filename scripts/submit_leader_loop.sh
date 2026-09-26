#!/usr/bin/env bash
set -euo pipefail
export PATH="${HOME}/.cargo/bin:${PATH}"
exec python3 "$(dirname "${BASH_SOURCE[0]}")/moa_submit_loop.py" "$@"
