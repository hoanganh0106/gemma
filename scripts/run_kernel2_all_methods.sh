#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC_FILE="src/device/sliding/projection.rs"
BACKUP_FILE="scripts/kernel2_method_backlog.csv"
WORKDIR="target/kernel2_bench_sweeps"
LOG_FILE="KERNEL2_BENCHMARK_LOG.md"
RUNS="${KERNEL2_RUNS:-3}"
TIMEOUT="${KERNEL2_TIMEOUT:-70}"
BASENAME="kernel2_current_source_before_sweep_$(date +%Y%m%d_%H%M%S)"
ORIG_BACKUP="${WORKDIR}/${BASENAME}"
cleanup() {
  cp "$ORIG_BACKUP" "$SRC_FILE" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "$WORKDIR"
cp "$SRC_FILE" "$ORIG_BACKUP"
SESSION_TS="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="${WORKDIR}/${SESSION_TS}"
mkdir -p "$SESSION_DIR"

echo "method,run,status,cycles,note" > "$SESSION_DIR/summary.csv"

while IFS='|' read -r method note backup; do
  [[ -z "$method" ]] && continue
  [[ "$method" == E* ]] || continue
  method=$(echo "$method" | tr -d '\r')
  note=$(echo "$note" | tr -d '\r')
  backup=$(echo "$backup" | tr -d '\r')

  if [[ "$method" == "E1-CHUNK-2048" ]]; then
    method_id="E1"
  elif [[ "$method" == "E2-FP8-STREAM-DECODE" ]]; then
    method_id="E2"
  elif [[ "$method" == "E2-SCALE-RMS-FUSION" ]]; then
    method_id="E2-FUSION"
  elif [[ "$method" == "E3-F32-PARTIAL-ACCUM" ]]; then
    method_id="E3"
  elif [[ "$method" == "E4-TILE-SWEEP" ]]; then
    method_id="E4"
  elif [[ "$method" == "E5-EARLY-PARTIAL-ADD" ]]; then
    method_id="E5"
  elif [[ "$method" == "E6-CHUNK-768" ]]; then
    method_id="E6"
  elif [[ "$method" == "E7-SWAP-DMA-MAIN" ]]; then
    method_id="E7"
  elif [[ "$method" == "E8-CONTRACT-ORDER-ALT" ]]; then
    method_id="E8"
  elif [[ "$method" == "E9-LANEMODE-SEQUENTIAL" ]]; then
    method_id="E9"
  elif [[ "$method" == "E10-SPLIT-3015" ]]; then
    method_id="E10"
  elif [[ "$method" == "E11-MUL1-SCALE" ]]; then
    method_id="E11"
  elif [[ "$method" == "E12-STREAMING-ADD" ]]; then
    method_id="E12"
  elif [[ "$method" == "E13-CHUNK-1536" ]]; then
    method_id="E13"
  elif [[ "$method" == "E14-LANE-4" ]]; then
    method_id="E14"
  elif [[ "$method" == "E15-ROLLBACK" ]]; then
    method_id="E15"
  else
    method_id="$method"
  fi

  run_log="$SESSION_DIR/${method_id}.log"
  {
    echo "===== ${method} ====="
    echo "note: ${note}"
    echo "backup: ${backup:-<missing>}"
  } > "$run_log"

  if [[ -z "$backup" || "$backup" == "<missing>" ]]; then
    echo "[SKIP] no backup for ${method_id}; mark as manual implementation required." | tee -a "$run_log"
    echo "${method_id},manual,missing_backup,," >> "$SESSION_DIR/summary.csv"
    continue
  fi

  if [[ ! -f "backups/${backup}" ]]; then
    echo "[SKIP] backup file missing: backups/${backup}" | tee -a "$run_log"
    echo "${method_id},manual,backup_not_found,," >> "$SESSION_DIR/summary.csv"
    continue
  fi

  cp "backups/${backup}" "$SRC_FILE"
  rm -rf target/furiosa-opt
  echo "[${method_id}] compile" | tee -a "$run_log"
  if ! cargo furiosa-opt compile >> "$run_log" 2>&1; then
    echo "${method_id},compile,failed,," >> "$SESSION_DIR/summary.csv"
    echo "compile failed, see ${run_log}" | tee -a "$run_log"
    continue
  fi

  method_ok=1
for i in $(seq 1 "$RUNS"); do
    if RNGD_TIMEOUT="$TIMEOUT" ./scripts/rngd_test.sh --no-build >> "$run_log" 2>&1; then
      echo "${method_id},${i},pass," >> "$SESSION_DIR/summary.csv"
    else
      method_ok=0
      echo "${method_id},${i},submit_error,," >> "$SESSION_DIR/summary.csv"
    fi
  done

  if [[ "$method_ok" -eq 0 ]]; then
    echo "${method_id},all,incomplete," >> "$SESSION_DIR/summary.csv"
  else
    cycles=$(grep -A1 'sliding_attention_output' "$run_log" 2>/dev/null \
      | grep 'cycles=' | head -n 1 | sed -E 's/.*cycles=([0-9]+).*/\1/' | tr -d '[:space:]')
    echo "${method_id},all,done,${cycles:-N/A}" >> "$SESSION_DIR/summary.csv"
  fi

  echo "run log: ${run_log}" >> "$run_log"
done < "$BACKUP_FILE"

{
  echo ""
  echo "## Kernel2 sweep session ${SESSION_TS}"
  echo ""
  echo "- Created: \`${SESSION_DIR}\`"
  echo "- Runs per method: \`${RUNS}\`"
  echo "- Timeout per job: \`${TIMEOUT}s\`"
  echo "- Script: \`scripts/run_kernel2_all_methods.sh\`"
  echo ""
  echo "### New method runs (append by hand to this section)"
  echo ""
  cat "$SESSION_DIR/summary.csv"
  echo ""
} | tee -a "$LOG_FILE"

echo "Done. Session files in $SESSION_DIR; see summary.csv"
