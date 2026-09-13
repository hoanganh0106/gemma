#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC_FILE="src/device/sliding/projection.rs"
BACKUP_DIR="backups"
LOG_FILE="KERNEL2_BENCHMARK_LOG.md"
WORKDIR="target/kernel2_bench_sweeps"
RUNS="${KERNEL2_RUNS:-3}"
TIMEOUT="${KERNEL2_TIMEOUT:-70}"

mkdir -p "$WORKDIR"
SESSION_TS="$(date +%Y%m%d_%H%M%S)"
SESSION_DIR="${WORKDIR}/${SESSION_TS}"
mkdir -p "$SESSION_DIR"

declare -a METHODS=(
  "E5_early_partial|kernel2_current_20260912_131556|Early partial add order (p0+p1 then p2+p3) + legacy tiling"
  "E6_tile768|kernel2_e6_tile768_20260912_1351|CHUNK=768 (keep other code unchanged)"
  "E7_swap_dma_main|kernel2_e7_swap_dma_main_20260912_1352|Swap DMA/Main order in output_partial"
  "E8_alt_contract_order|kernel2_e8_alt_contract_order_20260912_1400|Alternate contract call order"
  "E9_lane_seq|kernel2_e9_lane_seq_20260912_1345|contract_lane LaneMode::Sequential"
  "E10_split3015|kernel2_e10_split3015_20260912_1400|vector_narrow_split 30/30/15"
  "E11_mul1_scale|kernel2_e11_mul1_scale_20260912_1404|cast/FMA order variant"
  "E12_streaming_add|kernel2_e12_streaming_add_20260912_1354|Streaming partial add pattern"
  "E13_chunk1536|kernel2_e13_chunk1536_20260912_1356|CHUNK=1536"
  "E14_lane4|kernel2_e14_lane4_20260912_1358|contract_lane m![H % 120], m![1 # 4]"
  "E15_rollback|kernel2_e15_rollback_baseline_20260912_1402|Rollback baseline variant"
)

extract_cycles() {
  local file="$1"
  grep -A1 'sliding_attention_output' "$file" 2>/dev/null \
    | grep 'cycles=' \
    | head -n 1 \
    | sed -E 's/.*cycles=([0-9]+).*/\1/' \
    | tr -d '[:space:]'
}

log_run() {
  local method="$1"
  local run="$2"
  local status="$3"
  local cycles="$4"
  printf '%s;%s;%s;%s\n' "$method" "$run" "$status" "${cycles:-N/A}" >> "$SESSION_DIR/summary.csv"
}

echo "method,run,status,cycles" > "$SESSION_DIR/summary.csv"

for entry in "${METHODS[@]}"; do
  IFS='|' read -r method backup note <<< "$entry"
  backup_path="${BACKUP_DIR}/${backup}"
  if [[ ! -f "$backup_path" ]]; then
    echo "[SKIP] backup missing: $backup_path"
    echo "[SKIP] ${method}: ${note} -- backup missing" >> "$SESSION_DIR/notes.txt"
    continue
  fi

  run_log="$SESSION_DIR/${method}.log"
  method_ok=1
  {
    echo "===== ${method} ====="
    echo "backup: ${backup}"
    echo "note: ${note}"
  } > "$run_log"

  cp "$backup_path" "$SRC_FILE"
  echo "[${method}] reset source from ${backup}"
  rm -rf target/furiosa-opt
  echo "[$method] compile"
  if ! cargo furiosa-opt compile >> "$run_log" 2>&1; then
    log_run "$method" "compile" "compile_error" ""
    echo "  compile failed, see $run_log" | tee -a "$run_log"
    method_ok=0
    log_run "$method" "all" "incomplete" ""
    {
      echo "run log: ${run_log}"
      echo "---"
    } >> "$run_log"
    continue
  fi

  for i in $(seq 1 "$RUNS"); do
    echo "[$method] run ${i}/${RUNS}: rebuild+submit"
    if RNGD_TIMEOUT="$TIMEOUT" ./scripts/rngd_test.sh --no-build >> "$run_log" 2>&1; then
      log_run "$method" "$i" "pass" ""
    else
      # keep going; rngd submit can fail due quota or arena auth, still useful to log
      log_run "$method" "$i" "submit_error" ""
      method_ok=0
    fi
  done
  if [[ "$method_ok" == 1 ]]; then
    log_run "$method" "all" "done" "$(extract_cycles "$run_log")"
  else
    log_run "$method" "all" "incomplete" "$(extract_cycles "$run_log")"
  fi
  {
    echo "run log: ${run_log}"
    echo "---"
  } >> "$run_log"
done

{
  echo ""
  echo "## Kernel2 sweep session ${SESSION_TS}"
  echo ""
  echo "- Created: \`${SESSION_DIR}\`"
  echo "- Runs per method: \`${RUNS}\`"
  echo "- Timeout per job: \`${TIMEOUT}s\`"
  echo "- Script: \`scripts/run_kernel2_methods.sh\`"
  echo ""
  echo "### New method runs (append by hand to this section)"
  echo ""
  cat "$SESSION_DIR/summary.csv"
  echo ""
} | tee -a "$LOG_FILE"

echo "Done. Session files in $SESSION_DIR; summary.csv includes status and cycle extraction."
