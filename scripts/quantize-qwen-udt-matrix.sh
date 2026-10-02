#!/usr/bin/env bash
# Batch quantize: 2 models x 4 ftypes x 4 variants (base,v1,v2,v3) = 32 jobs.
# Skip base after validation with QWEN_UDT_SKIP_BASE=1.
#
# Environment: same as scripts/quantize-qwen-udt.sh plus
#   QWEN_UDT_SKIP_BASE   if 1, skip base variant rows
#   QWEN_UDT_LOG_DIR     per-job logs (default: ${ROOT}/.scratch/quant-logs)

set -euo pipefail

unset IMATRIX_FILE BF16_INPUT || true

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${QWEN_UDT_LOG_DIR:-${ROOT}/.scratch/quant-logs}"
TS="$(date +%Y%m%d-%H%M%S)"
MASTER_LOG="${LOG_DIR}/matrix-${TS}.log"
mkdir -p "$LOG_DIR"

MODELS=(27b 35a3b)
FTYPES=(Q3_K_M Q4_K_M Q5_K_M Q6_K)
VARIANTS=(base v1 v2 v3)

if [[ "${QWEN_UDT_SKIP_BASE:-0}" == "1" ]]; then
  VARIANTS=(v1 v2 v3)
fi

echo "info: logging to ${MASTER_LOG}" | tee -a "$MASTER_LOG"

for m in "${MODELS[@]}"; do
  for f in "${FTYPES[@]}"; do
    for v in "${VARIANTS[@]}"; do
      echo "" | tee -a "$MASTER_LOG"
      echo "=== $(date -Iseconds) START ${m} ${f} ${v} ===" | tee -a "$MASTER_LOG"
      t0=$(date +%s)
      if ! "${ROOT}/scripts/quantize-qwen-udt.sh" "$m" "$f" "$v" >>"$MASTER_LOG" 2>&1; then
        echo "FAIL: ${m} ${f} ${v}" | tee -a "$MASTER_LOG"
        exit 1
      fi
      t1=$(date +%s)
      echo "=== DONE ${m} ${f} ${v} in $((t1 - t0))s ===" | tee -a "$MASTER_LOG"
    done
  done
done

echo "info: matrix complete. Master log: ${MASTER_LOG}" | tee -a "$MASTER_LOG"
