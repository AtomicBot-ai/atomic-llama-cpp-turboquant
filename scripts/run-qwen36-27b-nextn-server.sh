#!/usr/bin/env bash
# Qwen3.6 27B target + NextN draft (second load, same GGUF, override_arch) on llama-server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER="${LLAMA_SERVER:-${ROOT}/build/bin/llama-server}"
MAIN="${MAIN_GGUF:-${ROOT}/.scratch/qwen3-6-27b/qwen3-6-27b-q8_0.gguf}"
DRAFT="${DRAFT_GGUF:-$MAIN}"

VERIFY_GGUF="${VERIFY_NEXTN_GGUF:-1}"

CTX="${CTX:-32768}"
NGL="${NGL:-99}"
NGL_DRAFT="${NGL_DRAFT:-99}"
CTK="${CTK:-f16}"
CTV="${CTV:-f16}"
CTKD="${CTKD:-f16}"
CTVD="${CTVD:-f16}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FA="${FA:-on}"
SPEC="${SPEC:-nextn}"

ENABLE_METRICS="${ENABLE_METRICS:-1}"
ENABLE_SLOTS="${ENABLE_SLOTS:-1}"
LOG_TIMESTAMPS="${LOG_TIMESTAMPS:-1}"
LOG_PREFIX="${LOG_PREFIX:-1}"
NO_WARMUP="${NO_WARMUP:-0}"

if [[ ! -f "$SERVER" ]]; then
  echo "error: missing ${SERVER}" >&2
  exit 1
fi
if [[ ! -f "$MAIN" ]]; then
  echo "error: main GGUF not found: ${MAIN}" >&2
  exit 1
fi

if [[ "$SPEC" == "nextn" ]]; then
  if [[ ! -f "$DRAFT" ]]; then
    echo "error: draft path not found: ${DRAFT}" >&2
    exit 1
  fi
  if [[ "$VERIFY_GGUF" != "0" ]]; then
    python3 "${ROOT}/scripts/verify-qwen36-nextn-gguf.py" "$MAIN" || exit 1
  fi
fi

PARALLEL="${PARALLEL:-1}"

ARGS=(
  -m "$MAIN"
  -c "$CTX"
  -ngl "$NGL"
  -ngld "$NGL_DRAFT"
  -ctk "$CTK"
  -ctv "$CTV"
  -ctkd "$CTKD"
  -ctvd "$CTVD"
  -fa "$FA"
  --host "$HOST"
  --port "$PORT"
  --parallel "$PARALLEL"
  -np "$PARALLEL"
  --cont-batching
)

if [[ "$SPEC" == "nextn" ]]; then
  ARGS+=(
    -md "$DRAFT"
    --spec-type nextn
    --spec-draft-n-max "${DRAFT_MAX:-2}"
    --spec-draft-n-min "${DRAFT_MIN:-1}"
  )
else
  echo "info: speculative decoding disabled (SPEC=${SPEC}); running baseline" >&2
fi

[[ "$ENABLE_METRICS"  != "0" ]] && ARGS+=(--metrics)
[[ "$ENABLE_SLOTS"    != "0" ]] && ARGS+=(--slots)
[[ "$LOG_TIMESTAMPS"  != "0" ]] && ARGS+=(--log-timestamps)
[[ "$LOG_PREFIX"      != "0" ]] && ARGS+=(--log-prefix)
[[ "$NO_WARMUP"       != "0" ]] && ARGS+=(--no-warmup)

echo "info: SPEC=${SPEC} CTX=${CTX} NGL=${NGL} FA=${FA} CTK=${CTK} CTKD=${CTKD}" >&2
echo "info: MAIN=${MAIN}" >&2
echo "info: DRAFT=${DRAFT}" >&2
exec "$SERVER" "${ARGS[@]}" "$@"
