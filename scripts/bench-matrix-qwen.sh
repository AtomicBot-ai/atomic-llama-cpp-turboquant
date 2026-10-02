#!/usr/bin/env bash
# Matrix bench: Qwen3.6 (27B + 35B A3B) x 4 modes (f16-base, turbo3-base, f16-nextn, turbo3-nextn)
#   x 2 lengths (short / long) x RUNS.
# Requires MTP GGUF for nextn modes (same file for main + draft); base GGUF for baseline modes.
#
# Usage:
#   bash scripts/bench-matrix-qwen.sh
# Env overrides:
#   QWEN27_BASE, QWEN27_MTP, QWEN35_BASE, QWEN35_MTP — GGUF paths
#   QWEN_BENCH_COMBINED_GGUF_ONLY — if 1, set QWEN*_BASE to same path as QWEN*_MTP (only *_MTP.gguf exists)
#   BENCH_MODES_FILTER — comma-separated mode ids (subset of f16-base,turbo3-base,f16-nextn,turbo3-nextn)
#   BENCH_MATRIX_MD — append markdown summary table to this file
#   BENCH_LABEL — optional markdown heading printed before the summary block
#   BENCH_QWEN_MODELS — all (default) | 27 | 35 (run only one row in the matrix)
#   HOST, PORT, SHORT_N, LONG_N, RUNS, CTX (context size for server)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18080}"
HP="${HOST}:${PORT}"

SHORT_N="${SHORT_N:-128}"
LONG_N="${LONG_N:-512}"
RUNS="${RUNS:-3}"
CTX="${CTX:-8192}"
# Speculative knobs (only used for nextn modes). DRAFT_MAX=2 + DRAFT_MIN=1 matches
# the async pipeline depth-2 sweet spot on Apple Silicon Metal (worker overlaps
# one draft decode with one target verify, same as Gemma MTP DRAFT_BLOCK_SIZE=2).
# Override via env to sweep, e.g. DRAFT_MAX=8 bash scripts/bench-matrix-qwen.sh.
DRAFT_MAX="${DRAFT_MAX:-2}"
DRAFT_MIN="${DRAFT_MIN:-1}"

QWEN27_BASE="${QWEN27_BASE:-$ROOT/.scratch/Qwen3.6-27B-UD-Q4_K_XL/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
QWEN27_MTP="${QWEN27_MTP:-$ROOT/.scratch/Qwen3.6-27B-UD-Q4_K_XL_MTP/Qwen3.6-27B-UD-Q4_K_XL_MTP.gguf}"
QWEN35_BASE="${QWEN35_BASE:-$ROOT/.scratch/qwen-3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
QWEN35_MTP="${QWEN35_MTP:-$ROOT/.scratch/Qwen3.6-35B-A3B-UD-Q4_K_XL_MTP/Qwen3.6-35B-A3B-UD-Q4_K_XL_MTP.gguf}"

if [[ "${QWEN_BENCH_COMBINED_GGUF_ONLY:-0}" == "1" ]]; then
  QWEN27_BASE="$QWEN27_MTP"
  QWEN35_BASE="$QWEN35_MTP"
fi

PROMPT='Write a detailed 300-word essay about the history of artificial intelligence, including early pioneers like Alan Turing and John McCarthy, key milestones such as the Dartmouth Conference and the development of expert systems, and future predictions about AGI and superintelligence.'

# model_id | run_script | gguf_baseline (no NextN) | gguf_mtp (NextN draft = same file)
ALL_MODELS=(
  "qwen-27B|${ROOT}/scripts/run-qwen36-27b-nextn-server.sh|${QWEN27_BASE}|${QWEN27_MTP}"
  "qwen-35B-A3B|${ROOT}/scripts/run-qwen36-35ba3b-nextn-server.sh|${QWEN35_BASE}|${QWEN35_MTP}"
)

case "${BENCH_QWEN_MODELS:-all}" in
  all) MODELS=("${ALL_MODELS[@]}") ;;
  27)  MODELS=("${ALL_MODELS[0]}") ;;
  35)  MODELS=("${ALL_MODELS[1]}") ;;
  *)
    echo "error: BENCH_QWEN_MODELS must be all|27|35 (got ${BENCH_QWEN_MODELS})" >&2
    exit 1
    ;;
esac

# mode_id | SPEC (off|nextn) | CTK (propagated to CTV/CTKD/CTVD by run script)
MODES=(
  "f16-base|off|f16"
  "turbo3-base|off|turbo3"
  "f16-nextn|nextn|f16"
  "turbo3-nextn|nextn|turbo3"
)

mode_allowed() {
  local mid="$1"
  [[ -z "${BENCH_MODES_FILTER:-}" ]] && return 0
  [[ ",${BENCH_MODES_FILTER}," == *",${mid},"* ]]
}

SRV_LOG=$(mktemp -t bench-matrix-qwen-srv.XXXXXX.log)

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 1
    kill -9 "$SERVER_PID" 2>/dev/null || true
  fi
  pkill -9 -f "llama-server.*--port ${PORT}" 2>/dev/null || true
  if command -v lsof >/dev/null 2>&1; then
    for p in $(lsof -ti:"${PORT}" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null || true
    done
  fi
  sleep 1
}

trap stop_server EXIT

wait_for_ready() {
  local max_wait=240
  local elapsed=0
  while (( elapsed < max_wait )); do
    if curl -sf "http://${HP}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  echo "  ERROR: server did not become ready in ${max_wait}s" >&2
  return 1
}

resolve_chat_model() {
  CHAT_MODEL="$(
    curl -sf "http://${HP}/v1/models" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    data = d.get('data') or []
    print(data[0]['id'] if data else '')
except Exception:
    print('')
" 2>/dev/null || true
  )"
  export CHAT_MODEL
}

# One measurement via /v1/chat/completions: returns "tps|accept|ct" via stdout.
measure_one() {
  local n_predict="$1"
  PROMPT="${PROMPT}" N_PREDICT="${n_predict}" HP="${HP}" CHAT_MODEL="${CHAT_MODEL:-}" python3 - <<'PY'
import json, os, sys, urllib.request
body = {
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": int(os.environ["N_PREDICT"]),
    "temperature": 0,
    "cache_prompt": False,
    "stream": False,
}
m = os.environ.get("CHAT_MODEL", "").strip()
if m:
    body["model"] = m
req = urllib.request.Request(
    f"http://{os.environ['HP']}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
except Exception:
    print("ERR|0.0|0")
    sys.exit(0)
t = d.get("timings", {})
u = d.get("usage", {}) or {}
ct = u.get("completion_tokens", 0) or 0
tps = t.get("predicted_per_second", 0.0)
dn = t.get("draft_n", 0) or 0
da = t.get("draft_n_accepted", 0) or 0
acc = (100.0 * da / dn) if dn else 0.0
print(f"{tps:.2f}|{acc:.1f}|{ct}")
PY
}

median_tps() {
  python3 - "$@" <<'PY'
import sys, statistics
vals = []
for s in sys.argv[1:]:
    try:
        vals.append(float(s.split("|")[0]))
    except Exception:
        pass
if not vals:
    print("ERR")
else:
    print(f"{statistics.median(vals):.2f}")
PY
}

mean_accept() {
  python3 - "$@" <<'PY'
import sys
vals = []
for s in sys.argv[1:]:
    try:
        v = float(s.split("|")[1])
        vals.append(v)
    except Exception:
        pass
if not vals:
    print("-")
else:
    print(f"{sum(vals)/len(vals):.1f}")
PY
}

declare -A RESULTS

run_cell() {
  local model_id="$1"
  local run_script="$2"
  local gguf_base="$3"
  local gguf_mtp="$4"
  local mode_id="$5"
  local spec="$6"
  local ctk="$7"

  local main_g draft_g verify
  if [[ "$spec" == "nextn" ]]; then
    main_g="$gguf_mtp"
    draft_g="$gguf_mtp"
    verify="${VERIFY_NEXTN_GGUF:-1}"
  else
    main_g="$gguf_base"
    draft_g="$gguf_base"
    verify="0"
  fi

  echo "" >&2
  if [[ "$spec" == "nextn" ]]; then
    echo "=== ${model_id} :: ${mode_id} (SPEC=${spec} CTK=${ctk} DM=${DRAFT_MAX} DN=${DRAFT_MIN} MAIN=$(basename "${main_g}")) ===" >&2
  else
    echo "=== ${model_id} :: ${mode_id} (SPEC=${spec} CTK=${ctk} MAIN=$(basename "${main_g}")) ===" >&2
  fi

  stop_server

  echo "  starting server..." >&2
  SPEC="${spec}" CTK="${ctk}" CTV="${ctk}" CTKD="${ctk}" CTVD="${ctk}" \
    MAIN_GGUF="${main_g}" DRAFT_GGUF="${draft_g}" \
    VERIFY_NEXTN_GGUF="${verify}" \
    CTX="${CTX}" \
    DRAFT_MAX="${DRAFT_MAX}" DRAFT_MIN="${DRAFT_MIN}" \
    NO_WARMUP=1 \
    HOST="${HOST}" PORT="${PORT}" \
    "${run_script}" > "${SRV_LOG}" 2>&1 &
  SERVER_PID=$!

  if ! wait_for_ready; then
    echo "  FAIL: server not ready (last 20 lines):" >&2
    tail -20 "${SRV_LOG}" >&2
    RESULTS["${model_id}|${mode_id}|short"]="FAIL|-"
    RESULTS["${model_id}|${mode_id}|long"]="FAIL|-"
    return 1
  fi

  resolve_chat_model
  if [[ -z "${CHAT_MODEL:-}" ]]; then
    echo "  WARN: could not resolve CHAT_MODEL from /v1/models; requests may fail" >&2
  else
    echo "  CHAT_MODEL=${CHAT_MODEL}" >&2
  fi

  echo "  warmup (n=${LONG_N})..." >&2
  measure_one "${LONG_N}" > /dev/null || true

  for length_id in short long; do
    local n
    n=$([[ "$length_id" == "short" ]] && echo "${SHORT_N}" || echo "${LONG_N}")
    echo "  measuring ${length_id} (n=${n}) x ${RUNS}..." >&2
    local samples=()
    for ((i=1; i<=RUNS; i++)); do
      local s
      s=$(measure_one "${n}") || s="ERR|0|0"
      samples+=("${s}")
      echo "    run $i: ${s}" >&2
    done
    local med acc
    med=$(median_tps "${samples[@]}")
    acc=$(mean_accept "${samples[@]}")
    RESULTS["${model_id}|${mode_id}|${length_id}"]="${med}|${acc}"
  done

  stop_server
}

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r model_id run_script gguf_base gguf_mtp <<< "${model_entry}"
  if [[ ! -x "${run_script}" ]]; then
    echo "skip ${model_id}: ${run_script} not executable" >&2
    continue
  fi
  if [[ ! -f "${gguf_base}" ]]; then
    echo "skip ${model_id}: base GGUF missing: ${gguf_base}" >&2
    continue
  fi
  if [[ ! -f "${gguf_mtp}" ]]; then
    echo "skip ${model_id}: MTP GGUF missing: ${gguf_mtp}" >&2
    continue
  fi
  for mode_entry in "${MODES[@]}"; do
    IFS='|' read -r mode_id spec ctk <<< "${mode_entry}"
    mode_allowed "$mode_id" || continue
    run_cell "${model_id}" "${run_script}" "${gguf_base}" "${gguf_mtp}" "${mode_id}" "${spec}" "${ctk}" || true
  done
done

emit_summary() {
  if [[ -n "${BENCH_LABEL:-}" ]]; then
    echo "${BENCH_LABEL}"
    echo ""
  fi
  echo "## Qwen3.6 bench matrix (median tps over ${RUNS} runs; accept% from draft_n / draft_n_accepted)"
  echo ""
  printf "| model | mode | short tps (n=%d) | long tps (n=%d) | short accept | long accept |\n" "${SHORT_N}" "${LONG_N}"
  echo "|---|---|---:|---:|---:|---:|"
  for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_id _ _ _ <<< "${model_entry}"
    for mode_entry in "${MODES[@]}"; do
      IFS='|' read -r mode_id _ _ <<< "${mode_entry}"
      mode_allowed "$mode_id" || continue
      short_val="${RESULTS["${model_id}|${mode_id}|short"]:-N/A|-}"
      long_val="${RESULTS["${model_id}|${mode_id}|long"]:-N/A|-}"
      short_tps="${short_val%|*}"
      short_acc="${short_val#*|}"
      long_tps="${long_val%|*}"
      long_acc="${long_val#*|}"
      printf "| %s | %s | %s | %s | %s%% | %s%% |\n" \
        "${model_id}" "${mode_id}" "${short_tps}" "${long_tps}" "${short_acc}" "${long_acc}"
    done
  done
  echo ""
  echo "(last server log: ${SRV_LOG})"
}

if [[ -n "${BENCH_MATRIX_MD:-}" ]]; then
  echo ""
  emit_summary | tee -a "$BENCH_MATRIX_MD"
else
  echo ""
  emit_summary
fi
