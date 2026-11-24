#!/bin/bash
# Inkling 2 Mini long-input serving profile (Apple Silicon, non-budget path).
#
# See docs/inkling-long-input-profile.md for the measured results and the
# rejected alternatives. Budgeted profiles (--engine-budget-mib) cannot serve
# long inputs: the engine requires ctx <= 4096 and rejects mmproj.
#
# Usage: profiles/inkling-long-input.sh [start|stop|serve]
#   serve = foreground (launchd), start = background, stop = kill
# Env overrides: INKLING_PORT (8081), INKLING_CTX (131072),
#   INKLING_DRAFT_N_MAX (3), INKLING_MLOCK (1), INKLING_MODELS (dir)
set -uo pipefail

ROOT="${ATOMIC_ROOT:-$HOME/code/atomic-engine}"
BIN="$ROOT/build-metal-release/bin/llama-server"
MODELS="${INKLING_MODELS:-$HOME/models/atomicchat}"
MODEL="$MODELS/Inkling-2-Mini-AD-IQ2_M.gguf"
DRAFT="$MODELS/dspark-Inkling-2-Mini-Q8_0.gguf"
MMPROJ="$MODELS/mmproj-Inkling-2-Mini-F16.gguf"

PORT="${INKLING_PORT:-8081}"
CTX="${INKLING_CTX:-131072}"
DRAFT_N_MAX="${INKLING_DRAFT_N_MAX:-3}"
MLOCK="${INKLING_MLOCK:-1}"

for f in "$BIN" "$MODEL" "$DRAFT"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done

ARGS=(
  --model "$MODEL"
  --model-draft "$DRAFT"
  --spec-type draft-dspark --spec-draft-n-max "$DRAFT_N_MAX"
  --n-gpu-layers-draft 99
  --ctx-size "$CTX" --parallel 1 --n-gpu-layers all
  --flash-attn on
  --cache-type-k q4_0 --cache-type-v q4_0
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0
  --batch-size 1024 --ubatch-size 1024
  --threads 8 --threads-batch 8
  --temp 0
  # thinking off (template switch is reasoning_effort, not --reasoning)
  --chat-template-kwargs '{"reasoning_effort":"none"}'
  --no-reasoning-preserve
  --metrics --jinja --alias inkling-2-mini
  --host 127.0.0.1 --port "$PORT"
)
[ -f "$MMPROJ" ] && ARGS+=(--mmproj "$MMPROJ")
[ "$MLOCK" = "1" ] && ARGS+=(--load-mode mmap+mlock)

stop_server() {
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    kill $(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t) 2>/dev/null
    sleep 2
    echo "stopped port $PORT"
  else
    echo "not listening on $PORT"
  fi
}

case "${1:-start}" in
  serve) exec "$BIN" "${ARGS[@]}" ;;
  start)
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && { echo "port $PORT in use"; exit 0; }
    nohup "$BIN" "${ARGS[@]}" > "$HOME/.hermes/logs/inkling-atomic.log" 2>&1 &
    echo "started pid $! on port $PORT (ctx $CTX, draft $DRAFT_N_MAX, mlock $MLOCK)"
    ;;
  stop) stop_server ;;
  *) echo "usage: $0 [start|stop|serve]" >&2; exit 2 ;;
esac
