cltk#!/usr/bin/env bash
# Cold-start sanity check: 27B turbo3-base short n=128 x3 (no NextN).
# Compares against historical baseline of ~18.4 TPS to detect thermal vs code regression.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${PORT:-18080}"
HP="127.0.0.1:${PORT}"
N="${N:-128}"
RUNS="${RUNS:-3}"
MAIN="${MAIN:-$ROOT/.scratch/Qwen3.6-27B-UD-Q4_K_XL/Qwen3.6-27B-UD-Q4_K_XL.gguf}"

PROMPT='Write a detailed 300-word essay about the history of artificial intelligence, including early pioneers like Alan Turing and John McCarthy, key milestones such as the Dartmouth Conference and the development of expert systems, and future predictions about AGI and superintelligence.'

pkill -9 -f llama-server 2>/dev/null || true
sleep 1

SRV_LOG=$(mktemp -t sanity-srv.XXXX.log)
"$ROOT/build/bin/llama-server" \
  -m "$MAIN" -c 8192 -ngl 99 -ctk turbo3 -ctv turbo3 -fa on \
  --host 127.0.0.1 --port "$PORT" --parallel 1 -np 1 --cont-batching \
  --metrics --slots --no-warmup \
  >"$SRV_LOG" 2>&1 &
SRV_PID=$!

echo "info: server pid=$SRV_PID log=$SRV_LOG" >&2

for i in $(seq 1 60); do
  if curl -fsS "http://${HP}/health" >/dev/null 2>&1; then
    echo "info: server ready after ${i}s" >&2
    break
  fi
  sleep 1
done

echo "info: warmup n=512..." >&2
curl -fsS -X POST "http://${HP}/completion" -H 'Content-Type: application/json' \
  -d "$(jq -n --arg p "$PROMPT" --argjson n 512 '{prompt:$p,n_predict:$n,temperature:0,cache_prompt:false}')" \
  | jq -r '.timings | "warmup: \(.predicted_per_second // 0)|\(.predicted_n // 0)"' || true

echo "info: measuring short n=${N} x${RUNS}..." >&2
for i in $(seq 1 "$RUNS"); do
  RESP=$(curl -fsS -X POST "http://${HP}/completion" -H 'Content-Type: application/json' \
    -d "$(jq -n --arg p "$PROMPT" --argjson n "$N" '{prompt:$p,n_predict:$n,temperature:0,cache_prompt:false}')")
  TPS=$(echo "$RESP" | jq -r '.timings.predicted_per_second // 0')
  PRED=$(echo "$RESP" | jq -r '.timings.predicted_n // 0')
  echo "  run $i: ${TPS}|${PRED}"
done

kill "$SRV_PID" 2>/dev/null || true
sleep 1
kill -9 "$SRV_PID" 2>/dev/null || true
pkill -9 -f llama-server 2>/dev/null || true
echo "info: done. server log: $SRV_LOG" >&2
