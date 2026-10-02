#!/usr/bin/env bash
# Run llama-perplexity over wikitext-2 + sample-chat for every UDT quant on the remote CUDA box.
#
# Writes one CSV row per quant to .scratch/bench-logs/qwen-udt-ppl-<UTC>.csv.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PPL="${LLAMA_PERPLEXITY:-${ROOT}/build/bin/llama-perplexity}"
QDIR="${QWEN_UDT_QUANT_DIR:-${ROOT}/.scratch/qwen-udt-quants}"
WIKI="${WIKITEXT:-${ROOT}/wikitext-2-raw/wiki.test.raw}"
CHAT="${CHAT_CALIB:-${ROOT}/scripts/qwen-udt/sample-chat-calib.txt}"
NGL="${NGL:-99}"
THREADS="${THREADS:-8}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
OUT="${ROOT}/.scratch/bench-logs/qwen-udt-ppl-${STAMP}.csv"

mkdir -p "$(dirname "$OUT")"
[[ -f "$WIKI" ]] || { echo "error: wikitext not found: $WIKI" >&2; exit 1; }
[[ -x "$PPL" || -f "$PPL" ]] || { echo "error: $PPL not built" >&2; exit 1; }

echo "file,size_bytes,ppl_wikitext2,ppl_wikitext2_err,ppl_chat,ppl_chat_err,seconds" > "$OUT"

run_ppl() {
  local model="$1"
  local corpus="$2"
  "$PPL" -m "$model" -f "$corpus" -ngl "$NGL" -t "$THREADS" 2>&1 \
    | awk '/Final estimate: PPL =/ {print $5","$7}' \
    | head -1
}

for f in "$QDIR"/Qwen3.6-27B-UDT-*.gguf "$QDIR"/Qwen3.6-35B-A3B-UDT-*.gguf; do
  [[ -f "$f" ]] || continue
  base="$(basename "$f")"
  size="$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")"
  echo "info: PPL $base" >&2
  t0=$(date +%s)
  wiki_line=$(run_ppl "$f" "$WIKI")
  chat_line=""
  if [[ -f "$CHAT" ]]; then
    chat_line=$(run_ppl "$f" "$CHAT")
  fi
  t1=$(date +%s)
  wiki_val="${wiki_line%%,*}"
  wiki_err="${wiki_line##*,}"
  chat_val="${chat_line%%,*}"
  chat_err="${chat_line##*,}"
  echo "$base,$size,${wiki_val:-NA},${wiki_err:-NA},${chat_val:-NA},${chat_err:-NA},$((t1-t0))" >> "$OUT"
done

echo "ok: $OUT"
