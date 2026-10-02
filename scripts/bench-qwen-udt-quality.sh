#!/usr/bin/env bash
# Run llama-perplexity on one or more GGUFs (WikiText-2 test split by default;
# optional second pass on a small chat-style text file).
#
# Usage:
#   ./scripts/bench-qwen-udt-quality.sh model1.gguf [model2.gguf ...]
# Env:
#   ROOT, LLAMA_PERPLEXITY, WIKI_FILE, CHAT_FILE, PPL_THREADS, PPL_NGL
#
# CHAT_FILE defaults to scripts/qwen-udt/sample-chat-calib.txt when USE_CHAT_CALIB=1.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PPL="${LLAMA_PERPLEXITY:-${ROOT}/build/bin/llama-perplexity}"
WIKI="${WIKI_FILE:-${ROOT}/wikitext-2-raw/wiki.test.raw}"
CHAT="${CHAT_FILE:-${ROOT}/scripts/qwen-udt/sample-chat-calib.txt}"
USE_CHAT="${USE_CHAT_CALIB:-1}"
NGL="${PPL_NGL:-99}"
THREADS="${PPL_THREADS:-8}"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <model.gguf> [more.gguf ...]" >&2
  exit 1
fi

if [[ ! -f "$WIKI" ]]; then
  echo "error: wiki corpus not found: $WIKI" >&2
  echo "hint: run (cd \"$ROOT\" && sh scripts/get-wikitext-2.sh)" >&2
  exit 1
fi
if [[ ! -f "$PPL" ]]; then
  echo "error: llama-perplexity not found: $PPL" >&2
  exit 1
fi

run_ppl_table() {
  local label="$1"
  local datafile="$2"
  echo ""
  echo "## PPL — ${label}"
  echo ""
  echo "| file | tail metric line |"
  echo "|---|---|"
  for gguf in "${@:3}"; do
    if [[ ! -f "$gguf" ]]; then
      echo "| $gguf | MISSING |"
      continue
    fi
    log=$(mktemp -t ppl-qwen-udt.XXXXXX.log)
    if ! "$PPL" -m "$gguf" -f "$datafile" -ngl "$NGL" -t "$THREADS" >"$log" 2>&1; then
      echo "| $(basename "$gguf") | FAIL |"
      rm -f "$log"
      continue
    fi
    tail_line=$(grep -E '[Pp]erplexity|ppl' "$log" | tail -1 || true)
    echo "| $(basename "$gguf") | ${tail_line:-see $log} |"
    rm -f "$log"
  done
}

run_ppl_table "WikiText-2 (wiki.test.raw)" "$WIKI" "$@"

if [[ "$USE_CHAT" == "1" && -f "$CHAT" ]]; then
  run_ppl_table "chat-style calib (${CHAT})" "$CHAT" "$@"
elif [[ "$USE_CHAT" == "1" ]]; then
  echo "" >&2
  echo "info: USE_CHAT_CALIB=1 but CHAT_FILE missing: $CHAT (skip second pass)" >&2
fi
