#!/usr/bin/env bash
# Phase 2c sanity: reproduce Unsloth UD-Q4_K_XL on 27B with our base mask + imatrix, then compare PPL to their published GGUF.
#
# Prereqs: scripts/qwen-udt/hf-download-sources.sh completed; llama-quantize + llama-perplexity built.
#
# Env:
#   QWEN_UDT_SOURCES_DIR, ROOT, LLAMA_QUANTIZE, LLAMA_PERPLEXITY, WIKI_FILE

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SOURCES="${QWEN_UDT_SOURCES_DIR:-${ROOT}/.scratch/qwen-ud-sources}"
OUT="${SANITY_OUT:-${ROOT}/.scratch/qwen-udt-quants/Qwen3.6-27B-UDT-Q4_K_XL-base_MTP.gguf}"
REF="${SANITY_REF:-${SOURCES}/27b/Qwen3.6-27B-UD-Q4_K_XL.gguf}"
WIKI="${WIKI_FILE:-${ROOT}/wikitext-2-raw/wiki.test.raw}"

if [[ ! -f "$REF" ]]; then
  echo "error: reference GGUF missing: $REF (run scripts/qwen-udt/hf-download-sources.sh)" >&2
  exit 1
fi

echo "info: quantize 27B Q4_K_M base -> $OUT"
QWEN_UDT_SOURCES_DIR="$SOURCES" QWEN_UDT_OUT_DIR="$(dirname "$OUT")" \
  "${ROOT}/scripts/quantize-qwen-udt.sh" 27b Q4_K_M base "$OUT"

if [[ ! -f "$WIKI" ]]; then
  echo "info: fetching wikitext-2..."
  (cd "$ROOT" && sh scripts/get-wikitext-2.sh)
fi

PPL="${LLAMA_PERPLEXITY:-${ROOT}/build/bin/llama-perplexity}"
echo "info: PPL reference (Unsloth UD-Q4_K_XL)"
"$PPL" -m "$REF" -f "$WIKI" -ngl 99 -t 8 2>&1 | tail -5

echo "info: PPL ours (UDT base mask, MTP output)"
"$PPL" -m "$OUT" -f "$WIKI" -ngl 99 -t 8 2>&1 | tail -5

echo "ok: compare the two perplexity lines manually (expect small delta; MTP file vs non-MTP ref may differ slightly)."
