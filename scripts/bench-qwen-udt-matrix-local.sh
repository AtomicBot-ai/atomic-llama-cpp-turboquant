#!/usr/bin/env bash
# Run scripts/bench-matrix-qwen.sh for each UDT GGUF in a directory (combined MTP files).
#
# Usage:
#   QWEN_UDT_BENCH_DIR=/path/to/quants ./scripts/bench-qwen-udt-matrix-local.sh
# Env:
#   QWEN_UDT_BENCH_DIR   directory containing *.gguf (default: ${ROOT}/.scratch/qwen-udt-quants)
#   BENCH_MATRIX_MD      append all summaries to this markdown file
#   QWEN_UDT_BENCH_TAG   optional filter substring (e.g. "-V1" or "Q4_K_XL")
#   QWEN_UDT_ABLATION_AUTO  if 1 (default), restrict BENCH_MODES_FILTER for -V1 / -V2 filenames

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="${QWEN_UDT_BENCH_DIR:-${ROOT}/.scratch/qwen-udt-quants}"
TAG="${QWEN_UDT_BENCH_TAG:-}"
OUT_MD="${BENCH_MATRIX_MD:-}"
ABL_AUTO="${QWEN_UDT_ABLATION_AUTO:-1}"

if [[ ! -d "$DIR" ]]; then
  echo "error: directory not found: $DIR" >&2
  exit 1
fi

shopt -s nullglob
mapfile -t FILES < <(find "$DIR" -maxdepth 1 -name '*.gguf' -print | sort)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "error: no .gguf under $DIR" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  base=$(basename "$f")
  [[ -n "$TAG" && "$base" != *"$TAG"* ]] && continue
  echo "" >&2
  echo "===== bench: $base =====" >&2
  if [[ "$base" == *"35B-A3B"* ]] || [[ "$base" == *"35B-A3b"* ]]; then
    kind=35
  elif [[ "$base" == *"27B"* ]]; then
    kind=27
  else
    echo "skip (unknown model prefix): $base" >&2
    continue
  fi
  (
    if [[ "$kind" == "35" ]]; then
      export BENCH_QWEN_MODELS=35
      export QWEN35_MTP="$f" QWEN35_BASE="$f"
    else
      export BENCH_QWEN_MODELS=27
      export QWEN27_MTP="$f" QWEN27_BASE="$f"
    fi
    export QWEN_BENCH_COMBINED_GGUF_ONLY=1
    export BENCH_LABEL="### UDT bench: ${base}"
    if [[ -n "$OUT_MD" ]]; then
      export BENCH_MATRIX_MD="$OUT_MD"
    fi
    if [[ "$ABL_AUTO" == "1" ]]; then
      if [[ "$base" == *"-V1"* ]]; then
        export BENCH_MODES_FILTER="f16-nextn,turbo3-nextn"
      elif [[ "$base" == *"-V2"* ]]; then
        export BENCH_MODES_FILTER="turbo3-base,turbo3-nextn"
      else
        unset BENCH_MODES_FILTER || true
      fi
    fi
    bash "${ROOT}/scripts/bench-matrix-qwen.sh"
  )
done
