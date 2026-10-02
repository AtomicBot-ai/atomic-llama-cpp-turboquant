#!/usr/bin/env bash
# Quantize Qwen 3.6 27B / 35B-A3B MTP GGUF with Unsloth MTP-aware imatrix + UDT tensor-type masks.
#
# Usage:
#   ./scripts/quantize-qwen-udt.sh <27b|35a3b> <Q3_K_M|Q4_K_M|Q5_K_M|Q6_K> <base|v1|v2|v3> [out.gguf]
#
# Environment:
#   ROOT                 repo root (default: parent of scripts/)
#   LLAMA_QUANTIZE       path to llama-quantize (default: ${ROOT}/build/bin/llama-quantize)
#   QWEN_UDT_SOURCES_DIR directory with BF16 shards + imatrix (see docs/qwen-udt/RUNBOOK.md)
#   QUANT_THREADS        thread count (default: nproc)
#   QWEN_UDT_OUT_DIR     output directory (default: ${ROOT}/.scratch/qwen-udt-quants)
#
# Output filename (when out.gguf omitted):
#   Qwen3.6-27B-UDT-Q{3,4,5,6}_K_XL[-V1|-V2|-base]_MTP.gguf
#   Qwen3.6-35B-A3B-UDT-Q{3,4,5,6}_K_XL[-V1|-V2|-base]_MTP.gguf
#   v3 (release) omits the -V3 suffix.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QUANT="${LLAMA_QUANTIZE:-${ROOT}/build/bin/llama-quantize}"
SOURCES="${QWEN_UDT_SOURCES_DIR:-${ROOT}/.scratch/qwen-ud-sources}"
OUT_DIR="${QWEN_UDT_OUT_DIR:-${ROOT}/.scratch/qwen-udt-quants}"
THREADS="${QUANT_THREADS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)}"

usage() {
  echo "usage: $0 <27b|35a3b> <Q3_K_M|Q4_K_M|Q5_K_M|Q6_K> <base|v1|v2|v3> [out.gguf]" >&2
  exit 1
}

[[ $# -ge 3 ]] || usage

MODEL="$1"
FTYPE="$2"
VARIANT="$3"
CUSTOM_OUT="${4:-}"

case "$MODEL" in
  27b)
    PREFIX="Qwen3.6-27B"
    SUB="${QWEN_UDT_SUBDIR_27:-27b}"
    IMT="${IMATRIX_FILE:-${SOURCES}/${SUB}/imatrix_unsloth.gguf_file}"
    INP="${BF16_INPUT:-}"
    if [[ -z "$INP" ]]; then
      shopt -s nullglob
      cand=( "${SOURCES}/${SUB}/BF16"/Qwen3.6-27B-BF16-*.gguf "${SOURCES}/${SUB}"/Qwen3.6-27B-BF16-*.gguf )
      shopt -u nullglob
      INP="${cand[0]:-}"
    fi
    ;;
  35a3b)
    PREFIX="Qwen3.6-35B-A3B"
    SUB="${QWEN_UDT_SUBDIR_35:-35a3b}"
    IMT="${IMATRIX_FILE:-${SOURCES}/${SUB}/imatrix_unsloth.gguf_file}"
    INP="${BF16_INPUT:-}"
    if [[ -z "$INP" ]]; then
      shopt -s nullglob
      cand=( "${SOURCES}/${SUB}/BF16"/Qwen3.6-35B-A3B-BF16-*.gguf "${SOURCES}/${SUB}"/Qwen3.6-35B-A3B-BF16-*.gguf )
      shopt -u nullglob
      INP="${cand[0]:-}"
    fi
    ;;
  *)
    usage
    ;;
esac

case "$FTYPE" in
  Q3_K_M|Q4_K_M|Q5_K_M|Q6_K|Q8_0) ;;
  *) echo "error: unsupported ftype '$FTYPE' (expected Q3_K_M|Q4_K_M|Q5_K_M|Q6_K|Q8_0)" >&2; exit 1 ;;
esac

case "$FTYPE" in
  Q3_K_M|Q4_K_M|Q5_K_M) XL_TAG="${FTYPE/_K_M/_K_XL}" ;;
  Q6_K) XL_TAG="Q6_K" ;;
  Q8_0) XL_TAG="Q8_K_XL" ;;
esac
case "$VARIANT" in
  base) SUFFIX="-base" ;;
  v1)   SUFFIX="-V1" ;;
  v2)   SUFFIX="-V2" ;;
  v3)   SUFFIX="" ;;
  *)    echo "error: unknown variant '$VARIANT'" >&2; exit 1 ;;
esac

case "$VARIANT" in
  base) MASK="${ROOT}/scripts/quantize-masks/qwen36-ud-base.txt" ;;
  v1)   MASK="${ROOT}/scripts/quantize-masks/qwen36-ud-v1-nextn.txt" ;;
  v2)   MASK="${ROOT}/scripts/quantize-masks/qwen36-ud-v2-turbo3.txt" ;;
  v3)   MASK="${ROOT}/scripts/quantize-masks/qwen36-ud-v3-combined.txt" ;;
esac

if [[ ! -f "$MASK" ]]; then
  echo "error: mask file not found: $MASK" >&2
  exit 1
fi
if [[ ! -f "$IMT" ]]; then
  echo "error: imatrix not found: $IMT (set IMATRIX_FILE or QWEN_UDT_SOURCES_DIR)" >&2
  exit 1
fi
if [[ ! -f "$INP" ]]; then
  echo "error: BF16 input shard not found: $INP" >&2
  echo "hint: download Unsloth BF16 shards; use first shard *-00001-of-*.gguf as BF16_INPUT" >&2
  exit 1
fi
if [[ ! -x "$QUANT" && ! -f "$QUANT" ]]; then
  echo "error: llama-quantize not found: $QUANT" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
if [[ -n "$CUSTOM_OUT" ]]; then
  OUT="$CUSTOM_OUT"
else
  OUT="${OUT_DIR}/${PREFIX}-UDT-${XL_TAG}${SUFFIX}_MTP.gguf"
fi

echo "info: quantize model=${MODEL} ftype=${FTYPE} variant=${VARIANT}" >&2
echo "info: in=${INP}" >&2
echo "info: out=${OUT}" >&2
echo "info: imatrix=${IMT}" >&2
echo "info: mask=${MASK}" >&2

exec "$QUANT" \
  --imatrix "$IMT" \
  --tensor-type-file "$MASK" \
  "$INP" "$OUT" "$FTYPE" "$THREADS"
