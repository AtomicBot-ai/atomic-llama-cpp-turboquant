#!/usr/bin/env bash
# AMD backend bench: one llama-bench matrix per KV cache type, plus an optional
# perplexity pass, with the machine/driver state recorded next to the numbers.
# Run it once per backend build (HIP archive, Vulkan archive, upstream HIP) on
# the same box and compare the JSON files with scripts/bench-amd-report.py.
#
# Usage:
#   BIN_DIR=./build/bin LABEL=hip MODELS="a.gguf b.gguf" bash scripts/bench-amd.sh
#
# Env:
#   BIN_DIR    directory with llama-bench (and llama-perplexity)   [./build/bin]
#   LABEL      backend label used in file names, e.g. hip|vulkan|upstream-hip [hip]
#   MODELS     space-separated GGUF paths                          (required)
#   KV_TYPES   KV cache types, K and V always the same type        [f16 q8_0 turbo3 turbo4]
#   DEPTHS     context depths for -d                               [0 8192 32768]
#   PP         prompt sizes for -p                                 [512 4096]
#   TG         generation length for -n                            [128]
#   RUNS       repetitions per test                                [3]
#   NGL        layers on GPU                                       [99]
#   UB         ubatch size                                         [2048]
#   OUT_DIR    where the results go                                [docs/amd/results]
#   PPL_FILE   wikitext-2 raw file; when set, llama-perplexity runs per KV type
#   PPL_CTX    perplexity context                                  [4096]
#   PPL_CHUNKS perplexity chunks                                   [20]
#   EXTRA      extra llama-bench args, e.g. "-sm none -mg 0"

set -uo pipefail

BIN_DIR="${BIN_DIR:-./build/bin}"
LABEL="${LABEL:-hip}"
KV_TYPES="${KV_TYPES:-f16 q8_0 turbo3 turbo4}"
DEPTHS="${DEPTHS:-0 8192 32768}"
PP="${PP:-512 4096}"
TG="${TG:-128}"
RUNS="${RUNS:-3}"
NGL="${NGL:-99}"
UB="${UB:-2048}"
OUT_DIR="${OUT_DIR:-docs/amd/results}"
PPL_CTX="${PPL_CTX:-4096}"
PPL_CHUNKS="${PPL_CHUNKS:-20}"
EXTRA="${EXTRA:-}"

if [[ -z "${MODELS:-}" ]]; then
  echo "error: MODELS is empty (space-separated GGUF paths)" >&2
  exit 1
fi
if [[ ! -x "$BIN_DIR/llama-bench" ]]; then
  echo "error: $BIN_DIR/llama-bench not found" >&2
  exit 1
fi

HOST_TAG="$(hostname -s 2>/dev/null || hostname)"
STAMP="$(date -u +%Y%m%d-%H%M)"
# Double underscores: host names and labels may contain dashes, and
# bench-amd-report.py splits the file name on "__".
PREFIX="$OUT_DIR/${HOST_TAG}__${LABEL}__${STAMP}"
mkdir -p "$OUT_DIR"

# Machine state next to the numbers: without it a result file is not reproducible.
{
  echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "label: $LABEL"
  echo "bin_dir: $BIN_DIR"
  echo "version: $("$BIN_DIR/llama-bench" --version 2>&1 | head -1)"
  echo "kernel: $(uname -srm)"
  echo "kv_types: $KV_TYPES"
  echo "depths: $DEPTHS  pp: $PP  tg: $TG  runs: $RUNS  ngl: $NGL  ub: $UB  extra: $EXTRA"
  echo "env: ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-} HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-} GGML_VK_VISIBLE_DEVICES=${GGML_VK_VISIBLE_DEVICES:-} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-}"
  echo "models:"
  for m in $MODELS; do echo "  - $m ($(du -h "$m" 2>/dev/null | cut -f1))"; done
  echo "--- rocm"
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -E "Name:|Marketing Name|Wavefront Size|Compute Unit" | head -20
  fi
  if [[ -f /opt/rocm/.info/version ]]; then echo "rocm_version: $(cat /opt/rocm/.info/version)"; fi
  echo "--- vulkan"
  if command -v vulkaninfo >/dev/null 2>&1; then
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|driverInfo|apiVersion" | head -12
  fi
  echo "--- devices"
  "$BIN_DIR/llama-bench" --list-devices 2>/dev/null || true
} > "${PREFIX}.env.txt"
echo "wrote ${PREFIX}.env.txt"

MODEL_ARGS=()
for m in $MODELS; do MODEL_ARGS+=(-m "$m"); done

DEPTH_LIST="${DEPTHS// /,}"
PP_LIST="${PP// /,}"

# One invocation per KV type: llama-bench takes the cross product of -ctk and
# -ctv, and only K == V pairs are wanted here.
for kv in $KV_TYPES; do
  out="${PREFIX}__${kv}.json"
  echo "== llama-bench kv=$kv -> $out"
  "$BIN_DIR/llama-bench" "${MODEL_ARGS[@]}" \
    -ctk "$kv" -ctv "$kv" -fa 1 -ngl "$NGL" -ub "$UB" \
    -d "$DEPTH_LIST" -p "$PP_LIST" -n "$TG" -r "$RUNS" \
    $EXTRA -o json > "$out" 2> "${out%.json}.log"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  llama-bench failed (rc=$rc), see ${out%.json}.log" >&2
  fi
done

if [[ -n "${PPL_FILE:-}" ]]; then
  if [[ ! -x "$BIN_DIR/llama-perplexity" ]]; then
    echo "warning: PPL_FILE set but $BIN_DIR/llama-perplexity not found, skipping" >&2
  else
    ppl_out="${PREFIX}.ppl.txt"
    echo "== perplexity ($PPL_FILE, ctx $PPL_CTX, $PPL_CHUNKS chunks) -> $ppl_out"
    : > "$ppl_out"
    for m in $MODELS; do
      for kv in $KV_TYPES; do
        line=$("$BIN_DIR/llama-perplexity" -m "$m" -f "$PPL_FILE" -c "$PPL_CTX" --chunks "$PPL_CHUNKS" \
          -ctk "$kv" -ctv "$kv" -fa 1 -ngl "$NGL" 2>&1 | grep -E "Final estimate" | tail -1)
        echo "$(basename "$m") $kv ${line:-FAILED}" | tee -a "$ppl_out"
      done
    done
  fi
fi

echo "done: ${PREFIX}*"
