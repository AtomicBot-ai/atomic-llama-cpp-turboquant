#!/usr/bin/env bash
# Download BF16 shards + imatrix + reference UD-Q4_K_XL.gguf from Unsloth HF repos.
#
# Usage:
#   bash scripts/qwen-udt/hf-download-sources.sh [DEST_DIR]
#
# Default DEST_DIR: ${REPO}/.scratch/qwen-ud-sources with per-model subdirs 27b/ and 35a3b/
#
# Requires: `hf` (huggingface_hub>=1.0) or the older `huggingface-cli`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-${ROOT}/.scratch/qwen-ud-sources}"

if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  echo 'error: neither `hf` nor `huggingface-cli` found (pip install -U "huggingface_hub[cli]")' >&2
  exit 1
fi

mkdir -p "$DEST/27b" "$DEST/35a3b"

dl() {
  local repo="$1"; shift
  local local_dir="$1"; shift
  "$HF" download "$repo" "$@" --local-dir "$local_dir"
}

echo "info: 27B — imatrix..."
dl unsloth/Qwen3.6-27B-MTP-GGUF "$DEST/27b" imatrix_unsloth.gguf_file
echo "info: 27B — reference UD-Q4_K_XL..."
dl unsloth/Qwen3.6-27B-MTP-GGUF "$DEST/27b" Qwen3.6-27B-UD-Q4_K_XL.gguf
echo "info: 27B — BF16 shards..."
dl unsloth/Qwen3.6-27B-MTP-GGUF "$DEST/27b" --include "BF16/*"

echo "info: 35B-A3B — imatrix..."
dl unsloth/Qwen3.6-35B-A3B-MTP-GGUF "$DEST/35a3b" imatrix_unsloth.gguf_file
echo "info: 35B-A3B — reference UD-Q4_K_XL..."
dl unsloth/Qwen3.6-35B-A3B-MTP-GGUF "$DEST/35a3b" Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
echo "info: 35B-A3B — BF16 shards..."
dl unsloth/Qwen3.6-35B-A3B-MTP-GGUF "$DEST/35a3b" --include "BF16/*"

echo "ok: sources under $DEST/{27b,35a3b}"
