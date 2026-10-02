#!/usr/bin/env bash
# Upload release GGUFs to Hugging Face (AtomicChat org).
#
# Usage:
#   bash scripts/qwen-udt/hf-upload-qwen-udt.sh /path/to/local/quants/dir
#
# Repos (create empty model repos + README via the HF UI first if needed):
#   AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF
#   AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF
#
# Requires: `hf` (huggingface_hub>=1.0) or the older `huggingface-cli`.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${1:-${ROOT}/.scratch/qwen-udt-quants}"

if command -v hf >/dev/null 2>&1; then
  HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF=huggingface-cli
else
  echo 'error: neither `hf` nor `huggingface-cli` found' >&2
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "error: not a directory: $DIR" >&2
  exit 1
fi

upload_one() {
  local f="$1"
  local base
  base=$(basename "$f")
  local repo
  if [[ "$base" == Qwen3.6-27B-* ]]; then
    repo="AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF"
  elif [[ "$base" == Qwen3.6-35B-A3B-* ]]; then
    repo="AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF"
  else
    echo "skip: $base"
    return
  fi
  echo "info: upload $base -> $repo"
  "$HF" upload "$repo" "$f" "$base" --repo-type model --commit-message "Add ${base}"
}

shopt -s nullglob
for f in "$DIR"/Qwen3.6-27B-UDT-*.gguf "$DIR"/Qwen3.6-35B-A3B-UDT-*.gguf; do
  [[ -f "$f" ]] || continue
  upload_one "$f"
done
shopt -u nullglob

echo "ok: upload pass complete (large files may take hours)."
