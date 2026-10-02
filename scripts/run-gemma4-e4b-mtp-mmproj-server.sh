#!/usr/bin/env bash
# Gemma 4 E4B + MTP draft + multimodal projector (vision) + TurboQuant3 KV.
# Thin wrapper around run-gemma4-e4b-mtp-server.sh; override MMPROJ_GGUF if your mmproj lives elsewhere.
#
# Behaviour: text-only turns get the full MTP draft speedup; turns that include
# an image chunk fall back to plain target decoding for that turn only and the
# server logs "skipping speculative prime for multimodal prompt". See NEXTN.md §10.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MMPROJ_GGUF="${MMPROJ_GGUF:-${ROOT}/.scratch/gemma-4-e4b/mmproj-F16.gguf}"

if [[ ! -f "$MMPROJ_GGUF" ]]; then
  echo "error: mmproj not found: ${MMPROJ_GGUF}" >&2
  echo "hint: place mmproj-F16.gguf next to your E4B GGUF or export MMPROJ_GGUF=/path/to/mmproj.gguf" >&2
  exit 1
fi

exec env SPEC="${SPEC:-mtp}" \
  bash "${ROOT}/scripts/run-gemma4-e4b-mtp-server.sh" --mmproj "$MMPROJ_GGUF" "$@"
