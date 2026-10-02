#!/usr/bin/env bash
# Qwen3.6 35B A3B MoE target + NextN draft (same GGUF, second load). Override MAIN_GGUF for your artifact.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MAIN_GGUF="${MAIN_GGUF:-${ROOT}/.scratch/qwen3-6-35b-a3b/qwen3-6-35b-a3b-q8_0.gguf}"
export DRAFT_GGUF="${DRAFT_GGUF:-$MAIN_GGUF}"

exec "${ROOT}/scripts/run-qwen36-27b-nextn-server.sh" "$@"
