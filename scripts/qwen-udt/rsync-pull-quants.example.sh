#!/usr/bin/env bash
# Example: copy quantized GGUFs from remote CUDA host to local Mac for Metal benches.
#
#   export REMOTE=ubuntu@192.222.54.232
#   export REMOTE_DIR=~/atomic-llama-cpp-turboquant/.scratch/qwen-udt-quants
#   bash scripts/qwen-udt/rsync-pull-quants.example.sh
#
# Adjust REMOTE / REMOTE_DIR to match your layout.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
REMOTE="${REMOTE:-ubuntu@192.222.54.232}"
REMOTE_DIR="${REMOTE_DIR:-~/atomic-llama-cpp-turboquant/.scratch/qwen-udt-quants}"
LOCAL="${LOCAL_DIR:-${ROOT}/.scratch/qwen-udt-quants}"

mkdir -p "$LOCAL"
rsync -avP --progress "${REMOTE}:${REMOTE_DIR}/" "$LOCAL/"
