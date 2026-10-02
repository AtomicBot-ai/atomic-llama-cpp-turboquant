#!/usr/bin/env bash
# One-shot bootstrap for Ubuntu CUDA host: deps, optional clone, build quantize + perplexity + server.
#
# Typical usage (already inside a git checkout):
#   cd atomic-llama-cpp-turboquant
#   bash scripts/qwen-udt/remote-bootstrap.sh
#
# Fresh machine (no checkout yet):
#   export REPO_URL=https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant.git
#   export REPO_BRANCH=master
#   export DEST=$HOME/atomic-llama-cpp-turboquant
#   bash remote-bootstrap.sh
#
# Requires: NVIDIA driver + CUDA toolkit matching the driver (nvidia-smi works).

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"

if [[ -f "$(pwd)/CMakeLists.txt" ]]; then
  DEST="$(pwd)"
elif [[ -n "${DEST:-}" && -f "$DEST/CMakeLists.txt" ]]; then
  DEST="$(cd "$DEST" && pwd)"
else
  DEST="${DEST:-$HOME/atomic-llama-cpp-turboquant}"
  if [[ ! -d "$DEST/.git" ]]; then
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$DEST"
  else
    git -C "$DEST" fetch --depth 1 origin "$REPO_BRANCH" || true
    git -C "$DEST" checkout "$REPO_BRANCH" || true
    git -C "$DEST" pull --ff-only || true
  fi
fi

sudo apt-get update
sudo apt-get install -y build-essential cmake git git-lfs python3-venv python3-pip curl ca-certificates pkg-config

cmake -S "$DEST" -B "$DEST/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON

cmake --build "$DEST/build" -j "$(nproc)" --target llama-quantize llama-imatrix llama-perplexity llama-server

python3 -m pip install --user -U "huggingface_hub[cli]"

echo "ok: repo at $DEST ; binaries in $DEST/build/bin/"
ls -la "$DEST/build/bin/llama-quantize" "$DEST/build/bin/llama-perplexity"
