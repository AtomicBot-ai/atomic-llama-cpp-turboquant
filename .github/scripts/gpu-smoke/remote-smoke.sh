#!/usr/bin/env bash
# Runs ON the rented GPU box. Smokes one released backend archive end-to-end:
#   download release asset -> quantize f16 -> NVFP4 with the SHIPPED
#   llama-quantize -> llama-server on GPU -> coherence assert -> llama-bench
#   -> assert the GPU backend actually did the work (no silent CPU fallback).
#
# args: $1 = backend id (linux-x64-vulkan | linux-x64-cuda-13.3)
#       $2 = release tag (e.g. dev-latest or b10018-1.0.0)
# Writes /root/smoke.log (progress) and /root/smoke.status (OK/FAIL last line).
set -uo pipefail   # NOT -e: we handle failures explicitly to always write status

BACKEND="${1:?backend id required}"
TAG="${2:-dev-latest}"
REPO="AtomicBot-ai/atomic-llama-cpp-turboquant"
WORK=/root/smoke
BIN="$WORK/bin/build/bin"

fail() { echo "FAIL: $*"; echo "FAIL" > /root/smoke.status; exit 1; }

mkdir -p "$WORK" && cd "$WORK"
export DEBIAN_FRONTEND=noninteractive

echo "== [1/6] runtime deps for $BACKEND =="
apt-get update -q >/dev/null 2>&1 || true
apt-get install -yq curl jq >/dev/null 2>&1 || fail "apt basic deps"
if [ "$BACKEND" = "linux-x64-vulkan" ]; then
  # Lessons from manual runs on vast boxes:
  #  - libGLX_nvidia (the vulkan ICD) silently needs X11 client libs
  #  - the stock jammy vulkan loader (1.3.204) cannot negotiate with the ICD
  #    of current NVIDIA drivers -> take the loader from LunarG
  apt-get install -yq libvulkan1 libxext6 libx11-6 wget gnupg >/dev/null 2>&1 || fail "apt vulkan deps"
  wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - >/dev/null 2>&1
  wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list \
    https://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
  apt-get update -q >/dev/null 2>&1
  apt-get install -yq vulkan-sdk >/dev/null 2>&1 || fail "apt vulkan-sdk (LunarG)"
fi
# CUDA backend: driver comes from the host, cudart/cublas are bundled in the archive.

echo "== [2/6] release asset =="
curl -sfLo bin.tar.gz \
  "https://github.com/$REPO/releases/download/$TAG/llama-turboquant-$BACKEND.tar.gz" \
  || fail "asset download llama-turboquant-$BACKEND.tar.gz @ $TAG"
mkdir -p bin && tar xzf bin.tar.gz -C bin || fail "unpack"
VERSION_LINE=$("$BIN/llama-server" --version 2>&1 | head -1)
echo "version: $VERSION_LINE"
echo "$VERSION_LINE" | grep -q "version:" || fail "llama-server --version"

echo "== [3/6] f16 -> NVFP4 with shipped llama-quantize =="
curl -sfLo m-f16.gguf \
  "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-f16.gguf" \
  || fail "model download"
"$BIN/llama-quantize" m-f16.gguf m-nvfp4.gguf NVFP4 >quant.log 2>&1 || fail "llama-quantize NVFP4"
[ -s m-nvfp4.gguf ] || fail "nvfp4 gguf empty"
ls -lh m-*.gguf

echo "== [4/6] llama-server on GPU =="
"$BIN/llama-server" -m m-nvfp4.gguf --port 8099 --no-webui -ngl 99 >server.log 2>&1 &
SRV=$!
UP=""
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1; then UP=1; break; fi
  kill -0 $SRV 2>/dev/null || break
  sleep 2
done
[ -n "$UP" ] || { tail -20 server.log; fail "server did not become healthy"; }
echo "health: ok"

echo "== [5/6] generation + coherence assert =="
ANSWER=$(curl -sf http://127.0.0.1:8099/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France? Reply with just the city name."}],"max_tokens":20,"temperature":0}' \
  | jq -r '.choices[0].message.content // empty')
echo "answer: $ANSWER"
echo "$ANSWER" | grep -qi paris || { kill $SRV 2>/dev/null; fail "answer lacks 'Paris'"; }
kill $SRV 2>/dev/null; wait $SRV 2>/dev/null

echo "== [6/6] llama-bench + GPU-actually-used assert =="
"$BIN/llama-bench" -m m-nvfp4.gguf -ngl 99 -p 512 -n 128 >bench.log 2>&1 || fail "llama-bench"
sed -n '/| model/,$p' bench.log
case "$BACKEND" in
  linux-x64-vulkan)
    grep -q "load_backend: loaded Vulkan backend" bench.log || fail "Vulkan backend not loaded"
    grep -Eq "ggml_vulkan: 0 = NVIDIA" bench.log || fail "Vulkan device is not the NVIDIA GPU"
    ;;
  linux-x64-cuda-13.3)
    grep -q "load_backend: loaded CUDA backend" bench.log || fail "CUDA backend not loaded"
    grep -Eiq "cuda.*(NVIDIA|GeForce|RTX|H[0-9]+)" bench.log || fail "CUDA device is not an NVIDIA GPU"
    ;;
  *) fail "unknown backend $BACKEND" ;;
esac
TG=$(awk -F'|' '/tg128/ {gsub(/ /,"",$7); print $7}' bench.log | head -1)
echo "tg128: ${TG:-?} t/s"

echo "SMOKE OK: $BACKEND @ $TAG ($VERSION_LINE)"
echo "OK" > /root/smoke.status
