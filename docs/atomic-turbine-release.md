# Atomic Turbine fixed-budget server profiles

These are the P11 profiles with targeted runtime evidence. Use the listed GGUF,
backend, memory caps and batch geometry together. Each profile covers one
text slot with greedy sampling. For a split checkpoint, pass the first shard
and keep the remaining shards alongside it.

The budget covers admitted engine allocations and the runtime reserve.
The OS file cache and total process footprint are measured separately.
Use `/metrics` while the server is idle to inspect
`llamacpp:engine_memory_budget_{applied,used,high_water}_bytes` on Metal.
CUDA exposes separate `llamacpp:engine_memory_ram_*` and
`llamacpp:engine_memory_vram_*` gauges. Compare used bytes with the applied
cap. The high-water gauge records the peak managed usage.

## Metal

### Build the server

From the repository root on Apple Silicon with the macOS SDK available,
configure a fresh Release build and build `llama-server`:

```sh
cmake -S . -B build-metal-release \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF -DGGML_CCACHE=OFF
cmake --build build-metal-release --target llama-server -j 6
```

Set `MODEL` to the local GGUF or first shard for the profile and `SLOTS` to a
writable directory. The commands below serve HTTP on loopback.

### Qwen3.8 fixed-budget server

**Qwen3.8 Flash Next GSQ-RCO IQ3_XXS (`qwen4exp`)** on Apple M4 Max.
Use 8 GiB, context 4096, batch 256, microbatch 8 and eight CPU threads.
The selected server profile covers a 3481-token prompt and its repeat,
a short follow-up, and exact logits and slot state against ordinary Metal.
Short completion, non-thinking chat and natural EOS have separate evidence
at the effective context-256 geometry described in the family plan.

```sh
mkdir -p "$SLOTS"
build-metal-release/bin/llama-server -m "$MODEL" -ngl 99 -c 4096 -b 256 -ub 8 \
  -np 1 -t 8 -tb 8 -fit off --engine-budget-mib 8192 --no-warmup \
  --cache-ram 0 --ctx-checkpoints 0 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 8080
```

After `/health` returns `{"status":"ok"}`, send one short request:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say only OK."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"seed":42,"max_tokens":16,"id_slot":0,"cache_prompt":true,"return_tokens":true,"verbose":true}'
```

The fresh server build on M4 Max returned `OK` and EOS (`[3793, 248046]`).
A follow-up carrying that assistant turn and `Repeat your previous answer
exactly.` returned the same IDs. It reused 17 of 37 input tokens from the
live slot. The managed peak was 8,589,668,826 bytes under the 8,589,934,592-byte
cap. This smoke check covers the build and launch instructions above. See
the [model profile](atomic-turbine-qwen38.md) for numerical controls and
the long-request results.

Add `--engine-prefill-window` to opt into the checked long-prefill acceleration.
The opt-in profile covers long text completion, stop, forced EOS, disconnect
during a prefill slice, same-slot recovery, and explicit file-based rollback
and continuation. Cancellation takes effect at the server slice boundary.
On the 3481-token fixture, windows reduced measured TTFT by about 25% and
logical expert reads by 51.5% while preserving exact logits and state.
The follow-up decode experiment measured about 12.7% higher TPS versus
the earlier window implementation on that server sequence. See the
[family plan and steady-state measurements](atomic-turbine-qwen38.md) and
[window lifecycle evidence](atomic-turbine-qwen38-window-p11.json).

The window path requires an eligible long prefill, including at least 320
uncached prompt tokens and successful workspace admission. A short or
ineligible request uses sequential budgeted streaming. If a manual resize
is pending during window execution, the engine completes the current window,
releases its workspace, applies the request at the synchronized boundary, and
uses sequential streaming for the remainder. This resize check uses the
embedding API. A failed window after state mutation makes that context
unusable and requires a fresh context.

### Ornith fixed-budget server

**Ornith 1.5 35B A3B Q4_K/IQ4_XS (Qwen3.5 MoE)** on Apple M4 Max.
Use 5 GiB, context and batch 512, microbatch 32 and eight CPU threads.
Greedy HTTP completion covers first and repeated requests, EOS/stop, SSE
disconnect and reuse, file-based save/erase/restore, impossible budget and
controlled source modification. The profile uses bounded known-route
overlap, prepared expert maps and the budget-admitted route index.
Manual shrink/grow has separate embedding API coverage. See the
[P11 checks and failure handling](atomic-turbine.md#p11-fixed-budget-release-profile).

```sh
mkdir -p "$SLOTS"
build-metal-release/bin/llama-server -m "$MODEL" -ngl 99 -c 512 -b 512 -ub 32 \
  -np 1 -t 8 -tb 8 -fit off --engine-budget-mib 5120 --no-warmup \
  --cache-ram 0 --ctx-checkpoints 0 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 8080
```

LFM2.5-24B on Metal has separate engine qualification at an initial 4 GiB
budget, context 512 and microbatch 32, including manual resize.
Its HTTP profile is qualified on CUDA below. See
[the Metal engine profiles](atomic-turbine-second-architecture.md#qualified-support-profiles-and-limits).

## CUDA

Both selected profiles use Ubuntu, one RTX 4070 Laptop GPU with 8 GiB of
physical VRAM, separate 4096 MiB RAM / 2048 MiB VRAM engine caps, four CPU
threads, microbatch 8 and a single text slot. Use the qualified CUDA build
and matching backend libraries. Select the device explicitly and retain
partial offload.

- **LFM2.5-24B-A2B-AD-Q4_K_M**, 14 GPU layers, context and batch 256.
  Short HTTP completion, stop/EOS, SSE recovery and slot round-trip pass.
  See the [launch command and evidence](atomic-turbine-cuda-server.md#reproduction).
- **Qwen3.8 Flash Next GSQ-RCO IQ3_XXS**, 22 GPU layers, context 4096 and
  batch 128. Short raw completion, stop, SSE recovery and file-based slot
  restore pass. Selected non-thinking chat cases pass with natural EOS,
  including a long two-turn dialogue that reuses 3473 of 3493 input tokens
  from the live slot and matches ordinary CUDA logits and state byte for
  byte. A selected [two-turn thinking dialogue](atomic-turbine-qwen38-cuda-chat.md#two-turns-with-thinking-and-preserved-history)
  preserves reasoning, reuses 93 of 122 tokens and matches ordinary CUDA
  logits and slots byte for byte. A selected [single tool call and result](atomic-turbine-qwen38-cuda-chat.md#single-tool-call-and-result-in-the-same-slot)
  reuses 322 of 349 tokens from the live slot with exact logits and state.
  A 3481-token raw HTTP prompt passes across 28 server batches,
  with identical output IDs and byte-identical full slot files. A fresh
  server restores the long slot and reuses 3484 cached tokens for an exact
  continuation. The long engine comparison separately verifies full logits.
  See the [launch command](atomic-turbine-qwen38-cuda-server.md#reproduction-and-remaining-boundaries),
  [chat evidence](atomic-turbine-qwen38-cuda-chat.md),
  [HTTP prefill](atomic-turbine-qwen38-cuda-http-prefill.md),
  [HTTP restore](atomic-turbine-qwen38-cuda-http-restore.md) and
  [long engine comparison](atomic-turbine-qwen38-cuda-long-prefill.md).
  The [same-slot long series](atomic-turbine-qwen38-cuda-http-series.md)
  passes an exact repeat and a one-token edit with full re-evaluation.

### CUDA disk prefix checkpoints (opt-in)

P11 is complete for this extension to the single-slot Qwen3.8 CUDA profile
above. Add `--engine-prefix-checkpoints "$PREFIX_DIR"` to that profile's
launch command, with `PREFIX_DIR` set to an existing writable directory,
and send requests with `cache_prompt=true`. The server owns the private
checkpoint files for the life of its process. Plan for two full prefix
states and space for a staged write during replacement. Public slot files
remain the interface for saving state across server restarts.

The checked 3481-token repeat reused 3456 tokens and had 4.794 s TTFT with
integrity checking. The selected one-token edit reused 1664 tokens. Missing,
truncated and same-size corrupted files each caused a full prefill in the
same slot. Full sampling logits and complete saved slots matched ordinary
CUDA in the selected comparisons within the 4 GiB RAM / 2 GiB VRAM engine
caps. See [launch and qualification details](atomic-turbine-qwen38-cuda-prefix.md#opt-in-server-qualification)
and [checkpoint integrity](atomic-turbine-qwen38-cuda-prefix.md#checkpoint-integrity).

## Release boundaries

Keep `-fit off`, `--cache-ram 0` and `--ctx-checkpoints 0` for these
profiles. File-based slot actions require `--slots` and a writable
`--slot-save-path`. Budgeted multi-slot operation, automatic checkpoint
rollback, prolonged soak, multimodal inputs and speculative decoding
remain outside the release profiles. Native macOS warning/critical
pressure delivery is pending an isolated test environment. Replay and
manual memory controls have separate engine evidence. CUDA resize
qualification is scoped to the engine at context 2048. The checked
Qwen3.8 CUDA HTTP profile at context 4096 uses fixed caps.

Treat additional GGUFs, devices, context geometry and server workloads as
new qualification cases. Known-route demand overlap is distinct from
next-layer preview. Preview and tree/MTP verification remain experimental
or deferred. The fixed-budget checks require only the relevant cases
from the linked reports.
