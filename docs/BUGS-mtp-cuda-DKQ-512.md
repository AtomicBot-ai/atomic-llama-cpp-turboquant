# MTP path on CUDA aborts at fattn.cu:109 (DKQ=512) for Gemma 4 — reproducible on Blackwell sm_120 and Ampere sm_86

## Summary

The MTP speculative-decoding worker crashes with a `GGML_ABORT("fatal error")` in `ggml/src/ggml-cuda/fattn.cu:109` whenever Gemma 4's global-attention layer (`head_dim = 512`) is processed via the FA-MMA kernel. It happens on **both Blackwell (RTX 5060 Ti, sm_12.0) and Ampere (RTX 3060, sm_8.6)** so it's not architecture-specific. Non-MTP inference of the same target is unaffected.

The bug appears specific to the CUDA `fattn-mma` path; the recent `fattn-tile` DKQ=DV=512 fix (`425db5b`) doesn't cover it. The Apple Metal benchmarks in the README (M4 Max, TurboFlash kernel) use a different code path and are unaffected.

## Reproduction

Pinned commit: **`2e81dc5`** (`feature/turboquant-kv-cache`, today's HEAD).

### Build

```
export PATH=/usr/local/cuda-12.8/bin:$PATH
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86;120" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.8 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc) --target llama-server
```

(Builds cleanly. `version: 72 (2e81dc5)`.)

### Models

- Target: `unsloth/gemma-4-E4B-it-GGUF` Q8_K_XL (`gemma-4-E4B-it-UD-Q8_K_XL.gguf`)
- Drafter: `AtomicChat/gemma-4-E4B-it-assistant-GGUF` Q4_K_M (verified clean by `scripts/verify-gemma4-assistant-gguf.py`)

### Serve command (matches `scripts/run-gemma4-e4b-mtp-server.sh` `throughput` preset)

```
build/bin/llama-server \
  -m /path/to/gemma-4-E4B-it-UD-Q8_K_XL.gguf \
  --mtp-head /path/to/gemma-4-E4B-it-assistant.Q4_K_M.gguf \
  --spec-type mtp \
  --draft-block-size 2 --draft-max 6 --draft-min 0 \
  -c 4096 \
  -ngl 99 -ngld 99 \
  -ctk turbo3 -ctv turbo3 -ctkd turbo3 -ctvd turbo3 \
  -fa on \
  --parallel 1 -np 1 --cont-batching \
  --host 0.0.0.0 --port 8002
```

Server starts cleanly:

```
srv    load_model: MTP assistant path '...gemma-4-E4B-it-assistant.Q4_K_M.gguf' (loaded into target model)
slot   load_model: id  0 | task -1 | speculative decoding context initialized
main: model loaded
main: server is listening on http://0.0.0.0:8002
```

### Trigger

Any chat-completion request with non-trivial output is enough. Example payload:

```json
{
  "model": "gemma-4-E4B-it-UD-Q8_K_XL.gguf",
  "messages": [
    {"role": "system", "content": "Output strict JSON: {severity, escalate, confidence, rationale}."},
    {"role": "user", "content": "Finding: prompt-injection attempt. Bot refused. No leak."}
  ],
  "max_tokens": 128,
  "temperature": 0.0
}
```

### Crash

```
slot update_slots: id  3 | task 0 | prompt processing done, n_tokens = 109, batch.n_tokens = 4
slot update_slots: id  3 | task 0 | created context checkpoint 1 of 32 ...
init: embeddings required but some input tokens were not marked as outputs -> overriding
ggml/src/ggml-cuda/fattn.cu:109: fatal error
[backtrace]
  ggml_cuda_flash_attn_ext
  llama_context::graph_compute_mtp
  llama_context::process_ubatch_mtp
  llama_context::decode_mtp_run
  llama_context::mtp_worker_loop
systemd: Main process exited, code=killed, status=6/ABRT
```

The non-MTP path (same target, no `--mtp-head`/`--spec-type`) responds correctly, so the model itself loads and runs fine; only the MTP worker hits the abort.

## Root cause analysis

`fattn.cu:109` is in `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2`:

```cpp
if constexpr (DKQ <= 256) {
    if (use_gqa_opt && gqa_ratio > 1) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }
    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
} else {
    GGML_ABORT("fatal error");   // ← line 109
}
```

Gemma 4 has heterogeneous head dimensions:
- Most layers: `head_dim = 256` → DKQ = 256 → fine
- Global-attention layers: `global_head_dim = 512` → DKQ = 512 → hits the `else` and aborts

The `fattn-tile` DKQ=DV=512 path was fixed in `425db5b`, but the MTP scheduler appears to dispatch to `fattn-mma` (this code path), which has no DKQ=512 implementation and aborts unconditionally.

vLLM 0.20.2rc1 handles the same model by detecting heterogeneous head dims at startup and forcing the TRITON_ATTN backend (log line: *"Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512). Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence."*). atomic appears to have no equivalent dispatch fallback.

## Workarounds attempted (all fail with the same abort)

| Attempt | Result |
| --- | --- |
| `-fa on` (default `auto`) → omitted (`auto`) | abort |
| `-fa on` explicit | abort |
| `-fa off` | abort (FA still required by MTP graph) |
| `CUDA_VISIBLE_DEVICES=1` (3060, sm_8.6) instead of 5060 Ti (sm_12.0) | abort — not arch-specific |
| Reference script flags exactly (`turbo3` KV cache, `--parallel 1`, `--draft-block-size 2 --draft-max 6`) | abort |
| Drafter `Q4_K_M` from your published GGUF (verified by `scripts/verify-gemma4-assistant-gguf.py`) | abort |

Non-MTP serve of the same target on the same hardware works correctly, so this is purely an MTP-path FA-MMA dispatch issue.

## Environment

```
atomic-llama-cpp-turboquant: 2e81dc5 (feature/turboquant-kv-cache, today's HEAD)
CUDA: 12.8.93
GCC: 12.2.0 (Debian 12)
GPUs: NVIDIA GeForce RTX 5060 Ti (sm_12.0, 16 GiB) + NVIDIA GeForce RTX 3060 (sm_8.6, 12 GiB)
Driver: 580.126.18
OS: Linux 6.x (Proxmox LXC, Debian 12 userspace)
```

## Suggested fix direction

Either:
1. Extend `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ=512, DV=512, ...>` template instances (the same way `fattn-tile` was extended in `425db5b`), or
2. Have the MTP graph builder dispatch to `fattn-tile` (which already has DKQ=512 support) instead of `fattn-mma` for Gemma 4's global-attention layers, or
3. Fall back gracefully (CPU FA, or non-FA path) when DKQ > 256 is detected at dispatch time.

Happy to test patches against this repro.

---

(Filed while benchmarking atomic + vLLM + llama-swap for the dev.to Gemma 4 Challenge contest. Will publish full A/B numbers once atomic CUDA MTP is healthy.)
