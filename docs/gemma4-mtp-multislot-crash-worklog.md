# Gemma4 A4B MTP Multislot Crash Worklog

## Scope

Goal: fix the deterministic Gemma4 A4B MTP crash when `llama-server` runs with multiple slots.

## Minimal Reproducer

Patched branch was reproduced first with a smaller setup to reduce iteration time:

```bash
CTX=4096 \
PARALLEL=2 \
BATCH=128 \
UBATCH=64 \
SPLIT_MODE=layer \
KV_K=turbo4 \
KV_V=turbo4 \
REASONING_BUDGET=1024 \
ENABLE_MTP=1 \
PORT=8084 \
NO_WARMUP=1 \
~/scripts/local-opencode-llama/scripts/run-gemma4-26b-a4b-mtp.sh
```

Request:

```bash
curl -sS -H 'Content-Type: application/json' \
  http://127.0.0.1:8084/v1/messages \
  -d '{
    "model": "gemma4-26b-a4b-mtp",
    "max_tokens": 16,
    "messages": [
      {"role": "user", "content": "hi"}
    ]
  }'
```

Before the fix this returned `curl: (52) Empty reply from server`.

## Failing Backtrace

The failure reproduced deterministically on the first `/v1/messages` request:

```text
slot get_availabl: id  1 | task -1 | selected slot by LRU, t_last = -1
...
/home/tdeburca/git/model-learning/atomic-llama-cpp-turboquant/ggml/src/ggml.c:3665: GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
...
#6  ggml_reshape_3d
#7  llm_build_gemma4_mtp::llm_build_gemma4_mtp(...)
#8  llama_model::build_graph(...)
#9  llama_context::ensure_sched_mtp()
#10 llama_context::decode_mtp_async(...)
#11 common_speculative_state_mtp::draft(...)
#12 common_speculative_draft(...)
#13 server_context_impl::update_slots()
```

## Root Cause

The crash was in the MTP scheduler reserve path, not in the real worker compute path.

`llama_context::ensure_sched_mtp()` reserved the MTP graph with:

- a single-token `ubatch`
- but a **full KV context** from `memory->init_full()`

For `PARALLEL=2`, the full KV context builds dummy `slot_info` spanning **all streams**:

- `src/llama-kv-cache.cpp`, `llama_kv_cache_context(llama_kv_cache * kv)`
- `s0 = 0`
- `s1 = n_stream - 1`

That changes the reserve graph topology from the single-stream shape expected by Gemma4 MTP into a multi-stream attention shape. The Gemma4 MTP builder later performs single-stream reshapes in `src/models/gemma4-assistant.cpp`, which then trip the `ggml_reshape_3d()` element-count assert during reserve.

This is why:

- `PARALLEL=1 ENABLE_MTP=1` worked
- `PARALLEL=2 ENABLE_MTP=0` worked
- `PARALLEL=2 ENABLE_MTP=1` crashed

The issue is fundamentally the reserve context shape, not prompt size or memory pressure.

## Patch

Initial fix:

- stop using `memory->init_full()` for MTP reserve
- reserve against a single-sequence / single-stream MTP topology instead

Follow-up hardening:

- added a dedicated reserve-only API on `llama_kv_cache_iswa`:
  - `init_mtp_reserve(llama_ubatch ubatch)`
- updated `llama_context::ensure_sched_mtp()` to use `kv_iswa->init_mtp_reserve(ub)`
- kept real decode on `kv_iswa->init_mtp(seq_id, ub)`

The reserve helper constructs a shape-only MTP memory context:

- one stream
- one index
- one ubatch

It does not depend on user `seq_id 0` existing or having KV state, so reserve no
longer borrows real decode semantics just to obtain the correct graph shape.

## Files Changed

- `src/llama-context.cpp`
- `src/llama-kv-cache-iswa.h`
- `src/llama-kv-cache-iswa.cpp`
- `src/llama-kv-cache.h`
- `docs/gemma4-mtp-multislot-crash-worklog.md`

## Validation

Build:

```bash
cmake --build build-hip-rocwmma --target llama-server -j "$(nproc)"
```

Validated combinations:

1. `PARALLEL=1 ENABLE_MTP=1 KV_K=turbo4 KV_V=turbo4`
   - `/v1/messages` returned `200`
   - generation completed

2. `PARALLEL=2 ENABLE_MTP=0 KV_K=turbo4 KV_V=turbo4`
   - `/v1/messages` returned `200`
   - generation completed

3. `PARALLEL=2 ENABLE_MTP=1 KV_K=turbo4 KV_V=turbo4`
   - tiny `/v1/messages` request returned `200`
   - normal Claude-style `/v1/messages` request with `system` + `messages` returned `200`
   - generation completed
   - no crash

4. Spot check: `PARALLEL=2 ENABLE_MTP=1 KV_K=f16 KV_V=f16`
   - `/v1/messages` returned `200`
   - generation completed

Metrics check with MTP enabled:

```bash
curl -sS http://127.0.0.1:8084/metrics | rg 'speculative|draft'
```

Observed Prometheus speculative draft metrics including:

- `llamacpp:speculative_drafts_generated_total{spec_type="mtp"}`
- `llamacpp:speculative_drafts_accepted_total{spec_type="mtp"}`
- `llamacpp:speculative_draft_tokens_generated_total{spec_type="mtp"}`
- `llamacpp:speculative_draft_tokens_accepted_total{spec_type="mtp"}`

## Result

`PARALLEL=2 ENABLE_MTP=1` now works and generates normally.

Reserve-time MTP setup no longer uses `seq_id 0` as a placeholder decode path.
It now uses a dedicated shape-only memory context.

## Remaining Risks

- I did not change non-Gemma speculative paths.
