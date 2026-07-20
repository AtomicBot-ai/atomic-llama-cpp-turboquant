# Merge notes: `pr-inkling` (upstream PR #25731 — TML Inkling architecture) → `inkling-support`

Merge commit: `1d2438f66` on `inkling-support`. 924 files changed; 22 files had
conflicts. The PR head also carried months of upstream master churn, so several
conflicts were plain fork-vs-master divergence unrelated to Inkling.

Resolution priority applied (in order):
1. Inkling-architecture correctness wins where both sides touch the same logic.
2. TurboQuant (turbo2/3/4 KV cache, TQ3_1S/TQ4_1S weights, TURBO_WHT, InnerQ) and
   MTP/NextN preserved for existing architectures.
3. TurboQuant KV cache is NOT made to work with inkling — it is gated off (see below).
4. Banded flash-attention and TurboQuant rotated-cache remain separate code paths.

## The inkling-vs-TurboQuant gate

`src/llama-context.cpp`, in `llama_init_from_model` right after the Grok
flash-attn check: when `model->arch == LLM_ARCH_INKLING` and `type_k`/`type_v`
is a turbo type, the context **falls back to the standard f16 KV cache with a
`LLAMA_LOG_WARN` line** (silent fallback, not an error). Inkling attention runs
through `GGML_OP_FLASH_ATTN_EXT_BANDED` (content-dependent relative position
bias), which the WHT-rotated turbo cache path does not implement. All other
architectures keep full turbo cache support.

## Cross-file enum/id decisions (canonical)

- `GGML_TYPE`: TURBO2_0=42, TURBO3_0=43, TURBO4_0=44, TQ3_1S=45, TQ4_1S=46,
  **Q2_0=47 (renumbered from upstream's 42** to avoid colliding with shipped
  TurboQuant types**)**, COUNT=48. Mirrored in `gguf-py/gguf/constants.py`.
  ⚠ Consequence: GGUFs containing Q2_0 tensors quantized by *upstream* builds
  are incompatible with this fork (and vice versa).
- `llama_ftype`: MOSTLY_Q2_0=41 (PR) united with MOSTLY_TQ3_1S=43 / MOSTLY_TQ4_1S=44 (fork).
- `GGML_OP` order: `...GATED_DELTA_NET, TURBO_WHT, LIGHTNING_INDEXER, UNARY...`;
  `FLASH_ATTN_EXT_BANDED` sits after `FLASH_ATTN_EXT` (PR placement). COUNT=100.
  RPC `RPC_PROTO_PATCH_VERSION` bumped to 4 (op enum is wire-visible).
- `llama_vocab`: `PRE_TYPE_LAGUNA=56` kept, `PRE_TYPE_INKLING` moved to 57
  (runtime-only enum, upstream PR had it at 56).

## Per-file conflict resolutions

| File | Resolution |
|---|---|
| `ggml/include/ggml.h` | Union: turbo types + renumbered Q2_0; TURBO_WHT + LIGHTNING_INDEXER ops; both API decls. |
| `ggml/include/ggml-rpc.h` | OP_COUNT assert → 100, patch version → 4. |
| `ggml/src/ggml.c` | Union of op name/symbol tables; both `ggml_turbo_wht` and `ggml_lightning_indexer` constructors; asserts → 100. |
| `ggml/src/ggml-cpu/ops.h`, `ggml-cpu.c` | Union: both forward-decls and dispatch cases. |
| `include/llama.h` | Union of ftype entries. |
| `src/llama-kv-cache.cpp` | Dropped fork's local `ggml_mul_mat_aux` (upstream moved it to `llama-impl.h`; the fork's `GGML_HINT_SRC0_IS_HADAMARD` hint is applied there). Kept fork's InnerQ cross-TU block and default-OFF rotation policy; merged theirs' DEEPSEEK4 into the DSA-indexer force-enable condition (still respects `LLAMA_ATTN_ROT_DISABLE`). |
| `src/llama-model-loader.cpp` | Theirs' `llama_ftype_name` rewrite (prefix-trick) + fork TQ3_1S/TQ4_1S name cases re-inserted. |
| `src/llama-vocab.h` | Union with INKLING renumbered to 57 (see above). |
| `ggml/src/ggml-cuda/fattn.cu` | Kept fork's ncols2 GQA dispatch tail (includes the non-power-of-2 gqa_ratio fix from `61ee3eb9d` — PR side would have regressed it). Added turbo2/3/4 to theirs' new `ggml_cuda_fattn_kv_type_supported()`; turbo head-dim %64 guard re-expressed against the new structure. Banded-op MMA selection from the PR untouched. |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Adopted theirs' deletion of the old `ggml_cuda_op_mul_mat` split infrastructure (upstream removed multi-GPU split-buffer support entirely) and theirs' flat early-return `ggml_cuda_mul_mat`. Ported fork hooks into the new skeleton: Hadamard-hint fWHT early path (already in theirs — fork upstreamed), `is_tq_weight` branch dispatching `ggml_cuda_mul_mat_tq` (≤ MMVQ_MAX_BATCH_SIZE) / `ggml_cuda_mul_mat_tq4_1s_cublas` (large TQ4_1S prefill) / cublas (large TQ3_1S) **before** the mmvf/mmf/mmvq/mmq chain. SET_ROWS supports_op: union (turbo types + head-dim guards + theirs' F16←F16 case). |
| `ggml/src/ggml-metal/ggml-metal.metal` | Union of fork TQ cpy/get_rows/set_rows kernels with theirs' template-signature refactor. Fork set_rows instantiations renamed to upstream's new `kernel_set_rows_{src}_{idx}_{dst}` host_name scheme (`kernel_set_rows_f32_i64_turbo3` etc.) so the host-side name builder resolves them; f32-source only. |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` + `vulkan-shaders-gen.cpp` | Adopted theirs' 2-D (`[src][dst]`) SET_ROWS pipeline table and shader naming; fork turbo set_rows pipelines/shaders re-grafted under the new scheme (f32 source only, `[0][TURBO*]`). supports_op: theirs' src/idx type check + fork's turbo %128 head-dim guard. **Correction (post-merge):** Vulkan DOES carry fork kernels for TURBO_WHT, turbo set_rows and GATED_DELTA_NET (pre-merge fork code, still present); what is genuinely absent is turbo3 flash-attn SPIR-V ("generation deferred") and banded-FA/lightning-indexer kernels (PR is CPU+CUDA only) — those ops are rejected via supports_op. Also: the merge resolution dropped a closing brace in the SET_ROWS turbo guard in `ggml_backend_vk_device_supports_op`, which broke compilation of the whole file (both Vulkan CI builds red from `066cc29` until fixed). |
| `gguf-py/gguf/constants.py` | Mirrors canonical ids (see above); `GGML_QUANT_SIZES` union incl. Q2_0 (64, 2+16). |
| `tests/test-backend-ops.cpp` | Union of fork TQ4_1S mul_mat suites and theirs' new cases. One auto-merge artifact fixed: `test_set_rows::max_nmse_err` TQ4_1S tolerance branch renamed `type` → `type_dst` (upstream member rename). |
| `tests/test-quantize-fns.cpp` | Union: fork TQ3_1S lowbit tolerance + theirs' Q2_0 ternary tolerance. |
| `tools/server/server.cpp` | Theirs' new `llama_server()` split + router detection; fork's child-process `ggml_set_abort_callback` (structured `CMD_CHILD_TO_ROUTER_ERROR` for `/v1/models` error reporting) re-inserted in the `!is_router_server` branch. |
| `tools/server/server-models.{h,cpp}` | Adopted theirs' consolidated state protocol (`CMD_CHILD_TO_ROUTER_STATE` + `handle_child_state`, `update_status(args)` struct, `subproc->stopped` atomic, single `mutex` for cv_stop — fork's separate `stop_mutex` removed). Kept fork additions on top: `CMD_CHILD_TO_ROUTER_ERROR` handling → `update_last_error` → `last_error` in `/v1/models`; force-kill of a still-alive child after stdout EOF (frees GPU memory when a client drops the pipe). Fork's READY/INFO/SLEEP string protocol superseded by theirs' STATE protocol. |
| `tools/server/server-context.cpp` | Kept fork's slot selection (cache-key slot affinity + `get_available_slot(task, allow_prompt_similarity)`); re-added `id_slot` local upstream had dropped. Restored fork's mmproj-draft mirroring: `server_tokens::process_chunk` ported back into `server-common.{h,cpp}` (upstream deleted it) and `llama-ext.h` include restored for `llama_get_ctx_other`. |
| `docs/speculative.md` | Union: fork MTP/NextN docs kept authoritative; theirs' EAGLE-3 / DFlash sections appended. |

## Post-merge integration fixes (outside conflict hunks)

- `tests/snapshots/*.schema` regenerated (`test-quant-type-selection --generate`):
  upstream changed `init_quantize_state_counters` to use `n_layer_all` (includes
  the NextN block), so the NextN layer (`blk.64` on Qwen3.5-27B) now receives
  higher-precision quant types. Intentional upstream behavior; deltas confined
  to NextN blocks + metadata lines.
- `src/llama-quant.cpp` (auto-merged, verified): inkling shortconv kernels and
  rel-proj table are kept unquantized, arch-gated.

## Verification done

- CPU build passes: `cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF && cmake --build build --config Release -j` → exit 0.
- Conflict-marker grep over `*.c/cpp/h/cu/cuh/py` is clean (the only hit is
  ASCII art in vendored `vendor/miniaudio/miniaudio.h`, present upstream).
- `test-backend-ops -o FLASH_ATTN_EXT_BANDED`: **only a CPU device exists on
  this machine and the harness tests other backends against the CPU reference,
  so `test` mode skips (nothing to compare)**. `support` mode confirms CPU
  reports SUPPORTED for the banded op at the PR's production shapes
  (d=128, n_kv up to 32768, rel_extent=1024) and for TURBO_WHT.
- `test-quantize-fns` passes (covers tq3_1s, tq4_1s, q2_0; turbo2/3/4
  intentionally skipped as rotated-domain KV quants).
- `test-quant-type-selection` passes after snapshot regeneration.
- CUDA sources: no conflict markers anywhere under `ggml/src/ggml-cuda/`
  (including all turbo template-instances); every symbol referenced by the
  resolved `ggml_cuda_mul_mat` (`ggml_cuda_op_fwht`, `ggml_cuda_mul_mat_tq`,
  `ggml_cuda_mul_mat_tq4_1s_cublas`, `ggml_cuda_mul_mat_cublas`,
  `should_use_mm{vf,f,vq,q}`, `MMVQ_MAX_BATCH_SIZE`) resolves to a live
  definition; deleted split-infra symbols (`ggml_cuda_cpy_tensor_2d`,
  `MUL_MAT_SRC1_COL_STRIDE`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`,
  `quantize_*_q8_1_cuda` wrappers) have zero remaining references.
  CUDA could not be compiled here (no nvcc).

## Known issues / pre-existing failures (NOT merge regressions)

- `test-llama-archs`: the **laguna** dense fixture fails with
  `key not found: laguna.expert_feed_forward_length`. Pre-existing on the fork
  branch: the laguna loader (`src/models/laguna.cpp`, unchanged by this merge)
  requires the key unconditionally, and the test fixture (identical logic at
  ORIG_HEAD) never writes it for the dense variant. Fix separately (either make
  the key optional for dense laguna or add laguna to `moe_mandatory`).
- The PR marks INKLING as unsupported in `llama_model_saver_supports_arch` and
  skips it in `test-llama-archs` (fixture params for d_rel/rel_extent/shortconv
  not yet modeled) — upstream PR state, kept as-is.

## Still required before release

1. **CUDA build** on a GPU machine (no nvcc here): compile with `-DGGML_CUDA=ON`,
   then `test-backend-ops -o FLASH_ATTN_EXT_BANDED` (exercises the PR's fused MMA
   banded kernel + fp16-accumulator overflow guard) and the full suite to
   regression-check turbo2/3/4 fattn-vec instances against the refactored
   `ggml_cuda_mul_mat` / `ggml_cuda_fattn_kv_type_supported` dispatch.
2. **TQ weight regression on GPU**: decode + prefill with a TQ4_1S/TQ3_1S model
   (the TQ dispatch was ported into upstream's new flat mul_mat — behavior
   should be identical, but the large-TQ3_1S-batch path now goes through
   `ggml_cuda_mul_mat_cublas` dequant instead of the old op_mul_mat wrapper).
3. **Real-model smoke tests**:
   - An Inkling GGUF end-to-end (convert → load → generate), including the
     typed-content-block chat template and >128K attention log-scaling;
     verify the turbo-cache fallback warning fires with `-ctk turbo4_0`.
   - Gemma 4 MTP and Qwen 3.6 NextN speculative decoding (server + CLI) to
     confirm the MTP plumbing survived the upstream server refactor.
   - turbo3/turbo4 KV cache PPL spot-check on one known-good model
     (e.g. gemma-4 E2B) vs pre-merge numbers.
4. **Metal / Vulkan builds** (macOS / any Vulkan box): the set_rows host-name
   scheme changed upstream; fork kernels were renamed to match
   (`kernel_set_rows_f32_i64_turbo3`, `set_rows_f32_turbo3_0_i32`) but have only
   been verified by inspection, not compiled.
5. Multi-GPU users: upstream **removed split-buffer (row-split) multi-GPU
   support** from the CUDA backend in this range — `-sm row` behavior is gone.
   Flag this in release notes.
