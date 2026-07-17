# Speculative Decoding

llama.cpp supports speculative decoding, a technique that can significantly accelerate token generation by predicting multiple tokens ahead of the main model.

[Speculative decoding](https://en.wikipedia.org/wiki/Transformer_(deep_learning)#Speculative_decoding) leverages the fact that computing n tokens in a batch (as in prompt processing) is more efficient than computing n sequentially (as in response generation). By generating draft tokens quickly and then verifying them with the target model in a single batch, this approach can achieve substantial speedups when the draft predictions are frequently correct.

## Implementations

The `llama-server` application supports several implementations of speculative decoding. An implementation with draft model can be mixed with an implementation without draft model.

### Multimodal (`--mmproj`) compatibility (atomic-llama-cpp-turboquant)

When `--mmproj` is set, **`mtp`**, **`nextn`**, and **`eagle3`** speculative types remain **enabled at load**: their draft paths do not depend on `get_text_tokens()` / `prompt_tgt` the way `draft` and `ngram_*` do. Other types are auto-disabled at load with a warning. Mixed speculative chains (e.g. `ngram_simple` + `draft`) are rejected at slot init if any impl is not multimodal-safe.

**Per-turn behaviour:**

- **Text-only turns on a multimodal slot** — draft head runs as usual (same acceptance as without `--mmproj`).
- **Turns containing an image chunk** — the slot logs `skipping speculative prime for multimodal prompt` and falls back to plain target decoding for that turn only. The image is still recognised correctly, just without draft speedup.

The fallback is required because NextN / MTP prime needs per-token target hidden states for every prompt position, but mtmd image-decode currently only emits an output row for the last token of each image batch. Lifting this restriction is on the roadmap (mtmd batches need to mark every position with `logits[i] = true`, or be replayed teacher-forced).

See `common_speculative_is_mtmd_safe` / `common_speculative_all_impls_mtmd_safe` in [`common/speculative.cpp`](../common/speculative.cpp), the `mmproj` gates and the `skip_draft_mtmd` per-turn gate in [`tools/server/server-context.cpp`](../tools/server/server-context.cpp). Validated configurations and the end-to-end recipe are documented in [`NEXTN.md` §10](../NEXTN.md#10-multimodal---mmproj--speculative-decoding-this-fork).

### Draft Model (`draft`)

A much smaller model (called the _draft model_) generates drafts.
A draft model is the most used approach in speculative decoding.

### Gemma 4 MTP assistant (`mtp`)

For **Gemma 4** targets with the **gemma4_assistant** (MTP head) GGUF, use `--spec-type mtp`. The assistant is **not** a second `llama_context`: weights are loaded into the target model via `llama_model_load_mtp_from_file` (done automatically when using the server/CLI init path). Cross-attention in the MTP graph reads **K/V from the target KV cache** (shared full/sliding layers).

- Prefer **`--mtp-head /path/to/assistant.gguf`** for clarity; **`--model-draft` (`-md`)** is accepted as a backward-compatible alias (same path field).
- The draft block size \(B\) is **`--draft-block-size`** (the head proposes `B - 1` tokens per round; default 4).
- **`--gpu-layers-draft` / `-ngld`** and **`-ctkd` / `-ctvd`** still apply to how the **assistant tensors** are placed and typed when the assistant GGUF is loaded; the target uses `-ngl` and `-ctk`/`-ctv`.

Example (paths illustrative). **TurboQuant** KV on the target: `-ctk`/`-ctv`. Assistant-side cache types follow the draft flags if you use them for offload/quant selection.

```sh
llama-server \
  -m /path/to/gemma-4-target.gguf \
  --mtp-head /path/to/gemma-4-assistant.gguf \
  --spec-type mtp \
  --draft-block-size 4 \
  -c 16384 \
  -ngl 99 -ngld 99 \
  -ctk turbo3 -ctv turbo3 \
  -ctkd turbo3 -ctvd turbo3 \
  -fa on \
  --host 127.0.0.1 --port 8080
```

Repo helper (defaults under `.scratch/`): `scripts/run-gemma4-mtp-server.sh`.

#### Async MTP draft pipeline

The MTP draft head runs on a dedicated `ggml_backend_sched` (`sched_mtp`) and a
worker thread, isolated from the target's scheduler. This is exposed via two C
APIs:

```c
LLAMA_API int32_t llama_decode_mtp_async(
        struct llama_context * ctx,
        llama_seq_id  seq_id,
        llama_pos     attn_pos,
        llama_token   last_token,
        const float * h_prev,
        int32_t       n_steps);

LLAMA_API int32_t llama_decode_mtp_wait(
        struct llama_context * ctx,
        llama_token * out_drafts,
        float       * out_h_prev_last);
```

Contract:

- At most **one in-flight request per context**. Calling `_async` while a
  previous request has not been waited on returns `-7`.
- `h_prev` is copied into the request, so the caller may free or reuse it
  immediately after `_async` returns.
- Target KV positions ≤ `attn_pos` must remain stable until `_wait` returns.
  The current append-only KV cache satisfies this as long as no eviction or
  overlapping `seq_rm` happens between submit and wait.
- `llama_decode_mtp(...)` is preserved as a backward-compatible synchronous
  facade (= `_async` immediately followed by `_wait`).

Why use the async pair?

- **Graph-cache isolation**: MTP graph reuse is no longer invalidated by target
  decode resets. On Gemma 4 + Q4_K_XL, this alone delivered **~+8% throughput**
  in single-slot benchmarks (95.3 → 102.8 tps at `--draft-block-size 3`), with
  identical accept rate.
- **Pipeline depth-2 (pure overlap, `llama-server`)**: after target
  sample/accept and `llama_memory_seq_rm`, the server calls
  `common_speculative_prepare_next(spec, last_accepted_token)`, which submits
  `llama_decode_mtp_async` for the *next* round. The blocking
  `llama_decode_mtp_wait` is deferred to the start of the next
  `common_speculative_draft`, so MTP work can overlap with post-accept
  bookkeeping and token I/O. This uses the real sampled token (no optimistic
  guess). **KV contract**: submit uses `attn_pos` after `seq_rm`; the next
  `llama_decode` appends only positions `> attn_pos`, so backbone cells read by
  MTP remain stable until `_wait` (append-only cache). Stale in-flight requests
  are drained in `common_speculative_begin` and on skip / param-mismatch paths.
- **In-graph argmax**: the MTP graph publishes a `ggml_argmax` of the final
  logits (I32 [1]) via `llm_graph_result::get_argmax()`. Per draft step the host
  reads back 4 bytes (one token id) instead of the full F32 [n_vocab] logits
  row, and the serial CPU argmax over `n_vocab` is gone. The full logits are
  still computed in-graph and can be fetched on demand by passing a non-null
  `out_logits` to the synchronous `llama_decode_mtp(...)` API (diagnostic /
  legacy path). On Gemma 4 + Q4_K_XL this delivered an additional **~+2-3%
  throughput** (`109.5 → 112.5 tps` at `n=128`, `95.8 → 97.8 tps` at `n=512`),
  with bit-identical greedy drafts.
- **Future work**: optimistic last-token prediction could hide an additional
  `llama_decode` latency on hits but risks wrong drafts on misses.

#### Diagnostic: per-draft acceptance trace

Set `LLAMA_MTP_ACC_TRACE` to enable a per-iteration NDJSON tracer in
`common/speculative.cpp`. Each `mtp_draft` event records `iter`, `path`
(`sync` / `lazy` / `skip-streak` / `skip-nsteps`), `seq_id`, `id_last`,
`h_idx`, `attn_pos`, `n_steps`, `h_l2` (L2 norm of the input hidden state),
and the drafted token ids; each `mtp_accept` event records the iteration
counter, `n_accepted`, and `n_drafted_prev`. The host can pair them by `iter`
to reconstruct per-position acceptance, MTP h_prev stability, and
selection-bias breakdowns by `h_idx`.

```sh
# write trace to stderr
LLAMA_MTP_ACC_TRACE=1 ./llama-server [...]
# write trace to a file
LLAMA_MTP_ACC_TRACE=/tmp/mtp.ndjson ./llama-server [...]
```

#### Throughput tuning (Edge assistants / heavy backends)

Ordered-embeddings MTP (`use_ordered_embeddings`) runs a centroid LM head (`top_k`, routed `get_rows`) per draft step; the greedy graph still materializes a full-vocab logits row (masked fill + scatter) so argmax and optional full-row export stay consistent with verify.

- **`LLAMA_MTP_SKIP_STREAK_THRESHOLD`**: **off by default** (`0` / unset). If set to `1`–`32`, after that many consecutive batches with **zero** accepted **MTP** drafts, MTP drafting is skipped for one verify-only batch, then retried.
- Helper scripts `scripts/run-gemma4-e4b-mtp-server.sh` and `scripts/run-gemma4-e2b-mtp-server.sh` support **`MTP_PRESET`**: `throughput` (block 2 / max 6), `lift` (block 3 / max 8), `balanced`, or `quality`. Override with `DRAFT_BLOCK_SIZE`, `DRAFT_MAX`, and optionally `LLAMA_MTP_SKIP_STREAK_THRESHOLD`.

Disabled at zero overhead (no env var or value `0` / empty); when enabled the
extra cost is one `n_bb`-wide L2 reduction per draft and a small NDJSON write.

#### Reconverting `gemma4_assistant` from Hugging Face

Example (paths relative to repo root):

```sh
python convert_hf_to_gguf.py .scratch/gemma-4-26B-A4B-it-assistant \
  --outfile .scratch/gemma-assistant-mtp.gguf --outtype f16
```

Use the resulting GGUF as `--mtp-head` (or `-md`) with `--spec-type mtp`. Older assistant GGUFs with `token_embd.weight` first axis 2816 (backbone width) instead of 1024 will fail load; run `scripts/verify-gemma4-assistant-gguf.py` on the file to check.

### Qwen 3.x NextN (`nextn`)

For **Qwen3.6** (and compatible) checkpoints that ship NextN head weights in the combined `*_MTP.gguf`, use `--spec-type nextn` with **`--model-draft` (`-md`)** pointing at the **same** GGUF as the main model. The server detects this and **reuses the already-loaded target `llama_model`** — a second `llama_context` is built over the same weights with `llama_context_params.nextn_draft = true`, which routes graph construction to `llm_build_qwen35_nextn` / `llm_build_qwen35moe_nextn` and sizes the draft KV cache only for the NextN layer (`kv_only_nextn = true`, mutated transparently inside `llama_context` ctor). There is **no second mmap of the GGUF**.

- Drafting reads **CPU-copied** pre-final-norm hidden states (`embeddings_pre_norm` path); it does **not** use Gemma's `llama_decode_mtp_*` APIs.
- **`llama_set_nextn`** only pairs target and draft for **`llama_context_nextn_seq_rm`**; see `NEXTN.md` for details.
- Standalone NEXTN_ONLY GGUFs (`general.architecture = qwen35*_mtp`) are still supported as a fallback for users who ship the draft head as a separate artifact (the server then performs a second `llama_model_load_from_file` with `override_arch`); the shared-model path is preferred whenever `--model` and `--model-draft` point at the same combined `_MTP.gguf`.

```sh
llama-server \
  -m /path/to/qwen3.6-MTP.gguf \
  -md /path/to/qwen3.6-MTP.gguf \
  --spec-type nextn \
  --draft-max 2 --draft-min 1 \
  -c 8192 -ngl 99 -ngld 99 -fa on \
  --host 127.0.0.1 --port 8080
```

Pair with `-ctk turbo3 -ctv turbo3` to compose with **TurboQuant** KV — on MoE targets (e.g. Qwen 3.6 35B-A3B) this combination is **+24-36% tps** over the `turbo3` baseline at single-slot in the matrix bench (see `NEXTN.md §7`).

Repo helpers: `scripts/run-qwen36-27b-nextn-server.sh`, `scripts/run-qwen36-35ba3b-nextn-server.sh`.

### EAGLE-3 (`draft-eagle3`)

EAGLE-3 uses a small draft model that reads the target model's hidden states to predict the next tokens, so it
reaches higher acceptance than a standalone draft model of the same size. The draft is a one-layer transformer
trained for a specific target model; it shares the target model's tokenizer and, optionally, uses a reduced draft
vocabulary with its own `lm_head`, which is mapped back using a `d2t` table.

Convert the EAGLE-3 checkpoint with `--target-model-dir` so it inherits the target's tokenizer and the layer
indices to read. Both the SpecForge `LlamaForCausalLMEagle3` and the vLLM/AngelSlim `Eagle3LlamaForCausalLM`
checkpoint formats are supported (for example [`AngelSlim/Qwen3-4B_eagle3`](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
for `Qwen/Qwen3-4B`):

```bash
python convert_hf_to_gguf.py AngelSlim/Qwen3-4B_eagle3 \
    --target-model-dir Qwen/Qwen3-4B --outtype bf16 --outfile Qwen3-4B-eagle3.gguf

llama-server -m Qwen3-4B.gguf -md Qwen3-4B-eagle3.gguf --spec-type draft-eagle3
```

Supported EAGLE-3 draft models include:

- [yuhuili/EAGLE3-LLaMA3.1-Instruct-8B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B)
- [yuhuili/EAGLE3-LLaMA3.3-Instruct-70B](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B)
- [RedHatAI/gemma-4-31B-it-speculator.eagle3](https://huggingface.co/RedHatAI/gemma-4-31B-it-speculator.eagle3)
- [RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3)
- [Tengyunw/qwen3_8b_eagle3](https://huggingface.co/Tengyunw/qwen3_8b_eagle3)
- [Tengyunw/qwen3_30b_moe_eagle3](https://huggingface.co/Tengyunw/qwen3_30b_moe_eagle3)
- [AngelSlim/Qwen3-1.7B_eagle3](https://huggingface.co/AngelSlim/Qwen3-1.7B_eagle3)
- [AngelSlim/Qwen3-4B_eagle3](https://huggingface.co/AngelSlim/Qwen3-4B_eagle3)
- [AngelSlim/Qwen3-8B_eagle3](https://huggingface.co/AngelSlim/Qwen3-8B_eagle3)
- [AngelSlim/Qwen3-14B_eagle3](https://huggingface.co/AngelSlim/Qwen3-14B_eagle3)
- [AngelSlim/Qwen3-32B_eagle3](https://huggingface.co/AngelSlim/Qwen3-32B_eagle3)
- [AngelSlim/Qwen3-a3B_eagle3](https://huggingface.co/AngelSlim/Qwen3-a3B_eagle3)
- [RedHatAI/gpt-oss-20b-speculator.eagle3](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3)
- [lmsys/EAGLE3-gpt-oss-120b-bf16](https://huggingface.co/lmsys/EAGLE3-gpt-oss-120b-bf16)
- [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context)

For the full and up-to-date list of supported models, see #18039.

### DFlash (`draft-dflash`)

DFlash produces an entire block of draft tokens in a single forward pass (block diffusion) and
injects the target model's hidden states into the draft model's attention, instead of drafting one
token at a time. This keeps the draft model small while making drafting GPU-friendly. Unlike EAGLE-3
(a single-layer autoregressive draft), the DFlash draft uses several transformer layers but emits a
whole block per draft step.

The draft is a small block-diffusion model trained for a specific target (for example
`z-lab/Qwen3-4B-DFlash` for `Qwen/Qwen3-4B`). Convert it with `--target-model-dir` so it inherits the
target's tokenizer and token embeddings:

```bash
python convert_hf_to_gguf.py z-lab/Qwen3-4B-DFlash \
    --target-model-dir Qwen/Qwen3-4B --outtype bf16 --outfile Qwen3-4B-DFlash.gguf

llama-server -m Qwen3-4B.gguf -md Qwen3-4B-DFlash.gguf \
    --spec-type draft-dflash --spec-draft-n-max 15 -fa on --jinja
```

`--spec-draft-n-max` is clamped to the draft model's trained block size.

See:

- #22105

### n-gram Cache (`ngram-cache`)

An n-gram is a sequence of n tokens. The n-gram cache implementation maintains statistics about short n-gram sequences.
A draft is computed using probabilities derived from these statistics. External statistics can also be loaded from files for improved accuracy.

See:

- #5479, #6828, #6848

### n-gram Map (`ngram-simple`, `ngram-map-*`)

These implementations search the token history for patterns and use matching sequences as draft candidates.
They require no additional model but rely on patterns that have already appeared in the generated text.
An example to use this approach can be the rewriting of source code by a LLM.

#### n-gram Map (`ngram-simple`)

This implementation looks for the last n-gram in history that matches the current n-gram and creates a draft using the m tokens following the matched n-gram. It is the simplest self-speculative approach with minimal overhead.

```
llama-server [...] --spec-type ngram-simple --spec-draft-n-max 64
```

#### n-gram Map Key (`ngram-map-k`)

This implementation looks for the current n-gram of size n (called the _key_) in the token history. If the key n-gram is followed by the same m tokens (called the _mgram_) multiple times, it creates a draft using these m tokens. This approach requires a minimum number of occurrences (argument `--spec-ngram-map-k-min-hits`, default is 1) before generating drafts.

The number of accepted tokens is stored for each used n-gram.

**Example:**
```
llama-server [...] --spec-type ngram-map-k --spec-draft-n-max 64
```

#### n-gram Map Key-4-Values (`ngram-map-k4v`)

This experimental implementation looks for the current n-gram of size n (called the _key_) in the token history. For each key, up to four _values_ (n-grams of size m, called _mgrams_) are tracked. An internal statistic counts the occurrences of each mgram after the key n-gram. If one mgram is significantly more frequent than the others, it is used as the draft.

The number of accepted tokens is stored for each used n-gram.

**Example:** Server options to be used if there are a lot of longer repetitions.
```
llama-server [...] --spec-type ngram-map-k4v --spec-ngram-map-k4v-size-n 8 --spec-ngram-map-k4v-size-m 8 --spec-ngram-map-k4v-min-hits 2 --spec-draft-n-max 64
```

### n-gram Mod (`ngram-mod`)

Add basic ngram hasher for speculative decoding:

- For each ngram, compute a hash using LCG
- For each computed hash, store the next token
- During speculation, iteratively compute the rolling hash of the last n tokens and pick the next token from the storage

Some characteristics:

- Lightweight (~16 MB)
- Constant memory and complexity
- Can generate variable draft lengths (i.e. m is not fixed)

Currently, a single hash pool is shared across all server slots, so different requests can benefit from each other.

**Sample usage:**

```
# notes:
# - small `n` are not recommended
# - MoEs require long drafts
# - dense models: can reduce `--spec-ngram-mod-n-min` and `--spec-ngram-mod-n-max`

llama-server ... --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64
```

Applications:

- Iterating over a block of text/code (e.g. in llama.vim)
- Reasoning models (when they have to repeat their thinking in the final answer)
- Summarization

Example Video:

- See #19164

### Differences between ngram-simple, ngram-map and ngram-mod

- ngram-simple looks for a previous matching n-gram and inserts the following m-gram.
- ngram-map-k looks for a previous matching n-gram and inserts the following m-gram but uses an internal hash-map of n-grams in the current context window.
- ngram-mod uses a hash pool which is shared across all server slots. The hash pool is a map from n-gram hash to the next token (not the next m-gram as in ngram-map).

## Command-Line Options

If a draft model is combined with a draftless decoding the draftless decoding has higher precedence.

### General Speculative Parameters

```
--spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-mtp|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]
                                        comma-separated list of types of speculative decoding to use
                                        (default: none)
                                        (env: LLAMA_ARG_SPEC_TYPE)
--spec-default                          use default speculative decoding config
                                        (enables ngram-mod)
```

### Draft Model Parameters

```
--spec-draft-model, -md, --model-draft  FNAME
                                        draft model for speculative decoding (default: unused)
                                        (env: LLAMA_ARG_SPEC_DRAFT_MODEL)
--spec-draft-hf, -hfd, -hfrd, --hf-repo-draft  <user>/<model>[:quant]
                                        HuggingFace repository for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_HF_REPO)
--spec-draft-n-max                      N
                                        number of tokens to draft for speculative decoding (default: 3)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MAX)
--spec-draft-n-min                      N
                                        minimum number of draft tokens to use for speculative decoding (default: 0)
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_MIN)
--spec-draft-p-split, --draft-p-split   P
                                        speculative decoding split probability (default: 0.10)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_SPLIT)
--spec-draft-p-min, --draft-p-min       P
                                        minimum speculative decoding probability (greedy) (default: 0.00)
                                        (env: LLAMA_ARG_SPEC_DRAFT_P_MIN)
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft  N
                                        max. number of draft model layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS_DRAFT)
--spec-draft-device, -devd, --device-draft  <dev1,dev2,..>
                                        comma-separated list of devices to use for offloading the draft model
                                        (use --list-devices to see available devices)
```

### Draft Model CPU Scheduling Parameters

```
--spec-draft-threads, -td, --threads-draft  N
                                        number of CPU threads to use during generation
--spec-draft-threads-batch, -tbd, --threads-batch-draft  N
                                        number of threads to use during batch and prompt processing (default: same as --threads-draft)
--spec-draft-cpu-mask, -Cd, --cpu-mask-draft  M
                                        Draft model CPU affinity mask. Complements cpu-range-draft
--spec-draft-cpu-range, -Crd, --cpu-range-draft  lo-hi
                                        Ranges of CPUs for affinity. Complements --cpu-mask-draft
--spec-draft-cpu-strict, --cpu-strict-draft  <0|1>
                                        Use strict CPU placement for draft model (default: same as --cpu-strict)
--spec-draft-prio, --prio-draft  N
                                        set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime
--spec-draft-poll, --poll-draft  <0|1>
                                        Use polling to wait for draft model work (default: same as --poll)
--spec-draft-cpu-mask-batch, -Cbd, --cpu-mask-batch-draft  M
                                        Draft model CPU affinity mask for batch. Complements cpu-range-batch-draft
--spec-draft-cpu-range-batch, -Crbd, --cpu-range-batch-draft  lo-hi
                                        Ranges of CPUs for affinity for batch. Complements --cpu-mask-batch-draft
--spec-draft-cpu-strict-batch, --cpu-strict-batch-draft  <0|1>
                                        Use strict CPU placement for draft model batch (default: --cpu-strict-draft)
--spec-draft-prio-batch, --prio-batch-draft  N
                                        set draft process/thread priority for batch : 0-normal, 1-medium, 2-high, 3-realtime
--spec-draft-poll-batch, --poll-batch-draft  <0|1>
                                        Use polling to wait for draft model work for batch (default: --poll-draft)
```

### Draft Model KV Cache and Tensor Override Parameters

```
--spec-draft-type-k, -ctkd, --cache-type-k-draft  TYPE
                                        KV cache data type for K for the draft model
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (env: LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K)
--spec-draft-type-v, -ctvd, --cache-type-v-draft  TYPE
                                        KV cache data type for V for the draft model
                                        allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                                        (env: LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V)
--spec-draft-override-tensor, -otd, --override-tensor-draft  <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type for draft model
--spec-draft-cpu-moe, -cmoed, --cpu-moe-draft
                                        keep all Mixture of Experts (MoE) weights in the CPU for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_CPU_MOE)
--spec-draft-n-cpu-moe, --spec-draft-ncmoe, -ncmoed, --n-cpu-moe-draft  N
                                        keep the MoE weights of the first N layers in the CPU for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE)
```

### n-gram Mod Parameters

```
--spec-ngram-mod-n-match                N
                                        ngram-mod lookup length (default: 24)
--spec-ngram-mod-n-min                  N
                                        minimum number of ngram tokens to use for ngram-based speculative decoding (default: 48)
--spec-ngram-mod-n-max                  N
                                        maximum number of ngram tokens to use for ngram-based speculative decoding (default: 64)
```

### n-gram Simple Parameters

```
--spec-ngram-simple-size-n              N
                                        ngram size N for ngram-simple speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-simple-size-m              N
                                        ngram size M for ngram-simple speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-simple-min-hits            N
                                        minimum hits for ngram-simple speculative decoding (default: 1)
```

### n-gram Map Key Parameters

```
--spec-ngram-map-k-size-n               N
                                        ngram size N for ngram-map-k speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k-size-m               N
                                        ngram size M for ngram-map-k speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k-min-hits             N
                                        minimum hits for ngram-map-k speculative decoding (default: 1)
```

### n-gram Map Key-4-Values Parameters

```
--spec-ngram-map-k4v-size-n             N
                                        ngram size N for ngram-map-k4v speculative decoding, length of lookup n-gram (default: 12)
--spec-ngram-map-k4v-size-m             N
                                        ngram size M for ngram-map-k4v speculative decoding, length of draft m-gram (default: 48)
--spec-ngram-map-k4v-min-hits           N
                                        minimum hits for ngram-map-k4v speculative decoding (default: 1)
```

### `--spec-type TYPE`

Specifies a comma-separated list of speculative decoding types to use.

| Type | Description |
|------|-------------|
| `none` | No speculative decoding (default) |
| `draft-simple` | Use a simple draft model for speculation |
| `draft-eagle3` | Use an EAGLE-3 draft model that reads the target's hidden states |
| `draft-dflash` | Use a DFlash block-diffusion draft model that emits a block per step |
| `draft-mtp` | Use Multi Token Prediction (MTP) heads from the main model |
| `ngram-cache` | Use n-gram cache lookup |
| `ngram-simple` | Use simple n-gram pattern matching |
| `ngram-map-k` | Use n-gram pattern matching with n-gram-keys |
| `ngram-map-k4v` | Use n-gram pattern matching with n-gram-keys and up to four m-gram values (experimental) |
| `ngram-mod` | Use basic ngram hasher for speculative decoding with shared pool |

**Example:** Server-instance used to refactor source code.
```bash
./llama-server [...] --spec-type ngram-simple
```

**Example:** Multiple speculative implementations.
```bash
./llama-server [...] --spec-type ngram-mod,ngram-map-k4v
```

### `--spec-ngram-*-size-n N`

Sets the size N of the lookup n-gram for n-gram map based speculative decoding.
The n-gram size N determines how many tokens in a row to look back when searching for matching patterns.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-n` for `ngram-simple`
- `--spec-ngram-map-k-size-n` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-n` for `ngram-map-k4v`
- `--spec-ngram-mod-n-match` for `ngram-mod`

### `--spec-ngram-*-size-m M`

Sets the size M of the draft m-gram for n-gram map based speculative decoding.
The m-gram size determines how many tokens to draft when a match is found.
Larger values can provide more speedup but may reduce acceptance rate.

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-size-m` for `ngram-simple`
- `--spec-ngram-map-k-size-m` for `ngram-map-k`
- `--spec-ngram-map-k4v-size-m` for `ngram-map-k4v`

### `--spec-ngram-*-min-hits H`

This option defines how often a key has to appear in the token history to be used as a draft (default is 1).

Each n-gram implementation has its own parameter:

- `--spec-ngram-simple-min-hits` for `ngram-simple`
- `--spec-ngram-map-k-min-hits` for `ngram-map-k`
- `--spec-ngram-map-k4v-min-hits` for `ngram-map-k4v`

## Statistics
Each speculative decoding implementation prints statistics.

```
draft acceptance rate = 0.57576 (  171 accepted /   297 generated)
statistics ngram_simple: #calls = 15, #gen drafts = 5, #acc drafts = 5, #gen tokens = 187, #acc tokens = 73
statistics draft: #calls = 10, #gen drafts = 10, #acc drafts = 10, #gen tokens = 110, #acc tokens = 98
```

```
draft acceptance rate = 0.70312 (   90 accepted /   128 generated)
statistics ngram_mod: #calls = 810, #gen drafts = 15, #acc drafts = 15, #gen tokens = 960, #acc tokens = 730, dur(b,g,a) = 0.149, 0.347, 0.005 ms
```

```
statistics ngram_map_k: #calls(b,g,a) = 6 1690 26, #gen drafts = 26, #acc drafts = 26, #gen tokens = 1248, #acc tokens = 968, dur(b,g,a) = 2.234, 1.427, 0.016 ms
```


- `#calls(b,g,a)`: number of calls of begin (new prompt), generation and accumulation of this implementations
- `#gen drafts`: number of drafts generated by this implementation
- `#acc drafts`: number of drafts accepted (partially) by the main model
- `#gen tokens`: number of tokens generated by this implementation (including rejected tokens)
- `#acc tokens`: number of tokens accepted by the main model
- `dur(b,g,a): durations of begin (new prompt), generation and accumulation (process acceptance).

## Benchmarking

To measure the end-to-end effect of speculative decoding (throughput, latency, and draft acceptance) across diverse prompts, see the SPEED-Bench client in [tools/server/bench/speed-bench](../tools/server/bench/speed-bench/README.md).
It runs against a running `llama-server` and can compare a baseline run against a speculative-decoding run.
