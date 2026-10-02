# Qwen 3.x NextN — shared-model speculative decoding

> Scope: **Qwen3.6** (and compatible) models with NextN / MTP auxiliary head weights in GGUF.
> The draft context now reuses the **target** `llama_model` (no second mmap of the combined
> `_MTP.gguf`); a second `llama_context` is built over the same model with
> `llama_context_params.nextn_draft = true`, which routes graph build to the NextN draft
> builder (`qwen35_nextn` / `qwen35moe_nextn`).
> Legacy standalone `*_mtp` GGUFs (`override_arch`) are still supported as a fallback for
> users who ship the draft head as a separate artifact.
> This path is **named `nextn`** in this fork to coexist with **Gemma 4 MTP** (`--spec-type mtp`), which uses a
> single target context and `llama_decode_mtp_*`.

See also `MTP.md` (Gemma) and `docs/speculative.md` for shared CLI concepts.

---

## 0. Pre-built model GGUFs

**Recommended:** the [AtomicChat — Qwen 3.6 UDT](https://huggingface.co/collections/AtomicChat/qwen-36-udt-atomicchat-6a0481f5cc5a057c07759176) collection — drop-in combined `*_MTP.gguf` quants tuned for this fork. Each repo ships Q3 / **Q4** / Q5 / Q6 / Q8 `_K_XL`, plus the `mmproj` for vision and a copy of `imatrix_unsloth.gguf_file` for reproducibility. Upstream Unsloth files keep working too — same arch metadata, same NextN tail.

| Target | Recommended (AtomicChat UDT) | Upstream baseline (Unsloth) | Architecture |
|---|---|---|---|
| Qwen 3.6 35B-A3B (MoE) | [`AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF) (`Q4_K_XL` ≈ 20.7 GiB) | [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) | `qwen35moe` |
| Qwen 3.6 27B (dense)   | [`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF) (`Q4_K_XL` ≈ 17.7 GiB)             | [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)             | `qwen35`    |

**Why UDT** — built on Unsloth's public MTP-aware [`imatrix_unsloth.gguf_file`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/main/imatrix_unsloth.gguf_file), then layered with this fork's tensor-type masks (see §8): every `blk.*.nextn.*` / `mtp.*` tensor pinned to `Q8_0` to preserve draft acceptance, and `attn_q` / `attn_k` lifted to `Q6_K` so the file pairs cleanly with TurboQuant3 KV. End-to-end recipe & runbook: [docs/qwen-udt/RUNBOOK.md](docs/qwen-udt/RUNBOOK.md). Attribution: Qwen team (weights), Unsloth (imatrix + BF16 sources), @TheTom (TurboQuant), AtomicChat (UDT masks + packaging).

The shared-model NextN path
works on **any** of them as long as the file contains the NextN auxiliary
head (`nextn_predict_layers > 0`) — which all `*-MTP-GGUF` quants do by
construction. `scripts/verify-qwen36-nextn-gguf.py` will refuse to load a
file missing the NextN layer.

Quick pull via `-hf` (target) + `-hfd` (draft); the server resolves both to
the same file in the HF cache and takes the shared-model branch:

```bash
# 35B-A3B MoE (headline +24-36 % cell in the matrix)
llama-server \
  -hf  AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF:Q4_K_XL \
  -hfd AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF:Q4_K_XL \
  --spec-type nextn --spec-draft-n-max 2 --spec-draft-n-min 1 \
  -c 8192 -ngl 99 -ngld 99 -fa on

# 27B dense
llama-server \
  -hf  AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF:Q4_K_XL \
  -hfd AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF:Q4_K_XL \
  --spec-type nextn --spec-draft-n-max 2 --spec-draft-n-min 1 \
  -c 8192 -ngl 99 -ngld 99 -fa on
```

---

## 1. Architecture

| Piece | Role |
|-------|------|
| Target context | Standard `qwen35` / `qwen35moe` forward; graph publishes `t_h_pre_norm` (hidden before final norm). |
| Draft context | Built over the **same** `llama_model` with `cparams.nextn_draft = true`. The graph dispatcher picks `llm_build_qwen35*_nextn` against the target's NextN-layer tensors (`model.layers[n_main + i].nextn.*`). KV cache is sized only for the NextN layer (`kv_only_nextn = true`, overridden transparently in `llama_context` ctor). |
| Hidden transfer | Target and draft enable `embeddings_pre_norm`; `llama_decode` copies `t_h_pre_norm` rows into a CPU `embd_pre_norm` buffer. `common_speculative_state_nextn` reads via `llama_get_embeddings_pre_norm_ith` (no per-ubatch tensor hook). |
| Speculative driver | `common_speculative_state_nextn` in `common/speculative.cpp` (greedy Top-1 chain). |
| KV pairing | `llama_set_nextn(target, draft)` registers the draft context so `llama_context_nextn_seq_rm` can trim both KVs. |

The shared-model path eliminates the ~22 GB second mmap (one `MTLBuffer` per `llama_model`)
that used to OOM the 35B-A3B target on Apple Silicon (38 GB unified memory). See
`llama_model_has_nextn_layer()` (target arch ∈ {qwen35, qwen35moe} **and**
`hparams.nextn_predict_layers > 0`).

---

## 2. CLI / server

- `--spec-type nextn` — enable NextN drafting (not Gemma `mtp`).
- `--model-draft` / `-md` — pass the **same** path as `--model`; the server detects this
  and switches to the shared-model path (no second model load). Pointing at a standalone
  NEXTN_ONLY GGUF (`general.architecture = qwen35*_mtp`) still works but loads a second
  `llama_model`.
- `--spec-draft-n-max` — max chained draft tokens per round (pre-b10018 name: `--draft-max`).
- Gemma MTP flags (`--mtp-head`, `llama_decode_mtp_*`, `llama_model_load_mtp_from_file`) are **unchanged**.

---

## 3. C API (subset)

- `llama_set_nextn(target_ctx, draft_ctx)` — pair contexts for paired `seq_rm`.
- `llama_context_nextn_seq_rm(target_ctx, …)` — remove KV on target **and** on the registered draft context (`seq_id` 0 on draft).

Internal (see `src/llama-ext.h`, not in stable `include/llama.h`):

- `llama_set_embeddings_pre_norm(ctx, bool)` — enable extraction/copy of pre-norm hidden rows into `embd_pre_norm`.
- `llama_get_embeddings_pre_norm_ith(ctx, i)` — row `i` of the last decode’s pre-norm buffer (`i < 0` supported like other embedding getters).

---

## 4. Operations

- **Vocab**: draft and target share tokenizer; arch check ensures `qwen35`+`qwen35_mtp` (or MoE pair).
- **GDN rollback**: target may use `n_rs_seq` from speculative+GDN work; draft context forces `n_rs_seq = 0` (see `tools/server/server-context.cpp`).
- **Metal / Vulkan**: GDN partial rollback quality may still be upstream-limited; see PR #22400 notes in the project plan.

---

## 5. Verify GGUF

```bash
PYTHONPATH=gguf-py python3 scripts/verify-qwen36-nextn-gguf.py /path/to/model.gguf
```

---

## 6. Run scripts

- `scripts/run-qwen36-27b-nextn-server.sh`
- `scripts/run-qwen36-35ba3b-nextn-server.sh`

Set `MAIN_GGUF` to your Qwen3.6 `*_MTP.gguf` (see §0 for the recommended
unsloth quants); draft defaults to the same path so the server takes the
shared-model branch. Alternatively use `-hf` (target) + `-hfd` (draft) to
let `llama-server` pull both from Hugging Face into the local cache:

```bash
llama-server \
  -hf  AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF:Q4_K_XL \
  -hfd AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF:Q4_K_XL \
  --spec-type nextn --spec-draft-n-max 2 --spec-draft-n-min 1
```

---

## 7. Performance notes (MacBook Pro M4 Max, 40-core GPU, 48 GB, Metal)

Median TPS over 2 runs, prompt = 50-token instruction, `--spec-draft-n-max=2 --spec-draft-n-min=1`,
NextN draft DM=2 (single async chain), context 8192. Single-slot
(`--parallel 1 -np 1 --cont-batching`), full GPU offload (`-ngl 99 -ngld 99 -fa on`),
shared-model draft path (no second mmap of combined `_MTP.gguf`),
AtomicChat **`UDT-Q4_K_XL_MTP`** file. See
`.scratch/bench-logs/qwen-udt-ab-20260513-132549.md`.

### Bench host

| Component | Value |
|---|---|
| Machine | MacBook Pro (`Mac16,5`, MX313LL/A) |
| SoC | Apple **M4 Max** — 16 CPU cores (12P + 4E), **40-core GPU** |
| Unified memory | **48 GB** LPDDR5 |
| OS | macOS 26.3.1 (build 25D2128), Darwin 25.3.0 |
| llama.cpp backend | Metal (full GPU offload: `-ngl 99 -ngld 99`, `-fa on`) |
| Server | local `llama-server` over `127.0.0.1:8080` |
| Client | `python3 urllib` → `/v1/chat/completions`, `temperature=0`, `cache_prompt=false`, `stream=false` |
| Driver | `scripts/bench-matrix-qwen.sh` (3 runs/cell, median tps, mean accept) |

Single-slot configuration (`--parallel 1 -np 1 --cont-batching`); no other
heavy GPU/CPU workloads were running on the host during the matrix sweep.

| model | mode | short tps (n=128) | long tps (n=512) | short accept | long accept | Δ short | Δ long |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen-27B dense | f16-base       | 21.34 | 20.82 | — | — | — | — |
| qwen-27B dense | f16-nextn      | **22.86** | **21.57** | 93.9% | 85.1% | **+7.1%** | **+3.6%** |
| qwen-27B dense | turbo3-base    | 19.71 | 18.74 | — | — | — | — |
| qwen-27B dense | turbo3-nextn   | **20.75** | **19.73** | 85.5% | 78.7% | **+5.3%** | **+5.3%** |
| qwen-35B-A3B MoE | f16-base     | 70.09 | 69.63 | — | — | — | — |
| qwen-35B-A3B MoE | f16-nextn    | **95.22** | **89.13** | 88.2% | 78.7% | **+35.8%** | **+28.0%** |
| qwen-35B-A3B MoE | turbo3-base  | 61.84 | 62.01 | — | — | — | — |
| qwen-35B-A3B MoE | turbo3-nextn | **82.73** | **77.20** | 82.9% | 80.6% | **+33.8%** | **+24.5%** |

**Where NextN helps the most: MoE targets (qwen-35B-A3B).** Verify is heavy enough that the
draft compute fully overlaps via the async pipeline; acceptance stays high (≥78%) at both
prompt lengths. Wins range from **+24% (turbo3, long)** to **+36% (f16, short)**, on top of
the +13% TurboQuant memory-bandwidth lift from `turbo3` KV.

**Dense 27B is draft-compute-bound but no longer regresses.** The NextN-layer is a full
transformer block; on a dense model `t_draft ≈ 2.6× t_verify`, so the async pipeline cannot
overlap it fully and the upside is bounded by accept-rate × `(t_verify / (t_verify + non-overlapped t_draft))`.
With the shared-model draft path (no double mmap, no graph rebuilds across submits) we land
at **+5-7% across short/long, both KV typings** — modest but consistent, and *positive*
where the previous double-mmap path was negative (the old `qwen-matrix-shared` matrix logged
−7.6% / −11.9% on long for f16-nextn / turbo3-nextn respectively). `turbo3` KV adds ~5% extra
draft compute on this rig (Metal dequant inside NextN attention) but it is hidden in the
overlap and TurboQuant's bandwidth win covers the rest.

### History within this branch (27B regression resolved)

| Bench log (mtime) | Path | 27B f16-nextn long (Δ vs f16-base) | 27B turbo3-nextn long (Δ vs turbo3-base) | Note |
|---|---|---:|---:|---|
| `qwen-matrix-shared-20260512-202358.md` | double mmap | −7.6 % (18.93 vs 20.49) | −11.9 % (15.72 vs 17.85) | 35B-A3B OOM on long prompts |
| `qwen-matrix-fullrun-20260512-222625.md` | shared model | **+3.6 % (21.57 vs 20.82)** | **+5.3 % (19.73 vs 18.74)** | this matrix |

The jump came from a single architectural change: dropping the second
`llama_model_load_from_file` and reusing the target's already-loaded NextN tensors via
`cparams.nextn_draft = true`. Side-effects: (a) 22 GB second `MTLBuffer` gone — 35B-A3B MoE
now runs without OOM and posts +24-36%; (b) draft KV cache resized only for the NextN layer
(`kv_only_nextn = true` is mutated transparently in `llama_context` ctor for draft); (c) the
NextN graph builder now flows through `LLM_GRAPH_TYPE_NEXTN` instead of `override_arch`.

---

## 8. UDT quantization recipe (calibration + masks)

**Goal:** keep Unsloth’s **MTP-aware imatrix** (public `imatrix_unsloth.gguf_file` per HF repo) while applying **AtomicChat-specific** `--tensor-type-file` overrides:

| File | Extra tensors vs base |
|------|-------------------------|
| `scripts/quantize-masks/qwen36-ud-base.txt` | `token_embd` / `output` high bit width; `attn_v` / `ffn_down` lifted; `ffn_gate_inp` for MoE |
| `qwen36-ud-v1-nextn.txt` | All `blk.*.nextn.*` and `mtp.*` at `q8_0` (draft-head preservation) |
| `qwen36-ud-v2-turbo3.txt` | `attn_q` / `attn_k` at `q6_K` (stack with TurboQuant3 KV) |
| `qwen36-ud-v3-combined.txt` | Union of v1 + v2 (default release build) |

**Build entrypoints**

- Single quant: `scripts/quantize-qwen-udt.sh`
- Full sweep: `scripts/quantize-qwen-udt-matrix.sh`
- Remote / bench / HF: **[docs/qwen-udt/RUNBOOK.md](../docs/qwen-udt/RUNBOOK.md)**

**Note:** `UDT` filenames use `…Q4_K_XL…` as a product tag; `llama-quantize` is still invoked with family types `Q4_K_M`, `Q5_K_M`, etc.

---

## 9. Released artifacts — AtomicChat UDT collection

The recipe above ships as two ready-to-pull Hugging Face repos, grouped into one collection:

- Collection — [AtomicChat — Qwen 3.6 UDT](https://huggingface.co/collections/AtomicChat/qwen-36-udt-atomicchat-6a0481f5cc5a057c07759176)
- 27B dense — [`AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF)
- 35B-A3B MoE — [`AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`](https://huggingface.co/AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF)

What's actually in each repo, and why it's a bit unusual for a quant drop:

- **5 quants per model, all `_MTP.gguf`** — `Q3_K_XL` / `Q4_K_XL` / `Q5_K_XL` / `Q6_K` / `Q8_K_XL`. Every file already includes the NextN auxiliary head, so the same path works for `-m` *and* `-md` — no second GGUF, no second mmap, no second tokenizer.
- **NextN-preserve mask (V1)** — every `blk.*.nextn.*` and `mtp.*` tensor pinned to `Q8_0`. The cost is ~10 MiB of file size; the win is that the draft head stays close to BF16 fidelity, which keeps `acceptance` high under `--spec-type nextn`. Plain UD quants compress the head at the same bit-width as the body and bleed acceptance under `turbo3` KV.
- **TurboQuant3-friendly mask (V2)** — attention Q/K bumped to `Q6_K`. This is the piece we tuned specifically for this fork: when KV is compressed to 3-bit via `-ctk turbo3 -ctv turbo3`, the attention scores see extra dequant noise on K, so giving Q/K a little more headroom on the weight side cancels most of it out.
- **Default release = V3 (V1 ∪ V2)** — the combined mask shipped on Hugging Face. V1-only and V2-only quants exist as ablation artifacts in the build tree but are not published; the V3 file simply has both lifts at once.
- **mmproj mirrored from Unsloth** — `mmproj-F16.gguf` and `mmproj-BF16.gguf` re-hosted byte-for-byte from the corresponding `unsloth/Qwen3.6-*-MTP-GGUF` repo so a single `-hf` line gets you target + draft + projector.
- **`imatrix_unsloth.gguf_file` re-hosted** — same artifact as Unsloth's (77-chunk, MTP-aware), included in each repo so the build is reproducible from a clean clone of the recipe.
- **Apache-2.0**, attribution: Qwen team (weights), Unsloth (imatrix + BF16 sources), [@TheTom](https://github.com/TheTom) (TurboQuant), AtomicChat (UDT masks + packaging). Fork: [`AtomicBot-ai/atomic-llama-cpp-turboquant`](https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant).

The whole pipeline (download → quantize on H100 → bench on M4 Max → upload) is scripted in [`docs/qwen-udt/RUNBOOK.md`](../docs/qwen-udt/RUNBOOK.md); re-running it on the same Unsloth sources reproduces the published files bit-identical.

---

## 10. Multimodal (`--mmproj`) + speculative decoding (this fork)

Upstream `llama-server` used to disable **all** speculative modes whenever a projector was loaded, so a single Qwen 3.6 / Gemma 4 server could not host vision and a draft head at the same time. In **atomic-llama-cpp-turboquant** the load-time and slot-init gates accept `--mmproj` together with:

- **`--spec-type mtp`** (Gemma 4 assistant)
- **`--spec-type nextn`** (Qwen3 NextN draft context)
- **`--spec-type eagle3`** (stub impl; same contract)

These three never look at the flattened `prompt_tgt` token stream — they read target hidden states / KV directly — so they can coexist with mtmd image chunks. Other modes stay disabled with a warning: separate **`draft`** models, all **`ngram_*`** modes, **`ctx_shift`** and **`cache_reuse`**.

### What is and is not accelerated today

- **Text-only turns on a multimodal slot** — draft head runs as usual. Same acceptance rates as the no-mmproj configuration.
- **Turns that contain an image chunk** — server logs `skipping speculative prime for multimodal prompt` and falls back to plain target decoding for **that turn only**. The slot keeps generating text correctly, just without draft speedup.

The reason for the fallback: NextN / MTP `begin()` needs the target's pre-norm hidden state at every prompt position, but the mtmd image-decode path only writes outputs for the last row of an image batch (`get_embeddings_pre_norm_ith` returns `null` for image-pad positions, see `tools/server/server-context.cpp`). Until image chunks emit per-token outputs, priming on a mixed token stream would leave the draft KV partially seeded and desynced from the target by image-expanded positions. Skipping the prime keeps the slot stable and lets the next pure-text turn re-enable drafting from scratch.

### Verified configurations

| Model | Spec | KV | mmproj | Image | Text reply | Decode |
|---|---|---|---|---|---|---|
| Qwen 3.6-35B-A3B-UDT-Q4_K_XL_MTP | `nextn` | turbo3 | F16 | recognised | OK | ~69 t/s |
| Gemma 4-26B-A4B-it-UD-Q4_K_XL    | `mtp`   | turbo3 | F16 | recognised | OK | ~55 t/s |

Both runs were validated on M4 Max with a single shared model file (no second mmap), `-c 4096`, `-fa on`.

### Roadmap

Real draft acceleration on the vision turn itself requires making mtmd image batches emit per-token outputs (or a teacher-forced replay through target). Tracked as a follow-up; not blocking this fork's release.
