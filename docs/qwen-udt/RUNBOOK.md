# Qwen 3.6 UDT (UD-Turbo) — runbook

This runbook covers **remote CUDA quantization**, **local Metal throughput benches**, and **Hugging Face release** for the AtomicChat `UDT` GGUF line. It implements the mask variants:

| Variant | Mask file | Intent |
|--------|-----------|--------|
| `base` | `scripts/quantize-masks/qwen36-ud-base.txt` | Reproduce Unsloth-style imatrix + selective high-bit tensors |
| `v1` | `scripts/quantize-masks/qwen36-ud-v1-nextn.txt` | Bump NextN / MTP tensors to `q8_0` (acceptance-focused) |
| `v2` | `scripts/quantize-masks/qwen36-ud-v2-turbo3.txt` | Bump `attn_q` / `attn_k` to `q6_K` (TurboQuant3 KV stack) |
| `v3` | `scripts/quantize-masks/qwen36-ud-v3-combined.txt` | Union of v1 + v2 (default release recipe) |

**Naming:** filenames use `...UDT-Q4_K_XL...` while `llama-quantize` is invoked with base family types `Q4_K_M` (the `XL` token denotes the extra tensor-type-file overrides, matching Unsloth’s naming style).

**Attribution:** model weights follow the Qwen license; calibration uses Unsloth’s public `imatrix_unsloth.gguf_file` from each HF repo; masks and tooling are from this fork.

---

## 0. Layout

| Path | Role |
|------|------|
| `.scratch/qwen-ud-sources/27b/` | 27B BF16 shards + `imatrix_unsloth.gguf_file` + optional reference `UD-Q4_K_XL.gguf` |
| `.scratch/qwen-ud-sources/35a3b/` | Same for 35B-A3B |
| `.scratch/qwen-udt-quants/` | Output GGUFs |
| `.scratch/quant-logs/` | Matrix quantization logs |
| `.scratch/bench-logs/` | Local bench markdown |

---

## 1. Remote host (Ubuntu + CUDA)

```bash
ssh ubuntu@192.222.54.232
git clone --depth 1 --branch master https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant.git
cd atomic-llama-cpp-turboquant
bash scripts/qwen-udt/remote-bootstrap.sh
```

Optional: set `REPO_URL` / `REPO_BRANCH` / `DEST` before running `remote-bootstrap.sh` on a machine **without** an existing checkout (see script header).

Log in for downloads:

```bash
huggingface-cli login
```

### 1.1 Download BF16 + imatrix

From the repo root on the remote:

```bash
bash scripts/qwen-udt/hf-download-sources.sh
```

If `huggingface-cli` rejects `--include`, download `BF16/` manually from the Hugging Face UI into `.scratch/qwen-ud-sources/{27b,35a3b}/BF16/`.

### 1.2 Single quant

```bash
export LLAMA_QUANTIZE="$PWD/build/bin/llama-quantize"
export QWEN_UDT_SOURCES_DIR="$PWD/.scratch/qwen-ud-sources"
export QWEN_UDT_OUT_DIR="$PWD/.scratch/qwen-udt-quants"
./scripts/quantize-qwen-udt.sh 27b Q4_K_M v3
```

### 1.3 Full matrix (32 jobs: 2 models × 4 ftypes × 4 variants)

```bash
unset IMATRIX_FILE BF16_INPUT
export QWEN_UDT_SKIP_BASE=0   # set to 1 after sanity passes to save disk/time
./scripts/quantize-qwen-udt-matrix.sh
```

### 1.4 Sanity (27B Q4 base vs Unsloth reference)

```bash
./scripts/qwen-udt/run-sanity-q4-27b.sh
```

Compare reported perplexity and file size. Note: the reference `UD-Q4_K_XL.gguf` may be **non-MTP** while the reproduced artifact is **`*_MTP.gguf`** — small PPL deltas are expected.

---

## 2. Copy artifacts to local Mac (Metal)

```bash
export REMOTE=ubuntu@192.222.54.232
export REMOTE_DIR='~/atomic-llama-cpp-turboquant/.scratch/qwen-udt-quants'
bash scripts/qwen-udt/rsync-pull-quants.example.sh
```

---

## 3. Local throughput matrix

Build `llama-server` locally (Metal). Then:

```bash
export BENCH_MATRIX_MD="$PWD/.scratch/bench-logs/qwen-udt-matrix-$(date +%Y%m%d).md"
mkdir -p "$(dirname "$BENCH_MATRIX_MD")"
export QWEN_UDT_BENCH_DIR="$PWD/.scratch/qwen-udt-quants"
bash scripts/bench-qwen-udt-matrix-local.sh
```

Ablation subsets (optional):

```bash
export BENCH_MODES_FILTER='f16-nextn,turbo3-nextn'   # v1 focus
bash scripts/bench-matrix-qwen.sh   # with QWEN*_MTP paths set manually
```

---

## 4. Perplexity (quality)

```bash
sh scripts/get-wikitext-2.sh
export WIKI_FILE="$PWD/wikitext-2-raw/wiki.test.raw"
./scripts/bench-qwen-udt-quality.sh \
  .scratch/qwen-udt-quants/Qwen3.6-27B-UDT-Q4_K_XL_MTP.gguf
```

Append the printed table into `BENCH_MATRIX_MD` by hand or with `tee -a`.

---

## 5. Release decision

Copy the template into `.scratch` (gitignored) for local edits, then fill after benches:

```bash
mkdir -p .scratch/bench-logs
cp docs/qwen-udt/release-decision.md .scratch/bench-logs/qwen-udt-release-decision.md
```

Edit `.scratch/bench-logs/qwen-udt-release-decision.md` (or edit [release-decision.md](./release-decision.md) in-repo and sync).

---

## 6. Hugging Face upload

Create empty model repos (if they do not exist):

- `AtomicChat/Qwen3.6-27B-UDT-MTP-GGUF`
- `AtomicChat/Qwen3.6-35B-A3B-UDT-MTP-GGUF`

Add `README.md` from [release/qwen-udt/MODEL_CARD_TEMPLATE.md](../../release/qwen-udt/MODEL_CARD_TEMPLATE.md). Then:

```bash
huggingface-cli login
./scripts/qwen-udt/hf-upload-qwen-udt.sh /path/to/quants
```

Create an HF **Collection** in the UI grouping both repos.

---

## 7. MoE mask follow-up

If 35B-A3B shows router regressions, inspect tensor names (`llama-gguf-dump` / `gguf-py`) and extend `qwen36-ud-base.txt` with additional `ffn_*` or expert-specific overrides. This is expected to be an iterative step after the first 35B bench pass.
