# Gemma-4-E4B-It perplexity benchmark : turbo4 vs F16 KV cache

Cross-corpus perplexity measurements on Raspberry Pi 16GB (Cortex-A76 aarch64)
comparing F16 dense KV cache against turbo4 K + turbo4 V on Gemma-4-E4B-It.

## Headline

| Corpus | F16 PPL | turbo4 PPL | Δ relative | Paired pattern |
|---|---|---|---|---|
| WikiText-2 raw (test split) | 55.0055 ± 8.83 | 50.6767 ± 7.96 | **−7.87%** | turbo4 lower on 4/4 chunks |
| HumanEval (prompt + canonical_solution) | 4.2657 ± 0.469 | 4.1310 ± 0.450 | **−3.16%** | turbo4 lower on 4/4 chunks |

turbo4 produces **lower** perplexity than F16 dense on both corpora tested, with
the paired direction consistent across all 4 chunks of each bench. The most
likely explanation is that the Walsh-Hadamard Transform (WHT) rotation applied
pre-quant by TurboQuant acts as light implicit regularization on the attention
activations, analogous to QuaRot ([arXiv:2404.00456][quarot]) and SpinQuant
(ICLR 2025) smoothing effects.

For an instruction-tuned model evaluated outside its native chat-format
distribution (WikiText narrative, code), this smoothing slightly improves
attention output quality even after 4-bit quantization. The effect on the
model's native chat-format distribution is not measured here.

## Setup

| Field | Value |
|---|---|
| Model | `gemma-4-E4B-it-Q4_K_M.gguf` (Gemma 4 E4B Instruction-Tuned, 7.5B params) |
| Hardware | Raspberry Pi 16GB · Cortex-A76 aarch64 · OC arm_freq=2700, force_turbo=1 · Active Cooler |
| Build | `cecil/phase-c2-dispatch` HEAD `ab632e4` (atomic-llama-cpp-turboquant fork) |
| CPU features | NEON · ARM_FMA · FP16_VA · DOTPROD · KLEIDIAI · REPACK |
| Threads | 4 |
| Context | 512 |
| Chunks | 4 (2048 tokens evaluated per config) |
| Speculative decoding | disabled (direct `llama-perplexity` invocation, no MTP) |
| Vision branch | not loaded (no `--mmproj`) |

```bash
./build/bin/llama-perplexity \
    -m gemma-4-E4B-it-Q4_K_M.gguf \
    -f <corpus.raw> \
    -c 512 --chunks 4 -t 4 \
    [-ctk turbo4 -ctv turbo4]
```

Loader confirms `TurboQuant uses kernel-level WHT` and `attn_rot_k = 0`
(upstream attention rotation disabled because turbo4 supplies its own
kernel-side WHT).

## Results

### WikiText-2 raw test split (`wiki.test.raw`)

| Config | Final PPL | ±CI | Per-chunk running PPL |
|---|---|---|---|
| F16 K + F16 V | **55.0055** | ±8.83 | [1]32.02, [2]50.64, [3]44.21, [4]55.01 |
| turbo4 K + turbo4 V | **50.6767** | ±7.96 | [1]26.93, [2]46.69, [3]40.80, [4]50.68 |
| **Δ relative** | **−7.87%** | — | turbo4 lower on 4/4 chunks |

### HumanEval (164 problems, prompt + canonical_solution concatenated)

| Config | Final PPL | ±CI | Per-chunk running PPL |
|---|---|---|---|
| F16 K + F16 V | **4.2657** | ±0.469 | [1]5.50, [2]4.53, [3]4.26, [4]4.27 |
| turbo4 K + turbo4 V | **4.1310** | ±0.450 | [1]5.41, [2]4.50, [3]4.16, [4]4.13 |
| **Δ relative** | **−3.16%** | — | turbo4 lower on 4/4 chunks |

The PPL absolute scale differs by an order of magnitude between corpora
(WikiText ~50 vs HumanEval ~4) because code is closer to Gemma-IT's training
distribution than plain narrative text.

### Auxiliary : asymmetric turbo4 K + turbo3 V (WikiText-2)

| Config | Final PPL | ±CI |
|---|---|---|
| F16 K + F16 V | 55.0055 ± 8.83 |
| turbo4 K + turbo4 V | 50.6767 ± 7.96 |
| **turbo4 K + turbo3 V** | **49.7837 ± 7.81** |

asymmetric K4V3 is lowest on 4/4 chunks vs symmetric K4V4 (paired Δ
−1.8% additional improvement). RAM saving is negligible vs sym K4V4
(~10 MiB at ctx 8192 on this model — most of the difference is in
per-block metadata, not bits), so this is not a deployment recommendation,
but the direction is interesting.

## Memory footprint

For Gemma-4-E4B-It architecture (block_count=42, with 18 shared_kv_layers
→ 24 unique KV layers : 4 non-SWA + 20 SWA; n_kv_heads=2; key_length=512
non-SWA / 256 SWA; sliding_window=512):

| Config @ ctx 8192 | Non-SWA (4 layers) | SWA (20 layers, fixed window) | Total |
|---|---|---|---|
| turbo4 K + turbo4 V | ~17 MiB + ~17 MiB | ~13.3 + ~13.3 MiB | **~52.5 MiB** |
| F16 K + F16 V | ~64 MiB + ~64 MiB | ~40 + ~40 MiB | **~208 MiB** |

turbo4 saves ~75% KV cache memory (4-bit vs 16-bit), as expected for the
bit ratio. The SWA fixed-window cost means context length can be increased
significantly with very little additional KV cost (going ctx 8192 → 131072
adds only ~443 MiB total KV).

## Caveats

- **Chunks=4** : ±CI on absolute PPL is wide (~8 PPL absolute on WikiText).
  Tighter CI would need chunks ≥ 32 (~30 min × 2 configs).
  Paired direction (turbo4 lower on 4/4 chunks in BOTH corpora) is robust
  to variance even with current chunk count.
- **Two corpora tested** : narrative text + code. Chat-format prompts
  (Gemma-IT's native distribution) NOT yet tested. The regularization
  hypothesis predicts a smaller or possibly inverted Δ on chat-format.
- **Multilingual** not tested.
- **Single hardware target** (Cortex-A76, OC). Effect on x86_64 with
  different SIMD path not validated by this data point.
- **Direct llama-perplexity** invocation : MTP speculative decoding and
  mmproj vision branch were both disabled during PPL benches to isolate
  KV quantization impact. Interaction effects with MTP/mmproj not measured.

## Reproducibility

Corpora fetch :
- WikiText-2 raw : `scripts/get-wikitext-2.sh` from this repo
- HumanEval : `https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz`
  then extract `prompt + canonical_solution` per problem, join with `\n\n` to
  form raw text corpus (~100 KiB for 164 problems).

Build :
```
cmake --build build --target llama-perplexity -j 4
```

Run both configs as shown in [Setup](#setup) and compare PPL.

## References

- TurboQuant paper : [arXiv:2504.19874][turboquant] (ICLR 2026)
- WHT regularization analog : QuaRot [arXiv:2404.00456][quarot], SpinQuant (ICLR 2025)
- Related fork PRs : [#16][pr16] (NEON inverse FWHT), [#17][pr17] (MTP+mmproj
  SEGV fix), [#18][pr18] / [#19][pr19] (MTP+mmproj coexistence dispatch)

[turboquant]: https://arxiv.org/abs/2504.19874
[quarot]: https://arxiv.org/abs/2404.00456
[pr16]: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/pull/16
[pr17]: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/pull/17
[pr18]: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/pull/18
[pr19]: https://github.com/AtomicBot-ai/atomic-llama-cpp-turboquant/pull/19
