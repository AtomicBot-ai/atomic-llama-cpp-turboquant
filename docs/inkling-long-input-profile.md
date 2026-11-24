# Inkling 2 Mini long-input serving profile (Apple Silicon, non-budget)

Production launch profile for interactive long-input use (Open WebUI at
http://127.0.0.1:3000, llama-server at http://127.0.0.1:8081). All numbers
below were measured on an M5 Max / 64 GB with Docker Desktop running.

## Why the non-budget path

The budgeted Metal profiles (`--engine-budget-mib`) are the fastest option
but carry hard limits, verified at server startup:

- `engine budget profile requires explicit ctx <= 4096, batch <= 4096,
  ubatch <= 1024, one sequence, <=16 threads, no embeddings`
- `engine budget requires -fit off, a supported speculative profile if
  selected, and no multimodal projector`

A 4096-token window cannot serve long documents, and the budget path rejects
the vision projector outright. For long inputs the non-budget Metal path is
the only option, so the tuning below targets that path.

## Launch profile

```bash
llama-server \
  --model models/atomicchat/Inkling-2-Mini-AD-IQ2_M.gguf \
  --mmproj models/atomicchat/mmproj-Inkling-2-Mini-F16.gguf \
  --model-draft models/atomicchat/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 \
  --n-gpu-layers-draft 99 \
  --ctx-size 131072 --parallel 1 --n-gpu-layers all \
  --flash-attn on \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  --batch-size 1024 --ubatch-size 1024 \
  --threads 8 --threads-batch 8 \
  --load-mode mmap+mlock \
  --temp 0 \
  --chat-template-kwargs '{"reasoning_effort":"none"}' \
  --no-reasoning-preserve \
  --metrics --jinja --alias inkling-2-mini \
  --host 127.0.0.1 --port 8081
```

## Measured results (bench-long-prefill.py, warm, cache reuse off)

Prompt eval (prefill) and decode throughput by prompt size. Baseline is the
same build without `--load-mode mmap+mlock`.

| prompt tokens | baseline prefill | tuned prefill | baseline decode | tuned decode |
|---:|---:|---:|---:|---:|
| 2,261 | 726.0 t/s | 743.6 t/s (+2%) | 55.7 t/s | 59.0 t/s (+6%) |
| 14,727 | 539.8 t/s | 591.2 t/s (+10%) | 34.7 t/s | 68.6 t/s (+98%) |
| 35,653 | 364.3 t/s | 437.0 t/s (+20%) | 29.4 t/s | 54.1 t/s (+84%) |

Footprint: 3.36 GiB baseline, 3.74 GiB tuned at 40K (the mlock cost).

## What was tested and rejected

- `--ubatch-size 2048` / `4096`: fails at load. The Metal compute graph
  reservation scales with ubatch (104 GiB requested at 2048) and exceeds the
  device working set. 1024 is the ceiling on a 64 GB machine.
- `-sm row`: `device MTL0 does not support split buffers`.
- `--threads 12 --threads-batch 12`: +3% at 2.5K, nothing at 16.5K/40K.
- Docker Desktop VM memory (36 GiB wired): quitting it did not change
  prefill or decode. The Metal reservation is virtual; committed footprint
  is what matters.
- dev/TurboQuant branch (build 11427): cannot load this model at all
  (`unknown model architecture: 'inkling2_loop'` - the architecture is
  fork-only, present on codex/expert-weight-source). turbo KV cache types
  are therefore unavailable for Inkling 2.
- ngram speculative decoding (`--spec-type ngram-cache,draft-dspark`):
  +24-26% decode on repetitive prompts, but it produces WRONG answers on
  cached multi-turn requests (it proposes the previous turn's answer from
  the cached prefix and the target accepts it). Rejected: correctness on
  the doc-QA workflow outranks the decode gain.

## The remaining prefill ceiling

Fitting T = a*n + b*n^2 to the three measurements gives b ~= 2.85e-8 s/tok^2:
at 35,653 tokens about 44% of prompt-processing time is the quadratic
attention term (the 8 full-attention layers; 40 of 48 layers are sliding
window sw=1024). That term is inherent to causal attention over a long
context; flag tuning cannot remove it. Reordering the benchmark (40K first)
does not change the numbers, so the decay is length-inherent, not thermal.

## Long-document Q&A (the real workflow)

With the default prefix cache (cache_prompt on, as Open WebUI uses), only
the first question pays the full prefill:

| turn | wall time | prompt tokens | cached | answer |
|---|---:|---:|---:|---|
| Q1 (cold, 14.7K doc) | 57.8s | 27,477 | 10 | "The quick brown fox." |
| Q2 (same doc) | 1.6s | 27,503 | 27,473 | "The kettle." (correct) |

Follow-up questions on the same document are 17-36x faster than a cold
prefill and stay correct with draft-dspark alone.

## Quality note

AD-IQ2_M occasionally drops a token in code output (measured: `s[::-]`
instead of `s[::-1]` at effort none/medium; correct at effort low). This is
a quantization artifact of IQ2_M, not a serving-path issue. If code quality
matters more than the 4.3 GiB size difference, AD-IQ3_S (12.42 GiB) is the
drop-in alternative on the same profile.

## Reproduce

```sh
python3 scripts/bench-long-prefill.py 8081 "label" 2500,16500,40000
```

Run each size twice; the first request on a fresh server pays Metal pipeline
compilation. Measure with a single llama-server instance: two concurrent
servers on a 64 GB box drive the machine into Metal OOM
(`Insufficient Memory`), which poisons the backend until restart
(`backend is in error state ... recreate the backend`).
