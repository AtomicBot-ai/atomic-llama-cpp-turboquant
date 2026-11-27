# Inkling target-only decode on RTX Spark

Set `LLAMA_EXPERIMENTAL_SPARK_TARGET_FUSION=1` before starting the server to enable the F07 target-only path. The benchmark also enabled `LLAMA_EXPERIMENTAL_SPARK_SHORT_PROMPT_POOL=1` on both sides. The default is off. The measured configuration used Inkling 2 Mini NVFP4 on Windows ARM64, one RTX Spark N1X, full GPU offload, one slot, flash attention, Q4_0 KV, 4,096 context and a 20,580 MiB engine budget.

The graph selects the new CUDA operations for a single decode token and output with all expert banks resident. It packs depth rows before the original normalization, pairs independent RMS reductions, and combines K/V and FFN convolution with staging the next recurrent state. Attention-output convolution retains its original arithmetic. During eligible graph captures, the backend also folds clipping and two rank-64 controller projections, preserving the reduction order. Unsupported shapes and placements fall back to existing operations. The speculative target uses its existing graph.

## Normal chat results

Six conversations at two seeds, temperature 1, natural EOS and a 1,536-token cap. Each server received an excluded warmup. Decode rate pools committed updates across the twelve requests. TTFT is the median client time to the first streamed reasoning or answer text.

| Engine | Decode tokens/s | TTFT |
|---|---:|---:|
| Atomic target, fusion off | 35.49 | 317 ms |
| Atomic target, F07 | 41.10 | 329 ms |
| Atomic DFlash2, seven proposals, fusion off | 61.75 | 387 ms |
| Atomic DFlash2, seven proposals, F07 switch on | 61.28 | 386 ms |
| Upstream llama.cpp b12810, target only | 46.83 | 261 ms |

In the same experimental executable the target-only gain was **15.83%**. DFlash2 accepted the same proposals and returned the same twelve visible answers with either setting. Busy GPU medians across these runs ranged from 1.42 to 1.48 GHz. The earlier F06 target-only run reached 40.58 tokens/s with 315 ms TTFT. Earlier R08 measurements gave 44.63 / 301 ms for the aligned target, 66.91 / 395 ms for DFlash2 and 50.29 / 279 ms for upstream. Separate sessions used different power conditions.

## Integrated PR build

On the rebuilt Spark server, each mode reproduced its own archived full-vocabulary hashes at all 260 fixture positions and on the repeated 65-token request. The partial-residency fallback at 16,484 MiB matched all eight archived positions. A DFlash2 smoke accepted 18 of 82 proposals.

One clean off/on/on/off block with fixed short and long continuations measured **44.06 → 49.70 decode updates/s (+12.80%)**. Each setting contributed 256 measured updates. A single GPU reading during the block was 2,522 MHz at 89% utilization. The normal-chat table above comes from the earlier 12-request benchmark. [Integrated build checks and binary hashes](atomic-engine-inkling-rtx-spark-target-fusion.json)

## Evidence

- [F07 serving rerun, upstream comparison and per-request timings](../research/atomic-fusion-F07-serving-2026-11-27.zip)
- [Fusion experiments, fixed-token ABBA runs, patches and numerical checks](../research/atomic-fusion-F00-F03-2026-11-27.zip)
- [Earlier speculative-decoding research, S00–S02](../research/atomic-spec-S00-S02-2026-11-27.zip)
- [Pivot experiments and reference data, P00–P05](../research/atomic-spec-pivot-P00-P05-2026-11-27.zip)
