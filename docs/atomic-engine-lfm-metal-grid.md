# LFM resident Metal layers and memory-budget sweep

On November 18, 2026, the LFM2.5-24B-A2B Q4_K_M checkpoint ran on an Apple M4 Max using one slot, a 4096-token context, BF16 KV and eight CPU threads. Each successful table entry shows client time to first token in seconds followed by decode tokens per second. It uses the first full request after a short warmup. Values for the second full request and exact input and output IDs are in the [measurement record](atomic-engine-lfm-metal-grid.json).

| Budget (GiB) | Resident MoE layers | Atomic Engine hybrid | Atomic Engine streaming | Upstream llama.cpp |
|---:|---:|---:|---:|---:|
| 8 | 7/38 | 1.92 / 33.8 | 2.00 / 31.6 | over cap |
| 10 | 13/38 | 2.02 / 38.9 | 2.06 / 33.8 | over cap |
| 12 | 19/38 | 2.40 / 46.3 | 2.63 / 35.4 | over cap |
| 14 | 25/38 | 2.71 / 55.0 | 2.90 / 33.4 | over cap |
| 16 | 31/38 | 2.77 / 69.3 | 3.11 / 34.1 | 3.22 / 133.7 |
| 18 | 36/38 | 2.53 / 98.8 | 3.25 / 35.8 | 3.24 / 132.7 |
| 20 | 38/38 | 2.45 / 125.8 | 2.44 / 33.6 | 3.24 / 132.0 |
| 22 | 38/38 | 2.42 / 114.1 | 2.59 / 35.1 | 3.51 / 125.9 |
| 24 | 38/38 | 2.40 / 121.6 | 2.44 / 33.8 | 3.78 / 127.3 |

## Second full request in the same process

The prompt/KV cache was disabled for both requests. Expert and OS file-cache contents can differ between requests. Values again show TTFT in seconds and decode tokens per second.

| Budget (GiB) | Atomic Engine hybrid | Atomic Engine streaming | Upstream llama.cpp |
|---:|---:|---:|---:|
| 8 | 1.97 / 23.4 | 2.03 / 32.2 | over cap |
| 10 | 2.83 / 40.3 | 2.05 / 33.9 | over cap |
| 12 | 2.85 / 47.3 | 3.05 / 33.8 | over cap |
| 14 | 3.68 / 39.8 | 4.14 / 20.3 | over cap |
| 16 | 4.30 / 59.3 | 3.95 / 34.2 | 3.22 / 133.1 |
| 18 | 3.21 / 92.7 | 3.54 / 35.9 | 3.57 / 132.0 |
| 20 | 2.80 / 82.7 | 3.26 / 34.1 | 3.33 / 129.9 |
| 22 | 2.93 / 85.8 | 3.29 / 35.2 | 3.77 / 130.8 |
| 24 | 2.47 / 80.9 | 2.88 / 28.3 | 3.80 / 129.7 |

## Reading the results

All 38 routed layers first fit at 20 GiB. At that budget the first hybrid request reached 125.8 tokens/s and the second reached 82.7. At 24 GiB the pair was 121.6 and 80.9. The wide prefill path still stages its own layer weights even when decode banks are resident. At 24 GiB the process recorded 14.67 and 14.74 GB of disk reads over the two complete requests. This counter spans prefill and decode. The measurement does not isolate the cause of the second-request slowdown.

The previous restored-slot experiment measured 56 tokens/s at 12 GiB and about 133–135 at 24 GiB. The HTTP grid starts each full request by processing all 4000 input tokens. At 12 GiB the two hybrid requests reached 46.3 and 47.3 tokens/s.

## Configuration and lifetime

Set `LLAMA_LFM_GQA4=1` on the managed Metal server to try grouped attention
for single-token decode on LFM2.5-24B-A2B. The kernel shares BF16 K/V reads
across four query heads and uses FP32 matrix operations. It selects the
32-query-head, 8-KV-head, width-64 shape with 4096 physical KV rows on Apple
GPU family 9. Prefill and expert residency follow the selected budget profile.
The [GQA4 comparison](atomic-engine-lfm-gqa4-metal-8-24gib.json) includes
client timings and full-vocabulary logit comparisons at nine budgets.

Automatic Metal streaming with `--engine-budget-mib` uses resident layers by default on the qualified LFM geometry. Set `LLAMA_LFM_RESIDENT_LAYERS=0` for the streaming-only control or `1` to select resident layers explicitly. The grid used explicit mode selection before the default changed. The context admits complete layers in model order, leaving 2 GiB above the fixed working floor for scratch and the remaining streaming path. All three original GGUF expert projections are loaded into a charged Metal buffer before the graph can reference the layer. Each context owns its tensors. The ordinary graph can bind those tensors directly. The existing wide prompt window retains its own layer bindings.

At a live shrink, the scheduler synchronizes and discards its reusable graphs before releasing complete layers from the end of the resident prefix. Automatic LFM streaming reclaims idle cache entries for the previous combined MoE workspace and its demand working set when the target permits. Growth increases the available streaming budget. Additional direct layers are selected at context creation. The 2 GiB headroom and layer order come from the tested prototype and have not been optimized across texts.

## Protocol and limits

Each cap and engine starts a fresh local server. After model load it processes 15 input IDs and eight output tokens as warmup, then twice processes all 4000 input IDs with 64 greedy output tokens, seed 42, EOS ignored and prefix reuse disabled. TTFT starts at the HTTP request and ends at the first token event. Decode uses 63 intervals between 64 emitted tokens. Model load and initial bank admission precede TTFT. The OS file cache was uncontrolled. Processes ran sequentially. Atomic Engine alternated hybrid/streaming order between caps.

Atomic Engine used `-c 4096 -b 4096 -ub 8 -np 1 -t 8 -tb 8 -ngl 99 -ctk bf16 -ctv bf16 -fit off --cache-ram 0 --ctx-checkpoints 0 --no-warmup`. The engine ledger was checked after both requests. The clean upstream server used the same settings with `-ub 128` and full Metal offload. Its external monitor sampled resident bytes and physical footprint every 50 ms and stopped the process upon exceeding the cap. These caps have different accounting semantics. Upstream low-budget entries describe the selected full-offload configuration. Other placements were outside this comparison.

The installed Splash 1.6.1 GGUF preparer rejected `lfm2moe`. The Strata 0.1.59-r5 Metal native binary rejected the same GGUF during startup. Its Metal bundle lists Qwen3.8 and DeepSeek4.1. Neither returned tokens for this model.

A separate production-library probe restored a common 3999-token slot, processed the final input token and applied a 24-to-12 GiB request after sampling output 32. For each Atomic Engine mode, 64 complete FP32 logit rows before the shrink, through the transition and after it matched the direct Metal reference byte for byte. The hybrid went from 38 direct layers to 19. The streaming control retained combined MoE across the transition. These checks cover one checkpoint and input. The per-cap HTTP sweep checked output IDs and engine high-water values.
