# Two MoE models across four Metal engines

Apple M4 Max, 48 GiB, November 18, 2026. Each cell gives client TTFT (seconds) / decode tokens per second, averaged across two uncached 4000-input / 64-output requests in one fresh process per budget and configuration. Startup is excluded. The [measurement record](atomic-engine-four-engines-metal.json) contains each request, launch settings, memory peaks and loader errors.

## Atomic Engine

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | 1.85 / 36.5 | 2.61 / 33.6 |
| 10 | 1.86 / 41.6 | 2.66 / 49.3 |
| 12 | 1.89 / 49.1 | 2.88 / 56.0 |
| 14 | 1.87 / 62.9 | 2.89 / 56.3 |
| 16 | 2.69 / 75.8 | 4.45 / 54.8 |
| 18 | 2.25 / 106.4 | 3.55 / 71.5 |
| 20 | 2.04 / 125.8 | 3.54 / 80.2 |
| 22 | 2.51 / 111.5 | 3.57 / 88.2 |
| 24 | 2.59 / 110.6 | 3.62 / 102.2 |

## Strata 0.1.59-r5 Metal

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | Unsupported | Unsupported |
| 10 | Unsupported | Unsupported |
| 12 | Unsupported | Unsupported |
| 14 | Unsupported | Unsupported |
| 16 | Unsupported | Unsupported |
| 18 | Unsupported | Unsupported |
| 20 | Unsupported | Unsupported |
| 22 | Unsupported | Unsupported |
| 24 | Unsupported | Unsupported |

## Splash 1.6.1

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | Unsupported | 2.45 / 124.5 |
| 10 | Unsupported | 2.16 / 87.4 |
| 12 | Unsupported | 2.08 / 99.6 |
| 14 | Unsupported | 2.05 / 124.2 |
| 16 | Unsupported | 2.29 / 126.7 |
| 18 | Unsupported | 1.97 / 189.3 |
| 20 | Unsupported | 1.90 / 239.2 |
| 22 | Unsupported | 2.02 / 302.2 |
| 24 | Unsupported | 1.84 / 348.9 |

## llama.cpp (upstream)

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | Over cap | Over cap |
| 10 | Over cap | Over cap |
| 12 | Over cap | Over cap |
| 14 | Over cap | Over cap |
| 16 | 4.29 / 115.7 | Over cap |
| 18 | 3.77 / 122.1 | Over cap |
| 20 | 3.65 / 126.2 | 4.77 / 93.6 |
| 22 | 4.16 / 115.3 | 5.23 / 90.2 |
| 24 | 3.47 / 128.3 | 4.34 / 97.7 |

Atomic Engine used its production fixed-budget profiles. Splash used its prepared Qwen model and speculative draft. Its LFM preparer rejected the architecture. The tested Strata Metal bundle rejected both input GGUFs at model selection. The Qwen backend reported CUDA/HIP support for this model. Those architecture rejections apply to all budget rows.

For llama.cpp, the full Metal configuration was tried first. If it crossed the cap, a second placement kept part of the MoE experts on CPU. When full-offload Qwen fit, a DFlash2 variant was also tested. For successful variants the table uses the shortest mean time to the last token. The raw record identifies the selected run in each cell.

Displayed speeds passed the external memory check. Atomic Engine also passed its internal ledger check. Splash includes both worker and HTTP process, with a smaller worker ceiling when necessary. RSS and physical footprint were sampled every 50 ms, supplemented by the process lifetime physical-footprint peak. For Splash, corresponding per-process maxima are added conservatively. OS page cache was uncontrolled. The same target GGUF and per-model prompt IDs were used across engines.
