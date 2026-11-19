# LFM and Qwen on Metal at 8–24 GiB

Apple M4 Max, 48 GiB unified memory. Each cell is mean client TTFT in seconds / decode tokens per second for two uncached 4000-token prompts with 64 output tokens. Each configuration used a fresh process and a short warmup. Startup is excluded. [Per-request measurements, commands and memory evidence](atomic-engine-four-engines-metal-8-24gib-gqa8.json).

## Atomic Engine

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | 2.46 / 33.8 | 3.47 / 29.0 |
| 10 | 2.29 / 39.9 | 2.74 / 49.5 |
| 12 | 2.15 / 47.4 | 2.89 / 51.6 |
| 14 | 2.19 / 59.6 | 2.93 / 54.2 |
| 16 | 2.30 / 78.3 | 3.09 / 68.0 |
| 18 | 2.40 / 103.1 | 3.27 / 72.1 |
| 20 | 2.34 / 127.0 | 3.26 / 83.2 |
| 22 | 2.30 / 125.0 | 3.26 / 92.6 |
| 24 | 2.45 / 125.0 | 3.29 / 117.9 |

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
| 8 | Unsupported | 2.11 / 67.6 |
| 10 | Unsupported | 2.28 / 73.3 |
| 12 | Unsupported | 2.20 / 79.1 |
| 14 | Unsupported | 2.18 / 82.2 |
| 16 | Unsupported | 2.04 / 104.9 |
| 18 | Unsupported | 2.09 / 149.2 |
| 20 | Unsupported | 2.06 / 266.2 |
| 22 | Unsupported | 1.93 / 311.6 |
| 24 | Unsupported | 1.84 / 346.8 |

## llama.cpp (upstream)

| Budget (GiB) | LFM2.5-24B-A2B | Qwen3.6-35B-A3B |
|---:|---:|---:|
| 8 | Over cap | Over cap |
| 10 | Over cap | Over cap |
| 12 | Over cap | Over cap |
| 14 | Over cap | Over cap |
| 16 | 3.58 / 126.6 | Over cap |
| 18 | 3.49 / 132.0 | Over cap |
| 20 | 3.32 / 134.5 | 4.35 / 97.0 |
| 22 | 3.51 / 129.8 | 4.24 / 150.6 |
| 24 | 3.25 / 134.2 | 4.24 / 151.6 |

The 24 GiB Atomic Engine Qwen cell uses the [experimental grouped BF16 attention patch](../tools/expert-source/qwen35-bf16-verify-gqa8.patch). The stock server measured 3.33 / 110.3 on the same fixture. Both used 36 direct layers and the BF16 DFlash2 sidecar. [Full-vocabulary comparison](atomic-engine-qwen35-bf16-verify-gqa8-24gib.json).

A budget denial is a loader or allocation refusal. Over cap means an external memory check failed for the tested configurations. Both RSS and physical footprint were monitored, including the lifetime physical-footprint high water mark. For Splash the worker and HTTP process peaks were added conservatively. The OS file cache was uncontrolled.

Splash rejected the LFM architecture during model preparation. The tested Strata Metal component rejected both input GGUFs during model selection. For llama.cpp full Metal placement was tried first, with CPU MoE placement after a cap crossing and a DFlash2 comparison when Qwen fit on Metal.
