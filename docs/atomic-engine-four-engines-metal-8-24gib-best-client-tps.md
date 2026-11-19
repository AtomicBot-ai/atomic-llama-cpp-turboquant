# LFM and Qwen on Metal at 8–24 GiB

Apple M4 Max, 48 GiB unified memory. Cells report TTFT in seconds / client decode tokens per second for uncached 4000-token prompts and 64-token outputs. Client TPS includes any pause to load weights after the first token. The Qwen Atomic Engine cell at 24 GiB was rerun with four timed requests per strategy. The other cells come from the previous four-engine run. [Current run record](atomic-engine-four-engines-metal-8-24gib-best-client-tps.json). [Previous run record](atomic-engine-four-engines-metal-8-24gib-gqa8.json).

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
| 24 | 2.45 / 125.0 | 3.23 / 125.2 |

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

## Atomic Engine Qwen placement at 24 GiB

The matrix uses **36 direct + 4 streamed**, the highest observed mean client TPS on this request. All variants generated the same 64 output IDs and passed the 24 GiB cap.

| Strategy | TTFT (s) | Client TPS | Min–max TPS | Load pause (ms) |
|---|---:|---:|---:|---:|
| 36 direct + 4 streamed | 3.23 | 125.2 | 124.7–126.0 | 0 |
| 40 direct, reuse 2 prefill banks and load 2 | 3.24 | 120.9 | 109.9–125.0 | 107 |
| 40 direct, reuse 2 and fully overwrite 2 new banks | 3.25 | 124.4 | 123.7–124.7 | 109 |

The [phase experiment](atomic-engine-qwen35-phase-resident-24gib.json) also records the full four-layer reload and the full-vocabulary logit comparison. The tested Splash build rejected LFM; the tested Strata Metal build rejected both GGUFs. Over cap denotes a failed external memory check for the tested placement.
