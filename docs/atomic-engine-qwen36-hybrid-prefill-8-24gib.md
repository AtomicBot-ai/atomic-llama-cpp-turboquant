# Qwen3.6-35B-A3B hybrid prefill on M4 Max

Each cell shows the previous profile followed by the candidate at the same fixed budget. The CPU and GPU share physical memory. The ledger also accounts for reservations, so the two memory peaks measure different quantities.

## Shift handover, 912 input tokens

Previous short prefill versus the latency preset. The 24 GiB latency preset uses all 40 layers and DFlash2. Each arm uses two fresh processes. Timings are independent medians and memory peaks are maxima.

| GiB | Expert layers | TTFT, s | Decode, tok/s | Full request, s | Physical peak, GiB | Ledger peak, GiB | Output IDs |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 2 / 1 | 37.92 / 1.40 | 38.9 / 33.4 | 39.54 / 3.29 | 6.79 / 6.22 | 8.00 / 8.00 | 64/64 match |
| 10 | 4 / 4 | 44.01 / 2.02 | 47.5 / 45.9 | 45.34 / 3.40 | 8.22 / 8.22 | 10.00 / 10.00 | 64/64 match |
| 12 | 9 / 9 | 40.45 / 1.96 | 41.9 / 46.2 | 42.05 / 3.34 | 10.22 / 10.21 | 12.00 / 12.00 | 64/64 match |
| 14 | 13 / 13 | 29.62 / 1.93 | 55.8 / 45.0 | 30.75 / 3.40 | 12.22 / 12.22 | 14.00 / 14.00 | 64/64 match |
| 16 | 18 / 18 | 26.41 / 1.68 | 63.4 / 61.5 | 27.40 / 2.70 | 14.21 / 14.21 | 16.00 / 16.00 | 64/64 match |
| 18 | 22 / 22 | 20.66 / 1.48 | 68.4 / 68.7 | 21.58 / 2.40 | 16.22 / 16.22 | 18.00 / 18.00 | 64/64 match |
| 20 | 26 / 26 | 15.42 / 1.33 | 75.1 / 78.4 | 16.26 / 2.14 | 18.23 / 18.20 | 20.00 / 20.00 | 64/64 match |
| 22 | 31 / 31 | 9.22 / 1.05 | 76.3 / 87.2 | 10.07 / 1.77 | 20.22 / 20.19 | 22.00 / 22.00 | 64/64 match |
| 24 | 36 / 40 | 4.77 / 0.62 | 91.2 / 117.3 | 5.47 / 1.16 | 21.79 / 22.23 | 23.59 / 23.99 | 64/64 match |

## Archive, 4000 input tokens

Previous wide prefill versus the new default using the same wide graph geometry. One fresh process per condition. The two measurements isolate direct reuse of resident weights; generation speed can vary between runs.

| GiB | Expert layers | TTFT, s | Decode, tok/s | Full request, s | Physical peak, GiB | Ledger peak, GiB | Output IDs |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 2 / 2 | 2.69 / 2.52 | 32.0 / 34.6 | 4.66 / 4.35 | 6.78 / 6.79 | 8.00 / 8.00 | 64/64 match |
| 10 | 4 / 4 | 2.71 / 2.63 | 35.4 / 49.7 | 4.49 / 3.90 | 8.23 / 8.23 | 10.00 / 10.00 | 64/64 match |
| 12 | 9 / 9 | 2.68 / 2.61 | 54.5 / 53.7 | 3.83 / 3.78 | 10.23 / 10.22 | 12.00 / 12.00 | 64/64 match |
| 14 | 13 / 13 | 2.76 / 2.64 | 56.4 / 48.9 | 3.88 / 3.93 | 12.23 / 12.23 | 14.00 / 14.00 | 64/64 match |
| 16 | 18 / 18 | 2.83 / 2.66 | 70.7 / 70.9 | 3.72 / 3.54 | 14.22 / 14.22 | 16.00 / 16.00 | 64/64 match |
| 18 | 22 / 22 | 3.28 / 2.78 | 54.5 / 74.5 | 4.43 / 3.62 | 16.23 / 16.23 | 18.00 / 18.00 | 64/64 match |
| 20 | 26 / 26 | 3.31 / 2.60 | 93.7 / 91.8 | 3.99 / 3.28 | 18.21 / 18.21 | 20.00 / 20.00 | 64/64 match |
| 22 | 31 / 31 | 3.32 / 2.60 | 104.7 / 108.0 | 3.93 / 3.19 | 20.20 / 20.20 | 22.00 / 22.00 | 64/64 match |
| 24 | 36 / 36 | 3.47 / 2.70 | 133.5 / 133.5 | 3.94 / 3.18 | 21.87 / 21.87 | 23.97 / 23.97 | 64/64 match |

Each process served one 8-token warmup before the uncached measurement. Output consisted of 64 greedy tokens. Client-side TTFT excludes loading, decode TPS is 63 divided by elapsed seconds between first and last token delivery, and full time covers the entire HTTP request. Matching token IDs on this corpus does not establish identical logits or outputs on unseen prompts.

[Full measurements, token IDs, threshold comparisons and launcher checks](atomic-engine-qwen36-hybrid-prefill-8-24gib.json)
