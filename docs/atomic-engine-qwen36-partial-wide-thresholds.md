# Qwen3.6-35B-A3B partial wide prefill on M4 Max

No single minimum prompt length below 1024 preserved all 64 greedy output IDs on the calibration corpus at 8, 10 or 12 GiB. Each budget had a counterexample at 1023 tokens. The measured result supports keeping short wide prefill opt-in when token identity is required.

Each cell gives the stock selector followed by the forced partial wide selector. The 4000-token archive already uses aligned wide prefill in both conditions. Timings are client observations on freshly started, warmed servers, excluding model load. Each request had an uncached prompt and 64 greedy output tokens. TTFT ends at the first token, decode TPS covers the subsequent 63 tokens, and the full request ends when the HTTP stream completes. Physical peak is the maximum lifetime physical footprint reported by macOS, including startup and warmup.

The 8 GiB profile uses serial decoding, ordinary attention and two direct expert layers. At 10 and 12 GiB both arms use BF16 DFlash2 with six proposals, GQA8, and four and nine direct layers respectively. Rows marked ordinary attention qualify the launcher without the GQA8 override.

## Original five prompts

Per-condition medians from two fresh processes, reversing the order of the two conditions on the second repetition. Token comparisons are 1-based, over IDs. The first disagreement position repeated in each original-prompt pair.

| Budget, GiB | Prompt | Input tokens | TTFT, s | Decode, tok/s | Full request, s | Physical peak, GiB | First differing output |
|---:|---|---:|---:|---:|---:|---:|---|
| 8 | Russian explanation | 52 | 1.84 / 1.44 | 32.8 / 33.6 | 3.79 / 3.32 | 6.785 / 6.786 | 11 |
| 8 | Python function | 95 | 3.64 / 1.54 | 32.9 / 28.6 | 5.56 / 3.78 | 6.788 / 6.786 | 64/64 match |
| 8 | Dispatch log | 227 | 8.10 / 1.54 | 37.8 / 27.7 | 9.77 / 3.82 | 6.787 / 6.787 | 37 |
| 8 | Shift handover | 912 | 38.78 / 1.63 | 34.9 / 33.1 | 40.59 / 3.53 | 6.790 / 6.787 | 37 |
| 8 | Original archive | 4000 | 3.05 / 3.18 | 28.6 / 27.9 | 5.32 / 5.45 | 6.786 / 6.785 | 64/64 match |
| 10 | Russian explanation | 52 | 2.02 / 2.10 | 40.3 / 36.1 | 3.58 / 3.84 | 8.217 / 8.217 | 11 |
| 10 | Python function | 95 | 3.44 / 2.02 | 72.8 / 60.6 | 4.30 / 3.06 | 8.221 / 8.221 | 64/64 match |
| 10 | Dispatch log | 227 | 9.42 / 2.08 | 37.6 / 34.2 | 11.09 / 3.92 | 8.221 / 8.216 | 37 |
| 10 | Shift handover | 912 | 47.96 / 2.17 | 44.8 / 38.9 | 49.37 / 3.81 | 8.222 / 8.219 | 64/64 match |
| 10 | Original archive | 4000 | 3.17 / 3.42 | 44.9 / 42.3 | 4.58 / 4.91 | 8.235 / 8.232 | 64/64 match |
| 12 | Russian explanation | 52 | 2.08 / 2.15 | 41.4 / 40.4 | 3.60 / 3.71 | 10.213 / 10.212 | 11 |
| 12 | Python function | 95 | 4.13 / 2.27 | 63.3 / 63.1 | 5.13 / 3.27 | 10.215 / 10.215 | 64/64 match |
| 12 | Dispatch log | 227 | 8.41 / 2.16 | 39.3 / 32.2 | 10.02 / 4.15 | 10.210 / 10.213 | 37 |
| 12 | Shift handover | 912 | 42.43 / 2.15 | 50.2 / 48.2 | 43.69 / 3.46 | 10.213 / 10.211 | 64/64 match |
| 12 | Original archive | 4000 | 3.03 / 3.17 | 43.2 / 51.7 | 4.50 / 4.39 | 10.228 / 10.225 | 64/64 match |

## Additional length and prompt controls

The handover variants retain the chat framing, shortening the body or inserting tokens from earlier notes. The independent archive inputs use the suffix of the original 4000-token fixture. The Python excerpts retain the chat framing and use two A/B repetitions per length. Other additional controls use one A/B pair per budget and attention mode. These are shape and output-identity checks on specific inputs.

| Budget, GiB | Prompt | Input tokens | TTFT, s | Decode, tok/s | Full request, s | Physical peak, GiB | First differing output |
|---:|---|---:|---:|---:|---:|---:|---|
| 8 | Python excerpt | 32 | 1.30 / 1.66 | 36.0 / 33.5 | 3.05 / 3.55 | 6.787 / 6.787 | 64/64 match |
| 8 | Archive suffix | 960 | 39.89 / 1.51 | 33.7 / 31.1 | 41.76 / 3.54 | 6.790 / 6.784 | 26 |
| 8 | Handover variant | 960 | 41.62 / 1.63 | 35.0 / 33.5 | 43.42 / 3.52 | 6.788 / 6.784 | 64/64 match |
| 8 | Archive suffix | 1023 | 43.11 / 1.65 | 32.9 / 33.7 | 45.02 / 3.52 | 6.792 / 6.785 | 1 |
| 8 | Handover variant | 1023 | 43.21 / 1.41 | 35.2 / 26.2 | 45.00 / 3.82 | 6.792 / 6.783 | 64/64 match |
| 10 | Python excerpt | 72 | 2.79 / 1.94 | 77.9 / 66.5 | 3.60 / 2.89 | 8.216 / 8.216 | 64/64 match |
| 10 | Handover variant | 256 | 12.02 / 1.98 | 35.8 / 39.5 | 13.77 / 3.57 | 8.227 / 8.221 | 53 |
| 10 | Handover variant | 512 | 25.93 / 2.08 | 42.2 / 30.1 | 27.42 / 4.18 | 8.215 / 8.215 | 20 |
| 10 | Handover variant | 768 | 38.72 / 2.09 | 37.2 / 35.4 | 40.41 / 3.87 | 8.219 / 8.225 | 32 |
| 10 | Archive suffix | 912 | 46.71 / 2.17 | 50.9 / 44.4 | 47.95 / 3.59 | 8.222 / 8.227 | 64/64 match |
| 10 | Archive suffix (ordinary attention) | 912 | 45.38 / 2.01 | 49.8 / 32.3 | 46.64 / 3.96 | 8.226 / 8.222 | 64/64 match |
| 10 | Handover variant (ordinary attention) | 912 | 47.05 / 2.01 | 48.4 / 33.8 | 48.35 / 3.87 | 8.217 / 8.220 | 64/64 match |
| 10 | Handover variant | 960 | 47.68 / 2.07 | 46.5 / 47.0 | 49.03 / 3.41 | 8.221 / 8.223 | 16 |
| 10 | Handover variant | 1023 | 52.41 / 2.16 | 46.2 / 44.6 | 53.78 / 3.57 | 8.221 / 8.216 | 2 |
| 10 | Handover variant (ordinary attention) | 1023 | 52.19 / 2.09 | 45.9 / 45.7 | 53.56 / 3.47 | 8.224 / 8.216 | 2 |
| 12 | Python excerpt | 72 | 2.91 / 2.21 | 71.9 / 60.5 | 3.79 / 3.28 | 10.209 / 10.209 | 64/64 match |
| 12 | Handover variant | 256 | 11.26 / 2.16 | 41.3 / 42.5 | 12.79 / 3.64 | 10.216 / 10.216 | 53 |
| 12 | Handover variant | 512 | 23.65 / 2.11 | 47.0 / 36.1 | 24.99 / 3.86 | 10.214 / 10.207 | 20 |
| 12 | Handover variant | 768 | 35.06 / 2.26 | 52.1 / 34.5 | 36.27 / 4.09 | 10.213 / 10.208 | 32 |
| 12 | Archive suffix | 912 | 42.34 / 2.15 | 40.6 / 49.2 | 43.89 / 3.44 | 10.215 / 10.216 | 64/64 match |
| 12 | Archive suffix (ordinary attention) | 912 | 42.27 / 2.25 | 55.9 / 37.1 | 43.40 / 3.95 | 10.215 / 10.220 | 64/64 match |
| 12 | Handover variant (ordinary attention) | 912 | 41.76 / 2.41 | 52.3 / 46.2 | 42.96 / 3.77 | 10.215 / 10.213 | 64/64 match |
| 12 | Handover variant | 960 | 43.43 / 2.40 | 50.0 / 47.0 | 44.69 / 3.74 | 10.212 / 10.211 | 16 |
| 12 | Handover variant | 1023 | 46.32 / 2.23 | 51.4 / 46.7 | 47.55 / 3.58 | 10.214 / 10.211 | 2 |
| 12 | Handover variant (ordinary attention) | 1023 | 47.12 / 2.27 | 50.6 / 44.4 | 48.37 / 3.69 | 10.213 / 10.212 | 2 |

[Individual requests, exact input references, output IDs, ledger high-water marks and Metal path counters](atomic-engine-qwen36-partial-wide-thresholds.json)
