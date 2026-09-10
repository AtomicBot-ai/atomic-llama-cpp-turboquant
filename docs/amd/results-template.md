# TurboQuant on <GPU>: ROCm (HIP) vs Vulkan

Copy this file to `benchmarks-<YYYY-MM-DD>.md`, fill in, commit the raw files
from `docs/amd/results/` next to it.

## Environment

| | |
|---|---|
| GPU | <name, gfx id, VRAM> |
| Host | <CPU, RAM, OS, kernel> |
| ROCm | <version> (`/opt/rocm/.info/version`) |
| Mesa / RADV | <version> (`vulkaninfo --summary`) |
| Fork build | `llama-bench --version` output, archive name |
| Upstream build | tag `b<N>`, archive name |
| Flags | `-fa 1 -ngl 99 -ub 2048 -r 3`, env vars |
| Raw files | `results/<host>__hip__<stamp>*`, `results/<host>__vulkan__<stamp>*` |

## Throughput

Output of `python3 scripts/bench-amd-report.py results/<host>-*.json --baseline hip/f16`.

| model | test | depth | hip/f16 | hip/q8_0 | hip/turbo3 | hip/turbo4 | vulkan/f16 | vulkan/turbo3 | upstream-hip/f16 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | pp512 | 0 | | | | | | | |
| | pp4096 | 0 | | | | | | | |
| | tg128 | 0 | | | | | | | |
| | tg128 | 8192 | | | | | | | |
| | tg128 | 32768 | | | | | | | |

## KV cache memory

| model | context | f16 | q8_0 | turbo3 | turbo4 | largest context that fits (turbo3) |
|---|---:|---:|---:|---:|---:|---:|
| | 32768 | | | | | |
| | 131072 | | | | | |

Source: `llama-server` log lines `KV self size` at startup, or the OOM point
when sweeping `-c`.

## Perplexity (wikitext-2, ctx 4096, 20 chunks)

From `results/<host>-<label>-<stamp>.ppl.txt`.

| model | f16 | q8_0 | turbo3 | turbo4 |
|---|---:|---:|---:|---:|
| | | | | |

## Speculative decoding

From `scripts/bench-matrix-qwen.sh` (NextN) and the Gemma MTP server scripts,
same backend builds.

| model | backend | KV | spec | short tok/s | long tok/s | accept % |
|---|---|---|---|---:|---:|---:|
| | hip | turbo3 | off | | | |
| | hip | turbo3 | nextn | | | |
| | vulkan | turbo3 | nextn | | | |

## Correctness (CDNA only)

`test-backend-ops -o SET_ROWS` / `-o FLASH_ATTN_EXT` result for turbo types,
wave size, and whether any fix was needed.

## Notes

What was surprising, what failed, what to tune next.
