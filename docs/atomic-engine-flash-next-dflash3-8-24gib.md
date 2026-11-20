# Flash Next DFlash2 on Metal at 8–24 GiB

This single-slot profile uses the streamed 48-layer Flash Next target, a compact DFlash2 draft and a fixed maximum of three proposals. Set `FLASH_NEXT_MODEL` to the first target GGUF shard and `FLASH_NEXT_DRAFT` to the compact draft GGUF. The example uses an 8 GiB managed budget. Set `--engine-budget-mib` to the desired budget in MiB.

```sh
LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=1 build-metal/bin/llama-server \
  -m "$FLASH_NEXT_MODEL" -md "$FLASH_NEXT_DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-n-ubatch 128 \
  --engine-budget-mib 8192 -fit off -c 4096 -b 4096 -ub 8 -np 1 \
  -ctk bf16 -ctv bf16 -ngl 99 -ngld 99 -t 8 -tb 8 \
  --cache-ram 0 --ctx-checkpoints 0 --host 127.0.0.1
```

The compact draft uses Q4_0 with BF16 selector weights. To reproduce the measured draft, run `llama-quantize --tensor-type selector_predecessor.weight=bf16 --tensor-type selector_successor.weight=bf16 BF16_DRAFT.gguf COMPACT_DRAFT.gguf Q4_0 6`. Its SHA-256 is recorded in the measurement record below. The local benchmark copy is `.work/models/flash-next-dflash2-q4_0-selector-bf16.gguf`.

During full-width verification the queued Metal path collects eight ready experts per group when available. After submitting the current group it can queue the next known reads while retaining the leases for work already submitted. An optional cache index lets the loader recycle expert buffers with the same allocated size. The engine charges its storage to the managed budget.

For eligible fresh prompts of at least 320 tokens, the target processes balanced blocks of at most 912 input tokens. The server parks the idle draft scheduler before prefill, then the target releases its prompt scratch before decoding. KV, recurrent state and the extracted DFlash2 features retain their own storage. Set `ATOMIC_WINDOW_PROFILE=1` to log actual block sizes.

The [measurement record](atomic-engine-flash-next-dflash3-8-24gib.json) includes client TTFT and decode throughput for each budget, prefill block sizes, the process-lifetime managed memory high-water mark and comparison with the 64 reference token IDs. Each request used a fresh server and an eight-token warmup. TTFT starts at the measured HTTP request. Decode throughput covers the 63 first-to-last token intervals.

| Budget (GiB) | Archive (4000) TTFT (s) | Archive decode (tok/s) | Handover (912) TTFT (s) | Handover decode (tok/s) |
|---:|---:|---:|---:|---:|
| 8 | 44.37 | 15.12 | 9.58 | 14.42 |
| 10 | 45.07 | 14.97 | 9.50 | 15.12 |
| 12 | 46.75 | 14.25 | 9.70 | 15.52 |
| 14 | 47.49 | 14.82 | 9.61 | 15.78 |
| 16 | 45.51 | 16.20 | 9.65 | 15.22 |
| 18 | 45.66 | 17.11 | 9.46 | 17.02 |
| 20 | 45.35 / 45.84 | 9.02 / 17.02 | 9.54 | 17.45 |
| 22 | 45.82 | 16.79 | 9.50 | 16.67 |
| 24 | 45.07 | 16.36 | 9.38 | 17.44 |

The two archive values at 20 GiB show the initial run and its repeat in that order. Every run matched all 64 reference output IDs for its prompt and stayed within the managed budget.

## Live throughput demo

From the repository root or the extracted macOS bundle, point `FLASH_NEXT_MODEL` at the first IQ3_XXS target shard and `FLASH_NEXT_DRAFT` at the compact Q4_0/BF16-selector draft. Keep both target shards together.

```sh
python3 tools/expert-source/demo-flash-next.py ./llama-server "$FLASH_NEXT_MODEL" "$FLASH_NEXT_DRAFT"
```

The demo starts a fresh loopback server on port 8082 with a 20 GiB budget and a 4096-token context. It warms up on eight tokens, streams the 912-token Handover fixture and checks the 64 greedy output IDs. Live decode throughput covers the 63 intervals between the first and last tokens. After printing the result it keeps the web chat open until Ctrl-C. Use `--measure-only` to stop after the comparison. The recorded M4 Max run reached 17.45 tok/s at this budget.

## Experimental 40,960-token context

For the long-context web chat, use the server invocation above with `-c 40960` and `--engine-budget-mib 12288`. Keep the batch at 4096, one slot and the explicit DFlash2 opt-in. The target streams long prompt slices in balanced blocks of at most 912 rows. When the draft needs its compute buffers back, it can reclaim idle expert cache from the synchronized target sharing its budget. The 40k profile reserves 1280 MiB for bounded draft slot staging.

On the M4 Max the experimental server processed 40,000 uncached input tokens and generated eight tokens at 12 GiB. macOS reported a peak process footprint of 9.14 GiB. At 8 GiB target KV allocation failed during startup. At 10 GiB the server started but failed during long prefill. The subsequent draft-buffer fix was exercised with a short chat followed by an uncached 501-token request. See the [40k measurement record](atomic-engine-flash-next-40k.json).
