# Flash Next on Apple Silicon

On the 912-token Handover prompt, live throughput peaked at **17.45 tokens/s** with a 20 GiB budget on the M4 Max. The signed macOS build reached **15.58 tokens/s** and matched all **64/64** reference tokens with the same 4,096-token context and DFlash2 depth of three.

## Run the macOS binary

The 8 GiB chat profile uses a 61,440-token context, Q4_0 target KV cache and a 16-token draft microbatch. It runs with slot file storage disabled. From the extracted bundle, set `FLASH_NEXT_MODEL` to the first target GGUF shard and `FLASH_NEXT_DRAFT` to the compact DFlash2 GGUF. Keep both target shards in the same directory. To share the chat on Wi-Fi, use the Mac's Wi-Fi IPv4 address as the host (usually on `en0`).

```sh
LAN_IP="$(ipconfig getifaddr en0)"
LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=1 ./llama-server \
  -m "$FLASH_NEXT_MODEL" -md "$FLASH_NEXT_DRAFT" \
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-n-ubatch 16 \
  --engine-budget-mib 8192 -fit off -c 61440 -b 512 -ub 8 -np 1 \
  -fa on -ctk q4_0 -ctv q4_0 -ngl 99 -ngld 99 -t 8 -tb 8 \
  --cache-ram 0 --ctx-checkpoints 0 --no-warmup \
  --ui --alias flash-next --host "${LAN_IP:?Set LAN_IP to the Wi-Fi IPv4 address}" --port 8082
```

Open `http://<LAN_IP>:8082/` on a device on the same network. The API base URL is `http://<LAN_IP>:8082/v1`. For a local-only demo set `LAN_IP=127.0.0.1`. The profile serves one generation at a time.

## Live throughput demo

From the repository root or the extracted macOS bundle, point `FLASH_NEXT_MODEL` at the first IQ3_XXS target shard and `FLASH_NEXT_DRAFT` at the compact Q4_0/BF16-selector draft. Keep both target shards together.

```sh
python3 tools/expert-source/demo-flash-next.py ./llama-server "$FLASH_NEXT_MODEL" "$FLASH_NEXT_DRAFT"
```

The demo starts a fresh loopback server on port 8082 with a 20 GiB budget and a 4096-token context. It warms up on eight tokens, streams the 912-token Handover fixture and checks the 64 greedy output IDs. Live decode throughput covers the 63 intervals between the first and last tokens. After printing the result it keeps the web chat open until Ctrl-C. Use `--measure-only` to stop after the comparison.
