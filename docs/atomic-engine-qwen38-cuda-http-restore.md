# Qwen4Exp CUDA long HTTP slot restoration

The CUDA server restored the 225,249,576-byte slot from the paired
[3,481-token HTTP prefill](atomic-engine-qwen38-cuda-http-prefill.md) in a
fresh process in each mode. The restore endpoint reported 3,484 tokens read
from each file. An immediate save matched the original slot byte for byte in
both modes.

The continuation sends the exact original prompt IDs and all four generated
IDs as its input. The saved state already includes the first three generated
IDs. Supplying IDs directly avoids changing the prefix through text
serialization and tokenization. Both servers reused all 3,484 cached tokens
and evaluated one new input token. They emitted four greedy IDs
`[271, 760, 6511, 314]`. The complete 225,372,664-byte continuation files
matched byte for byte, with SHA-256
`6036655d467a04526708cbbf33c14153e3f63ebf97fc163829136fd727307a7f`.

Streaming retained 4 GiB RAM / 2 GiB VRAM, with the existing 2 GiB runtime
reserve, at context 4096, batch 128, microbatch eight and 22 GPU layers.
The observed engine ledger peaks were 4,193,465,894 RAM bytes
(3999.201 MiB) and 2,147,483,648 VRAM bytes. Restore and save left the
post-operation ledger usage unchanged at their respective checkpoints.
The counters cover managed engine allocations. Both runs used the same
previously qualified server and libraries. Changes are limited to the
opt-in diagnostic and documentation.

Run the following commands sequentially with the qualified CUDA runtime on
the library search path. `PREFILL` is the previous diagnostic directory with
`native.json`, `stream.json` and the corresponding `long.slot` files.
`EVIDENCE` is a fresh directory and `MODEL` is the first GGUF shard.

```sh
python3 tools/expert-source/qwen38-cuda-http-restore.py \
  --server build-cuda/bin/llama-server --model "$MODEL" \
  --prefill "$PREFILL" --output "$EVIDENCE" --mode native
python3 tools/expert-source/qwen38-cuda-http-restore.py \
  --server build-cuda/bin/llama-server --model "$MODEL" \
  --prefill "$PREFILL" --output "$EVIDENCE" --mode stream
```

The [recorded evidence](atomic-engine-qwen38-cuda-http-restore.json)
contains the endpoint results, state hashes, binary fingerprints and ledger
snapshots. This qualifies fresh-process restore and one exact-prefix
continuation for the selected raw prompt. The [same-slot series](atomic-engine-qwen38-cuda-http-series.md)
separately covers an exact repeat and a one-token edit. Long chat, arbitrary
state files and full-vocabulary logits remain outside this check.
