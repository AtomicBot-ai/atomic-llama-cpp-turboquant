# Qwen4Exp CUDA long HTTP prefill

## Selected result

The fixed-budget server processes the 3,481-token field-archive prompt through
`/completion` and matches ordinary CUDA output IDs and complete saved slot
bytes. This extends the engine-level long-prefill qualification to actual
server batching on the existing Ubuntu / RTX 4070 Laptop snapshot and
Qwen3.8 GSQ-RCO IQ3_XXS checkpoint. No production code changed for this gate.

Both modes use context 4096, batch 128, microbatch eight, 22 GPU layers, four
threads and one slot. HTTP progress events prove 28 prefill batches: 27 of
128 tokens and a final 25 tokens. Every event reports zero cached input
tokens. Input token IDs and processed boundaries match across modes. The
four greedy output IDs are `[271, 248068, 271, 248069]`; generation stops
at the selected four-token limit with EOS ignored and no prompt truncation.

After completion, `/slots/0?action=save` produces 225,249,576 bytes in each
mode. A chunked byte comparison of the entire files passes, without tolerance
or normalization. Their SHA-256 is
`e3c279638456f5e6e712a0e866f261a28f2810a24da8c936896e4bb3a8e90128`.
The slot is idle after the request. The saved file includes the server's
prompt token list, so its size is not the engine probe's empty-token session
file size. This check compares full slot bytes and output IDs, not fresh
full-vocabulary logits through HTTP.

## Memory and timings

Streaming retains 4 GiB RAM / 2 GiB VRAM and the unchanged 2 GiB runtime
envelope. Managed high-water counters, captured after completion and after
slot save, are 4,252,495,030 RAM bytes (4055.495 MiB) and 2,147,483,648 VRAM
bytes. RAM headroom at the recorded peak is 40.505 MiB. Slot serialization
uses the existing ledger-admitted temporary buffer; saving does not raise
either high-water mark, and used bytes return to their pre-save values.
These are engine-ledger limits, not process RSS, OS file-cache or all driver
allocations. No reserve, placement or batch limit was reduced to pass.

The accepted ordinary control took 206.766 s for prefill; streaming took
713.540 s. These are single diagnostic timings with uncontrolled file-cache
state, not performance qualification. Ordinary CUDA uses `--cpu-moe
--moe-cache-mib 4096 --moe-cache-cpu-share 0`, the previously checked setting
that retains GPU execution for all selected expert layout groups. Its total
memory allowance is larger than streaming's, so the timings are not an
equal-budget comparison. A four-token output is too short to qualify
sustained decode throughput.

## Diagnostic and ordinary-path isolation

The opt-in `tools/expert-source/qwen38-cuda-http-prefill.py` runs one selected
long request per mode. It uses existing HTTP progress reporting, file-based
slot save and metrics; no graph hook, serializer, kernel, server scheduling
or default changed. It starts only its own loopback server and terminates
that process on exit. The ordinary run exposes no managed-engine metrics.
The existing qualified server and libraries were reused without rebuilding
unrelated local changes. Their hashes match across the paired runs.

The first diagnostic mistakenly collected placeholder `tokens: [0]` from
progress events as generated output and asserted before saving the native
slot. Its real prefill and four generated IDs were correct, but no reference
state survived process cleanup. The collector now excludes progress events
from output IDs and saves the slot before semantic assertions. Only that
affected native control was repeated; the failed diagnostic artifacts were
preserved. The subsequent native and streaming runs both pass. No production
change was needed to handle the progress-event format.

[Recorded evidence](atomic-turbine-qwen38-cuda-http-prefill.json) contains
requests, events, boundaries, token IDs, state hashes, metric snapshots,
timings and tested binary/diagnostic fingerprints. The prompt SHA-256 is
`ff564330c036acd2575b9c4174df6f1223b09d2e55fdc1f3a7d1ced3a0f7ea2e`.

## Reproduction and remaining boundaries

With the qualified CUDA runtime on the library search path, use the first
GGUF shard as `MODEL` and a fresh evidence directory. Run sequentially:

```sh
python3 tools/expert-source/qwen38-cuda-http-prefill.py \
  --server build-cuda/bin/llama-server --model "$MODEL" \
  --prompt docs/atomic-turbine-qwen38-context.txt --output "$EVIDENCE" --mode native
python3 tools/expert-source/qwen38-cuda-http-prefill.py \
  --server build-cuda/bin/llama-server --model "$MODEL" \
  --prompt docs/atomic-turbine-qwen38-context.txt --output "$EVIDENCE" --mode stream
```

This qualifies the selected raw HTTP request at fixed caps. The subsequent
[long-slot restore and exact-prefix continuation](atomic-turbine-qwen38-cuda-http-restore.md)
passes in fresh servers with the same caps. The [same-slot series](atomic-turbine-qwen38-cuda-http-series.md)
then passes an exact repeat and a one-token edit in one server process.
Long chat, concurrent slots, other checkpoints/contexts, native pressure
and HTTP resize remain outside these checks.
