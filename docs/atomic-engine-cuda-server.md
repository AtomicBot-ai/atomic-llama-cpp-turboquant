# P11 fixed-budget CUDA server profile

The later [Qwen4Exp server control](atomic-engine-qwen38-cuda-server.md)
separately qualifies short raw completion, stop, SSE cancellation and slot
restore. The LFM result and historical boundaries below are unchanged.

## Scope and result

The single-slot LFM2.5-24B-A2B-AD-Q4_K_M server profile passes the targeted
P11 HTTP checks on Ubuntu with an RTX 4070 Laptop GPU (8 GiB). The engine
has separate 4 GiB RAM and 2 GiB VRAM admission caps. This closes this
fixed-budget profile, not general CUDA server qualification. Qwen3.8 CUDA,
multiple slots, long contexts, automatic pressure control and HTTP live
resize remain outside this result.

The checked model SHA-256 is
`6cfa4cafd31b97cb008041fe9fc76f3d4c3793ec044cf996695fbff4d1d23140`.
The selected configuration uses 14 GPU layers, context/batch 256,
microbatch 8, four CPU threads, one sequence and greedy sampling.

## Integration and isolation

`--engine-ram-mib` and `--engine-vram-mib` opt into the existing discrete
streaming loader. Configuration checks live in `common/atomic-budget.cpp`.
Both flags are required, with one explicit CUDA device, `-fit off`, no
UMA/automatic budget, no upstream host-expert cache and no speculative,
multimodal or adapter context. The existing context gate requires partially
offloaded LFM2 MoE and op offload. The server also requires `--parallel 1
--cache-ram 0 --ctx-checkpoints 0` to avoid unaccounted external state copies.

The loader retains the P10B CPU expert placement and CUDA boundary weighted
reduction that reproduce ordinary CUDA fusion. No CUDA kernel or fusion
implementation changed in this step. No new pressure worker is started.
Ordinary loading still calls `common_model_load`; without the new flags the
added work is configuration-branch checks during initialization and one
disabled metrics branch. No feature buffer, read or GPU work is introduced
on the ordinary path. This is structural isolation, not a new throughput
measurement.

An idle inference-thread snapshot exposes independent
`llamacpp:engine_memory_ram_*` and `llamacpp:engine_memory_vram_*` gauges.
Snapshots are omitted on the server's concurrent yielded-task path.
The fork-only accessor lives in `llama-atomic.h`; public `llama.h` is unchanged
by this integration. `released_bytes` is cumulative allocation turnover,
not the amount currently free.

## Targeted evidence

Checked on 2026-11-15; selected outputs and per-phase ledger snapshots are
preserved in [the compact result record](atomic-engine-cuda-server.json).

- First and repeated raw completion produce `[4741, 20, 309, 278]`
  (` Paris, and the`), matching an ordinary CUDA server.
- Their 738,504-byte saved slot files match the ordinary server byte for byte.
  SHA-256: `6ae45b8400f8343125781a534c75b2a10b3647a1d771d5da3ad9f8409ee51e96`.
- A separate temporary server-library probe captured four complete vocabulary
  logit rows. All 2,048,000 bytes match the saved ordinary CUDA P10B control
  for the same prompt, context, microbatch and layer split. SHA-256:
  `ffd339bb13fdc241e536457a3dfea3f2a6fe28ff9df80fd798caa6bebaae263f`.
  Capture instrumentation is absent from the production server library.
- Stop string `Paris` is removed from the response and reports word stop.
- Chat completion naturally ends after 40 generated tokens and answers Paris.
  A first 32-token limit ended during reasoning; the remaining EOS check used
  128, without rerunning the already-passed completion and stop checks.
- Closing SSE after its first content chunk frees the slot. The following
  request reproduces the original token IDs and saved state.
- Save, erase and restore reproduce the slot bytes before further inference.
  Continuation uses 12 cached tokens and returns `[4569, 3368, 355, 4770]`
  both before and after restore. Its 820,488-byte states match bytewise:
  `f48e22c1bc290e865157b385c9b5f9033e829459e2329a999c6ebd0d5e4715ba`.
- Across these HTTP scenarios, ledger peaks are 3,097,860,944 RAM bytes
  and 2,147,483,648 VRAM bytes, within 4,294,967,296 / 2,147,483,648 caps.
- Incomplete caps, two slots and a 1 MiB RAM cap fail with exit code 1,
  diagnostic messages and no fallback to unbudgeted loading. Validation
  also handles the argument parser's empty tensor-override terminator.
- Three focused local component cases cover the new arguments, existing UMA
  arguments and disabled idle pressure control: 22 distinct assertions pass.
  Only the changed argument case was repeated after the terminator fix.

The CUDA executable was built in an isolated directory from the previously
qualified CUDA source snapshot plus this narrow integration patch. Its
prebuilt ggml CUDA backend was reused after checking relevant source hashes
against the local backend. Unrelated local Metal work was not transferred.
This is not qualification of a fresh full Linux build of every local change.

## Memory and support boundaries

The RAM ledger includes the existing 2 GiB runtime envelope. Driver-owned
allocations, OS file cache and total process RSS are not these engine caps.
Snapshots establish admitted engine memory, not an OS-wide hard limit.
File-based slot persistence is checked; RAM prompt-cache copies and context
checkpoints stay disabled.

The argument gate permits explicit contexts/batches up to 4096 and
microbatches up to 32, but this HTTP evidence covers only 256/256/8. Other
admitted configurations still require their own checks. Performance,
concurrent clients, server sleep/wake, other quantizations and long sessions
were not qualified. Engine-level P10B resize/cancellation evidence does not
imply an HTTP resize API. No aggregate verification or full suite was run.

## Reproduction

With a CUDA build and the selected model available:

```sh
build/bin/llama-server -m /path/to/LFM2.5-24B-A2B-AD-Q4_K_M.gguf \
  --device CUDA0 -ngl 14 -c 256 -b 256 -ub 8 -np 1 -t 4 -tb 4 \
  -fit off --no-warmup --cache-ram 0 --ctx-checkpoints 0 \
  --engine-ram-mib 4096 --engine-vram-mib 2048 \
  --metrics --slots --slot-save-path /path/to/slot-directory \
  --host 127.0.0.1 --port 19149
```

The explicitly invoked HTTP check runs ordinary and streamed servers
sequentially on port 19149, which must be free. It writes logs, metrics and
slot files to an empty directory:

```sh
python3 tools/expert-source/p11-cuda-server.py \
  --server build/bin/llama-server \
  --model /path/to/LFM2.5-24B-A2B-AD-Q4_K_M.gguf \
  --output /path/to/empty-evidence-directory
```

The scenario body was executed in two selected runs; the final CLI wrapper
was syntax-checked, not rerun as an aggregate. This script compares HTTP
tokens and saved states, not raw logits. The separate logit probe and
invalid-startup checks produced the additional evidence above.
