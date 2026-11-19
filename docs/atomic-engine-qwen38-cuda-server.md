# Qwen4Exp CUDA fixed-budget server lifecycle

## Selected result

The single-slot raw-completion profile passes short request, stop-string,
SSE-disconnect recovery and file-based slot persistence checks on the
qualified Ubuntu / RTX 4070 Laptop CUDA snapshot. It uses the existing
complete Qwen3.8 GSQ-RCO IQ3_XXS checkpoint, 22 GPU layers, context 4096,
batch 128, microbatch eight, four threads, one sequence and greedy sampling.
Maximum engine budgets remain 4 GiB RAM / 2 GiB VRAM, including the unchanged
2 GiB runtime envelope. This is a short HTTP workload in context 4096, not
long HTTP prefill qualification.

- Initial output is `[11751, 13, 561, 6511]` (` Paris. The capital`). The
  118,286,104-byte slot matches ordinary CUDA byte for byte.
- Stop string `Paris` reports word stop and is removed from the response.
- Closing SSE after its first content chunk releases the slot. The following
  fresh completion reproduces the IDs, and its complete slot matches an
  ordinary CUDA run with the same request history.
- Save/erase/restore preserves every byte of the saved slot. Continuation
  uses eight cached tokens and returns `[314, 9564, 369, 19241]` both before
  and after restore. The 118,409,192-byte continuation states match each
  other and the ordinary CUDA reference byte for byte.
- Captured initial/stop/restore metric peaks are 4,217,092,342 RAM bytes and
  2,147,483,648 VRAM bytes. These are managed engine counters, not process
  RSS, OS file cache or total CUDA-driver usage. The first cancellation run
  stopped at a diagnostic assertion before its post-cancel metric query;
  this missing snapshot is recorded explicitly, not replaced by an estimate.
- One LFM2.5 compatibility completion retains `[4741, 20, 309, 278]` and the
  original 738,504-byte slot hash. The existing LFM HTTP suite was not rerun.
- Unsupported Qwen microbatch 16 fails before model allocation with exit 1
  and `Qwen CUDA streaming requires microbatch <= 8`, with no unbudgeted
  fallback. An initial uncaught loader exception was corrected; only that
  negative startup was repeated after the error-only fix.

[Recorded evidence](atomic-engine-qwen38-cuda-server.json) preserves output
IDs, slot sizes/hashes, observed ledger counters, commands and binary/source
fingerprints. Full vocabulary logits were not recaptured in the server;
this result checks HTTP tokens and full saved state. The separate engine
[long-prefill logit/state gate](atomic-engine-qwen38-cuda-long-prefill.md)
and the targeted staging probe below remain distinct evidence.

## Integration and ordinary-path isolation

`common/atomic-budget.cpp` now reads first-shard GGUF metadata only when
explicit discrete budget flags select its loader. The temporary metadata is
released before model allocation. LFM2 MoE retains its existing boundary
reduction mode; Qwen4Exp uses the already qualified routed arithmetic and
rejects microbatches above eight. Other architectures are rejected explicitly.
Loader failures return the ordinary failed-load result instead of escaping
as an uncaught exception. Ordinary loading still calls `common_model_load`
and performs no additional metadata read for this feature.

The server retains `--parallel 1 --cache-ram 0 --ctx-checkpoints 0`, explicit
CUDA placement, `-fit off`, and fixed budget limits. No pressure worker,
HTTP live-resize API, speculative context or external RAM prompt-cache copy
is introduced.

File-based sequence persistence previously used a temporary vector outside
the engine ledger. A fork-owned reader/writer in `llama-context-atomic.cpp`
now admits this staging for discrete streamed Qwen. Each replacement buffer
is reserved while the old buffer is still charged. Allocation, destruction
and ticket release preserve that order on success and exceptions; the buffer
is released when the file operation ends. The adapters preserve original
serialized bytes and use the existing `llama_io_read_i` / `llama_io_write_i`
extension points. Their upstream source is the file IO implementation at
revision `31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`; track interface, tensor
ordering and error-handling changes during rebases.

Two small hooks at `state_seq_load_file` / `state_seq_save_file` select the
adapter. Without the discrete Qwen gate the helper returns null, and the
unchanged ordinary serializer executes. This adds one disabled dispatch per
file operation, not per token. There is no feature buffer, worker or GPU
operation on that path. No ordinary throughput claim is made. The adapter
lives in the context module so standalone memory-contract probes do not
acquire model/file-IO link dependencies. Public `llama.h` is unchanged.

A separate eight-token engine check reserves all remaining ledger capacity,
then verifies that both file save and restore refuse staging. Releasing that
synthetic reservation permits retry; scratch ownership returns to its prior
level after each failed and successful operation. The final complete state
and all 993,280 finite logits match the saved ordinary CUDA control. This
checks accounting, cleanup and retry, not real OS memory pressure. A denied
save can leave a partial file, which must not be treated as a valid slot.
Failed restore follows upstream cache clearing; recovery is validated by a
successful restore from the original valid file, not by assuming a failed
restore preserves the old context.

## Cold versus reused state

The first streaming scenario compared its post-cancellation state with its
own very first completion. It found 4,836 differences, all F32 `+0.0` versus
`-0.0`. Ordinary CUDA reproduces exactly the same reused-state bytes. The
proper cross-mode comparison with matching request history passes bytewise.
No sign normalization, numerical tolerance or runtime reset change was used.
The saved cancellation artifact was reused for that comparison; already
passed stop/cancel cases were not rerun. The remaining restore case ran in a
separate selected server instance.

## Reproduction and remaining boundaries

Build only the server target, then run the three explicitly selected phases
sequentially. They use an empty evidence directory and loopback port 19159:

```sh
cmake --build build-cuda --target llama-server -j6
python3 tools/expert-source/qwen38-cuda-server.py --server build-cuda/bin/llama-server \
  --model "$MODEL" --output "$EVIDENCE" --phase baseline
python3 tools/expert-source/qwen38-cuda-server.py --server build-cuda/bin/llama-server \
  --model "$MODEL" --output "$EVIDENCE" --phase ordinary-lifecycle
python3 tools/expert-source/qwen38-cuda-server.py --server build-cuda/bin/llama-server \
  --model "$MODEL" --output "$EVIDENCE" --phase lifecycle
```

The ordinary control uses `--cpu-moe --moe-cache-mib 4096
--moe-cache-cpu-share 0` to preserve GPU execution for each selected expert
group. Its total memory allowance is larger than the streaming budget.
The streamed command is:

```sh
llama-server -m "$MODEL" --device CUDA0 -ngl 22 -c 4096 -b 128 -ub 8 \
  -np 1 -t 4 -tb 4 -fit off --no-warmup --cache-ram 0 --ctx-checkpoints 0 \
  --engine-ram-mib 4096 --engine-vram-mib 2048 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 19159
```

The optional `llama-qwen38-cuda-compare` target exposes
`MODEL stream STEM PROMPT_FILE 8 "" --state-file-staging-probe` for the
selected denial/retry check. No aggregate verification was run. The HTTP
scenario bodies ran in the selected steps above; the final CLI wrapper was
syntax-checked without repeating the whole sequence. CUDA source/backend
snapshots were reused, not a fresh full build of unrelated local Metal work.

This closes the selected fixed-budget raw-completion lifecycle gate.
A subsequent selected [non-thinking chat/EOS control](atomic-engine-qwen38-cuda-chat.md)
passes with identical rendered template, input/output IDs and complete saved
state. A selected [long HTTP prefill](atomic-engine-qwen38-cuda-http-prefill.md)
also passes with 28 observed batches and exact full slot bytes at 4/2 GiB.
The selected [long-slot restore and exact-prefix continuation](atomic-engine-qwen38-cuda-http-restore.md)
also passes. The [same-slot long series](atomic-engine-qwen38-cuda-http-series.md)
passes an exact repeat and a one-token edit at the same caps. Thinking mode,
tools, multi-turn chat, concurrent slots, sleep/wake, arbitrary checkpoints,
native pressure and automatic resize remain unqualified. Engine-level
context-2048 resize evidence does not establish an HTTP resize interface,
and context 4096 remains fixed-budget.
