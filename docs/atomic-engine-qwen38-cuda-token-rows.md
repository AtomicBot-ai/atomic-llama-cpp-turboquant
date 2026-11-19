# Qwen4Exp CUDA token-row streaming

## Change

The discrete Qwen4Exp loader now also leaves `token_embd.weight` on disk.
The checked IQ3_S table is `[2560, 248320]`, occupying 273,152,000 bytes
(260.5 MiB). A separately admitted 1,100-byte packed row replaces that
resident allocation. The existing 2 GiB runtime envelope is unchanged.

Two token-gather boundaries in `build_inp_embd` select the custom row reader
only when the model has explicitly selected this streamed table and owns an
expert executor. The ordinary branch still builds GET_ROWS, including its
existing mixed-input, scaling and adapter logic. The new branch adds two
short-circuit checks during graph construction and one internal model pointer;
it adds no ordinary GPU work or file reads. No ordinary CUDA kernel, fusion,
allocation default or scheduling rule is changed.

The token-row callback is separate from the previously qualified PLE callback.
This small fork-owned duplication isolates Metal/PLE behavior. It reuses the
pinned GGUF source, original IQ3_S bytes and the upstream CPU type conversion.
Its host I/O ticket precedes allocation and outlives the buffer. The scheduler
owns the F32 output and any device copy under the existing memory observer.
Read failure marks the executor unusable, and the context rejects evaluation.
The model gate requires a distinct output matrix and the checked IQ3_S layout;
tied output weights and other embedding quantizations fail closed.

## Targeted operation and allocation evidence

The real-weight row fixture contains the first, second, middle and last
embedding rows, without requantization. Their 4,400-byte packed payload has
SHA-256 `7fa503e8a1c6ce6cfb2a45f1b60bf721459fa1ab0067f675699b14a168e7ee3e`.
Five gathers, including a repeated row, match ordinary CPU GET_ROWS in all
12,800 float values. The row workspace is 1,100 bytes and all executor storage
returns on teardown. Only the new token-row case was run; the seven previously
qualified expert formats were not rerun.

An allocation-only check with the original GGUF headers and sparse zero
payload now initializes at 22 GPU layers, requested context 128 (effective
256), batch 128, microbatch eight, one sequence and four CPU threads:

- Host used/high-water: 4,115,270,726 bytes, including the 2 GiB envelope.
- Device used/high-water: 2,020,540,416 bytes.
- Main graph buffers: approximately 8.55 MiB CUDA and 7.63 MiB host.

The caps remain 4,294,967,296 / 2,147,483,648 bytes. This admission result
includes state and graph reservation; it does not include all lazy kernel
workspace or expert cache growth during actual execution. Sparse zero payload
is an allocation fixture, never a numerical model.

## Full-model control protocol

`llama-qwen38-cuda-compare` selects one short raw prompt, four greedy outputs,
22 GPU layers, context 128, microbatch eight and four CPU threads. The streamed
mode keeps the 4/2 GiB caps. It reserves a separate 128 MiB state-capture buffer
inside the host ledger before inference, so the diagnostic cannot hide that
allocation outside the budget. All full-vocabulary output rows and serialized
state are written for bytewise comparison.

The native mode uses the upstream host-expert cache with a 2 GiB device cache,
original packed CPU expert buffers and zero CPU share for GPU-layer cache
misses. CPU layers retain CPU expert execution. A fully resident 22-layer
control cannot fit the 8 GiB GPU; this is an ordinary upstream cache control,
not a second fork streaming configuration.

The Ubuntu Strata installation already contains the complete checkpoint under
`~/atomic-strata-live-qwen38/data/strata/models/qwen38-iq3`. Both complete file
SHA-256 values match the Mac sources:

- First shard: `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`.
- Second shard: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

An initial network-backed load was stopped before inference when the user
identified that installation. Its temporary read-only mount was removed and
the interrupted duplicate copy discarded. Neither run used for numerical
comparison depends on network transport or a prefault wrapper. Both use the
local complete files and unchanged ordinary mmap/prefault behavior. The
native control enables only the existing host-expert cache configuration.

## Full-model result

Both sequential runs completed on Ubuntu / RTX 4070 Laptop (8 GiB).
The five-token prompt produced IDs `[11751, 13, 561, 6511]` in both modes.
All four full-vocabulary logit rows (993,280 finite floats; 3,973,120 bytes)
match byte for byte. The 118,286,060-byte serialized state also matches.
The final emitted token is not fed back: the captured state follows prefill
and three single-token decode steps.

- Logits SHA-256: `21098cc696fca6a15bc6c1a0d9d972a7cd05d94ec1975ac92a72c349e1ec2f31`.
- State SHA-256: `eca0fd3737df17ee2972fee332e4662bc76e42e6801b98f26f9314505c609bf6`.
- Managed host peak: 4,291,713,190 bytes; headroom 3,254,106 bytes (3.10 MiB).
- Managed device peak: 2,147,483,648 bytes, exactly the 2 GiB cap.

The 2 GiB runtime envelope and the separate 128 MiB diagnostic state buffer
are included in the host accounting. These are managed-ledger peaks, not
whole-process RSS or total driver-memory measurements. The ordinary control
is not constrained to the streaming host cap. See the
[machine-readable result](atomic-engine-qwen38-cuda-token-rows.json).

This closes the selected short-request capacity and numerical gates. It does
not qualify long prefill/context, live resize, server integration, mixed
embedding inputs, adapters or other checkpoints. Microbatches above eight
remain rejected. No throughput or ordinary-path performance claim is made.

The subsequent [3,481-token CUDA prefill control](atomic-engine-qwen38-cuda-long-prefill.md)
now passes the same 4/2 GiB caps at context 4,096 and matches all captured
logits and serialized state against a corrected ordinary CUDA control.
It includes a streaming-only routing-weight lifetime fix; see that scoped
qualification before generalizing this short result.

## Reproduction

The optional tools are excluded from default builds:

```sh
cmake --build build-cuda --target llama-qwen38-cuda-probe llama-qwen38-cuda-compare -j6
PYTHONPATH=gguf-py python3 tools/expert-source/qwen38-cuda-fixture.py \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --include-token-embedding --output /path/to/new-token-fixture.gguf
build-cuda/bin/llama-qwen38-cuda-probe /path/to/new-token-fixture.gguf all 0 token
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" stream stream-output
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" native native-output
cmp stream-output.f32 native-output.f32
cmp stream-output.state native-output.state
```

Use local complete GGUF files for reproduction.
Only directly relevant checks were selected. No aggregate suite or performance
benchmark is part of this qualification.
