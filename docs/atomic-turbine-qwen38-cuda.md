# Qwen3.8 CUDA: row streaming and bounded operation qualification

## Result

The subsequent [token-row follow-up](atomic-turbine-qwen38-cuda-token-rows.md)
closes the short-request 4/2 GiB capacity gate and full-model logits/state
comparison at 22 GPU layers. The PLE-only rejection below is historical;
its operation-level evidence remains applicable.
The later [long-prefill control](atomic-turbine-qwen38-cuda-long-prefill.md)
now passes the same caps and bitwise full-model logits/state comparison
for the selected 3,481-token context-4,096 profile after a streaming-only
routing-weight lifetime fix. Broader long-context coverage remains open.

The discrete streaming path now reads Qwen4Exp PLE rows on demand and admits
its seven routed weight formats. On the selected real-weight extracts, all
28 CPU/CUDA projection comparisons at one and eight tokens match bit for bit.
The PLE gather also matches ordinary CPU GET_ROWS, which is the input-layer
placement used with partial CUDA offload. This is operation-level evidence,
not full-model logits/state equivalence.

The PLE-only 4 GiB RAM / 2 GiB VRAM profile failed admission. With PLE
removed from residency, recurrent-state allocation exceeds RAM at 23 GPU
layers and VRAM at 24. No model inference or full-model numerical comparison
was run. The [machine-readable record](atomic-turbine-qwen38-cuda.json) includes
source-slice hashes, dimensions, selected indices and exact rejection sizes.
Checked on 2026-11-15, Ubuntu / RTX 4070 Laptop (8 GiB).

## Implementation boundary

The existing `ml.stream_ple` gate now also admits an explicitly configured
single-device discrete streaming contract. It leaves the full PLE tensor as
metadata. The already-existing graph hook calls `build_rows`; ordinary
execution still builds the upstream GET_ROWS node.

`configure_rows` now accepts Qwen4Exp on the discrete executor. The callback
uses the existing pinned GGUF reader and CPU IQ4_NL conversion, with one
90-byte packed row charged to host I/O before allocation. The CPU custom-op
output is scheduler-owned and remains under the graph's allocation observer;
the scheduler transfers it when a subsequent operation requires CUDA. No
28.8 GB shadow allocation or unaccounted row cache is introduced. File identity
checks and deferred evaluation failure use the existing source contract.

The seven extra formats are enabled only for Qwen4Exp on Metal or a validated
CUDA discrete executor. Original packed weights are preserved. Existing CUDA
MUL_MAT_ID kernels and CPU-layer placement provide the arithmetic. No CUDA
kernel, fusion, ordinary loader default or ordinary scheduling rule changed.
The new microbatch check executes only in the streaming graph builder; disabled
streaming has no new allocation, read, worker or graph node.

The common server's LFM2-only boundary-reduction profile is unchanged. This
work does not enable Qwen4Exp through that server profile. Diagnostic discrete
loading uses expert streaming and CPU-layer matching, without LFM boundary
reduction.

## Focused numerical checks

`qwen38-cuda-fixture.py` extracts original packed bytes without requantizing:
12 experts per format, selected from original IDs
`0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 383, 511`, and four PLE rows including
the first and last. Expert input/output widths remain 2560/640 or 640/2560.
The final fixture is 49,613,856 bytes. It is an operation fixture, not a model.

`llama-qwen38-cuda-probe` compares native MUL_MAT_ID against the streamed
projection with ten distinct selected experts per token, varying their order
and membership across tokens. Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and
IQ4_NL match on CUDA and on CPU-layer execution, at one and eight tokens:
28 comparisons and 1,497,600 float results, compared bytewise. The fixture uses
12 experts rather than the full 512-expert tensor; this does not establish
whole-model routing or fusion equivalence.

Five PLE gathers, including a repeated row, produce 800 float values identical
to ordinary CPU GET_ROWS. The original width is 160. Directly forcing that
GET_ROWS onto the CUDA quantized kernel is invalid: its IQ4_NL implementation
requires a 256-value super-block width. The initial forced-CUDA diagnostic
hit that assertion; it was corrected to use the ordinary input-layer CPU
placement. Production CUDA code was not changed to accommodate the probe.

The first projection diagnostic used two selected experts. At 32 tokens,
IQ4_NL differed at output index 129 (`-0.0459616408` versus `-0.0459616445`).
The streamed Qwen4Exp graph builder therefore rejects microbatches above eight
before evaluation. A focused nine-row negative check confirms rejection and
release of all executor allocations. The larger matrix profile stays open.
The diagnostic also changed native backend lifetime to keep CUDA graph
captures within each reference graph's buffer lifetime after an initial
multi-case CUDA error; the final top-ten comparisons all pass.

After each admitted projection profile, host usage returns to the existing
2 GiB runtime reservation and device usage to zero. The largest top-ten
operation run peaks at 2,160,863,456 host bytes and 54,329,344 device bytes;
these fixture peaks must not be presented as full-model memory use.

## Allocation-only budget check

Both original GGUF headers were copied to sparse files with zero payload,
retaining original tensor shapes, types, offsets and file lengths. The loader
and context then performed real CPU/CUDA allocations under the unchanged
4/2 GiB contract. The diagnostic contains no decode call and cannot produce a
numerical result. These files are not usable weights. They were removed after
the check; the headers and explicit diagnostic remain reproducible.

Payload placement leaves only two candidates even before state and scratch.
The input embedding stays on CPU; output and output hyper-connection weights
follow the output device. The existing layer split starts GPU layers at
`49 - n_gpu_layers`:

- At 22 GPU layers, non-routed RAM payload is 2,158,165,888 bytes, already
  above the 2 GiB available after the host runtime reservation. Lower offload
  cannot improve RAM admission.
- At 25 GPU layers, non-routed CUDA payload is 2,171,324,160 bytes, already
  above the 2 GiB device cap. Higher offload cannot improve VRAM admission.
- Only 23 and 24 remain candidates; both were checked, rather than sweeping
  all layer counts with runtime tests.

The requested context is 128, aligned to 256 by the runtime; batch 128,
microbatch eight, one sequence, four CPU threads. Results:

- **23 GPU layers:** RAM usage is 4,245,159,936 bytes before the next recurrent
  state allocation. Its admitted size is 65,798,144 bytes. The 4,294,967,296-byte
  cap rejects it with a 15,990,784-byte deficit (15.25 MiB).
- **24 GPU layers:** VRAM usage is 2,105,344,000 bytes before its recurrent state
  allocation. Its admitted size is 55,574,528 bytes. The 2,147,483,648-byte cap
  rejects it with a 13,434,880-byte deficit (12.81 MiB).

Both fail during context construction, before the executor and main compute
workspace are allocated. These deficits are lower bounds on additional space,
not sufficient budget increases. Shrinking the expert cache cannot fix state
admission before that cache exists. No runtime reserve was reduced, no model
quantization was changed and neither cap was increased.

The remote build used the previously qualified CUDA base/backend with the
current executor/source/store dependencies and the narrow existing Qwen PLE
loader/graph hooks. This was not a full Linux rebuild of every local change;
Metal-only code in the shared fork modules was not qualified. The local
library also builds with these changes. The selected existing CPU executor
control passes 16 assertions in the shared streaming builder; no full suite, aggregate gate,
ordinary throughput benchmark or real-model CUDA decode was run.

## Reproduction and next gate

The diagnostics are excluded from the default build:

```sh
cmake --build build-cuda --target llama-qwen38-cuda-probe llama-qwen38-cuda-admission -j6
PYTHONPATH=gguf-py python3 tools/expert-source/qwen38-cuda-fixture.py \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --output /path/to/new-packed-fixture.gguf
build-cuda/bin/llama-qwen38-cuda-probe /path/to/new-packed-fixture.gguf all 1 all
build-cuda/bin/llama-qwen38-cuda-probe /path/to/new-packed-fixture.gguf all 8 all
build-cuda/bin/llama-qwen38-cuda-probe /path/to/new-packed-fixture.gguf all 0 ple
build-cuda/bin/llama-qwen38-cuda-probe /path/to/new-packed-fixture.gguf all 0 reject
build-cuda/bin/llama-qwen38-cuda-admission "$MODEL" 23 --allocation-only
build-cuda/bin/llama-qwen38-cuda-admission "$MODEL" 24 --allocation-only
```

Run only the case relevant to a new change; these commands are not an automatic
verification gate. The allocation diagnostic also accepts the real first shard
and never evaluates it. The observed rejection occurs before payload values
could influence inference.

The token-row follow-up removes another bounded resident-weight allocation
and passes state/graph admission and a short real-model logits/state comparison
under the same 4/2 GiB caps. Server support, live resize,
long contexts, 32-row matrix execution and other quantizations remain open.
