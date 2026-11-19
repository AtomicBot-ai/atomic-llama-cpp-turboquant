# Native inference versus clean upstream

## Scope and decision

No repeatable speed, engine-buffer memory or numerical regression was found in
this focused native Qwen/Metal comparison. No runtime change was made in response.
This is evidence for one profile, not a guarantee for all models or backends.

The control is pristine upstream commit
`31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`, the fork's base, not an unrelated newer
release. An isolated archive in `/tmp/atomic-engine-clean-upstream-31953bcf` was
verified against all 4,164 tracked Git blob hashes and built separately. No fork
source files were inserted into that tree. The archive has no Git metadata, so
embedded build-identification strings may report unknown; the commit identity is
established by the blob verification.

## One short comparison profile

- Apple M4 Max, existing Ornith 1.5 35B A3B Q4_K/IQ4_XS GGUF.
- Context/batch 512, microbatch 32, one sequence, eight CPU threads, 99 GPU layers,
  default native model loading/mmap, greedy sampling, no expert profiling.
- The existing `atomic-engine-prefill.txt` prompt produces 190 input tokens and
  16 committed output tokens; there are 15 single-token decode calls per request.
- Streaming and engine budgets are disabled through the library defaults. The
  probe does not include or call any fork control API.
- Release/static CPU+Metal builds, Accelerate on, BLAS and KleidiAI off, native
  CPU tuning and embedded Metal library on. Compiler flags/definitions match for
  llama, ggml-base, ggml-cpu and ggml-metal.

The identical source `tools/expert-source/native-probe.cpp` is compiled separately
against each library's own headers and archives. This is necessary because the
fork's model parameter struct extends the upstream ABI. It uses only APIs present
in the clean upstream tree. It is not the fork's streaming diagnostic runner.

Primary order is upstream/fork/fork/upstream: four sequential processes, each with
one warmup request and two measured requests. No compilation or other test workload
runs concurrently. Phase timers explicitly synchronize the backend. Logits go to
a preallocated buffer; file writes and memory snapshots are outside phase timers.
Request time includes sampling and the identical in-memory logits copy. State
serialization happens after every timed request has completed.

## Primary results

Medians over four measured requests per variant, upstream then fork:

- Prefill: **367.986 ms / 366.608 ms** (-0.37%).
- Fifteen decode calls: **165.834 ms / 160.026 ms** (-3.50%).
- Request: **536.329 ms / 528.642 ms** (-1.43%).

Fork request times range from 527.813 to 529.446 ms; upstream ranges from 529.657
to 551.152 ms. The second upstream process is slower than the first. These small
samples share within-process state and do not establish a statistically reliable
speedup. They do not show a slowdown in the tested native path.

Every saved FP32 logits file is byte-identical across variants and repeated
requests. All committed token IDs, prompt IDs, stop status and serialized state
match; state FNV-1a is `9755c384a23111fe`. No output tolerance is used.

## Memory and the additional targeted control

Backend-reported buffer sizes match byte for byte in every process, compared by
buffer type. Raw pointer-keyed enumeration order can differ between processes:

- Model: 20,114,934,272 bytes, including mapped model buffers.
- Context: 76,349,440 bytes.
- Compute: 33,950,784 bytes.
- Total: 20,225,234,496 bytes.

RSS and macOS physical footprint are reported separately; footprint alone does
not represent the approximately 20 GB of mapped weights.

The first upstream process performs approximately 5.2 GB of process-attributed
disk reads during loading and has lower RSS than subsequent warm-cache processes.
Combining it with the later samples produces an apparent +55.4 MiB median RSS
increase for the fork, despite exactly equal backend buffers. This was treated as
an unresolved memory comparison, not immediately classified as a regression or
silently removed from the raw results.

One additional upstream/fork pair, on the same profile and already warm model
cache, was run specifically to investigate that discrepancy. Each process still
has one warmup and two measured requests. Their median request times are
529.775/528.250 ms. All additional logits/tokens/state remain exact.

Across the warm-cache processes, upstream RSS is 19,449.344–19,450.359 MiB and fork
RSS is 19,448.609–19,450.859 MiB. Physical footprint is 406.421–407.374 MiB upstream
and 405.546–407.796 MiB for the fork. Differences change direction between process
pairs, with a roughly 2–3 MiB process-to-process spread. There is no stable extra
engine allocation in these results; smaller structural costs are not ruled out.
Load times and the mixed cold/warm RSS median are not used as performance claims.

## Reproduction and limits

The opt-in `llama-native-probe` CMake target is excluded from the default build.
Build it explicitly, then pass `MODEL PROMPT_FILE OUTPUT_STEM`. It deliberately
runs only this fixed short profile. To compare with clean upstream, compile the
same source against the pristine library and its headers; do not link one variant
against the other's public struct definitions. Exact manual build commands used
here are retained in `/tmp/atomic-engine-native-probe-build-commands.json`.

Raw logs/reports/logits use `/tmp/atomic-engine-native-compare-{index}-{variant}`.
Indices 0–3 are the primary ABBA comparison; indices 4–5 are the memory follow-up.
There are six processes, six warmup requests and twelve measured short requests
in total. The optional target initially required CMake regeneration before Make
recognized it; the subsequent target build succeeds. No inference run failed.

No full suite, aggregate verification, E2E matrix, second model/backend, prolonged
thermal/throughput test, cold-start comparison or independent RSS attribution
study ran. The environment's file cache and background activity are uncontrolled.
Existing ABI and small integration-check costs remain described in the
[disabled-feature audit](atomic-engine-disabled-path.md).

Hashes, compile flags, raw per-request values and the separate primary/follow-up
summaries are in `native_upstream_comparison` in `atomic-engine-evidence.json`.
