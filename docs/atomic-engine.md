# Atomic Engine: baseline and streamed inference

## Status

The [fixed-budget server profiles](atomic-engine-release.md) list the
qualified launch settings, optional accelerators and support limits.

Qwen3.8 Flash Next, identified in the local GGUF as `qwen4exp`, has passed
the scoped Metal text-streaming gates below. Its implementation sequence and
acceptance gates are in [the Qwen3.8 plan](atomic-engine-qwen38.md).
P11 now also passes the single-slot LFM2.5-24B CUDA server profile with
4 GiB RAM / 2 GiB VRAM, including stop/EOS, SSE cancellation, file-based
slot restoration and exact short-profile logits/state. See
[the CUDA server evidence and boundaries](atomic-engine-cuda-server.md).
Qwen3.8 CUDA now passes a scoped short-request backend qualification gate.
CUDA PLE row reads and all seven routed formats now pass scoped operation
comparisons at microbatches one and eight, including ten selected experts.
Streaming the token embedding rows additionally removes 260.5 MiB of resident
weights. At 22 GPU layers, the 4 GiB RAM / 2 GiB VRAM profile now completes
a five-token prefill and four greedy outputs with byte-identical full logits
and serialized state against ordinary CUDA. See the
[token-row evidence and narrow limits](atomic-engine-qwen38-cuda-token-rows.md).
The subsequent [long CUDA control](atomic-engine-qwen38-cuda-long-prefill.md)
now passes the 4/2 GiB memory gate and exact logits/full-state comparison for
3,481 tokens at context 4,096. A streaming-only routing-weight copy preserves
CUDA fusion without changing ordinary execution or the runtime reserve.
The [manual CUDA resize control](atomic-engine-qwen38-cuda-resize.md) now
passes at context 2048, including worker-read cancellation, completed cache
release and exact logits/state. Context 4096 rejects resize with the unchanged
64 MiB allowance and remains fixed-budget. The selected
[Qwen CUDA raw-completion server lifecycle](atomic-engine-qwen38-cuda-server.md)
now passes stop, SSE-disconnect recovery and file-based slot restore, with
exact matching-history states and admitted file staging. A selected
[non-thinking chat/EOS control](atomic-engine-qwen38-cuda-chat.md) also passes
with identical template, output IDs and complete saved state. A selected
[two-turn dialogue](atomic-engine-qwen38-cuda-chat.md#two-chat-turns-in-the-same-slot)
reuses 17 live tokens on the follow-up, with exact full logits and slot files.
The selected [long HTTP prefill](atomic-engine-qwen38-cuda-http-prefill.md) now also passes:
3,481 tokens across 28 server batches, identical output IDs and full slot
bytes within 4/2 GiB, with the reserve unchanged. The selected
[long-slot restore](atomic-engine-qwen38-cuda-http-restore.md) passes in fresh
servers, including exact-prefix continuation with 3,484 cached tokens and
byte-identical full slot state. The [same-slot long HTTP series](atomic-engine-qwen38-cuda-http-series.md)
passes an exact repeat and a one-token edit through full re-evaluation, with
matching output IDs and complete slot bytes at 4/2 GiB. The qualified opt-in
[Linux disk prefix profile](atomic-engine-release.md#cuda-disk-prefix-checkpoints-opt-in)
reuses 3456 tokens on exact repeat and 1664 on the selected edit under the
same caps, with exact full logits and complete slot bytes. Missing, truncated
and same-size corrupted checkpoints trigger exact same-slot full prefill.
The selected [long two-turn chat](atomic-engine-qwen38-cuda-chat.md#long-two-turn-chat-over-sse)
reuses 3473 of 3493 input tokens on the follow-up from the live slot,
with exact full logits and complete slot bytes. A selected [tool call and result](atomic-engine-qwen38-cuda-chat.md#single-tool-call-and-result-in-the-same-slot)
reuses 322 of 349 tokens, also with exact full logits and slot files.
Parallel calls, automatic pressure handling, broader context coverage and
microbatches above eight remain open.
Bounded PLE row reads now support the tested 8 GiB Metal profile.
Short single-slot completion/chat, stop, EOS, SSE cancellation and
slot restoration have passed. A short ordinary Metal control using the
upstream host-expert cache now matches streamed logits byte for byte,
with identical token IDs and serialized-state digest. Its cache uses
original packed weights and enough slots for the checked microbatch.
Qwen4Exp vector prefill now overlaps bounded demand reads with Metal
execution. Two unprofiled comparisons reduced request time by 20.4%
and 21.6% on the checked short request, with exact logits and the
same 8 GiB budget; the family plan records the scope and reproduction.
Subsequent cache work removed linear LRU victim scans (13.8–14.2%
additional short-request reduction) and qualified uniform expert slots
on Metal (9.2–10.3% additional reduction). Slots include their full
physical padding in the ledger; exact logits/state and live shrink/grow
remain checked. These warm short-profile gains do not qualify cold I/O
or long-context throughput.
Qwen4Exp Metal vector submissions now group up to sixteen experts, with
budget-limited demand lookahead and a smaller-workspace fallback. On the
short profile, moving from eight to sixteen groups reduced submissions
from 4,230 to 2,271 while preserving exact logits/state and the 8 GiB cap.
Other families and matrix-wave prefill retain four-group compute limits.

The Qwen3.8 Metal single-slot profile now has a scoped context-4096
qualification at 8 GiB: a 3,481-token prompt, its full repeat and a
33-token follow-up. All 48 output-logit rows and three serialized states
match the ordinary Metal control byte for byte. Managed high-water stays
below 8 GiB across the series. A subsequent I/O profile found 1.66 TB of
logical expert reads in the long prefill. Bounded macOS file-cache advice
for the next known-demand group now reduces paired long TTFT by
5.4% / 14.2% at 8 GiB, with exact logits/state and unchanged managed
high-water. Short requests, decode and ordinary execution retain their
I/O path. See the family plan and prefill evidence for timing methodology,
file-cache limits and the unqualified multi-slot/long-soak boundaries.

The same automatic long-prefill gate now uses current decayed frequency
and both class-queue endpoints when selecting unpinned cache victims.
Uniform slots and explicit ClassLRU remain intact. On the 3,481-token
fixture, prefill reads fall by 24.7 GB (1.48%); decode reads rise by
0.192 GB for sixteen outputs, and the short follow-up's reads are unchanged.
Both candidate repetitions preserve all logits/state bytes and the 8 GiB
cap. TTFT reductions were 8.48% / 0.74% with uncontrolled file-cache state;
this is a measured reduction in logical work, not a stable speedup claim.
See the family plan's retention section and structured retention evidence.

Qwen3.8 Metal now has an opt-in full-prefill window path:
`--engine-prefill-window`, with the fixed 8 GiB / context 4096 / batch 256 /
microbatch 8 / one-text-slot profile. Eight independent microbatches visit each
layer in sequence, preserving the original arithmetic and PLE-history ordering.
Compact scheduler ownership reduces the metadata reservation from 1.63 GiB to
512 MiB. The full 3481-token request now reads 794.3 GB instead of 1638.5 GB
(51.52% less), exactly matching the trace prediction. Two paired runs reduce
TTFT from 167.47 / 163.00 s to 125.43 / 122.15 s (25.10% / 25.06%). All full
logits and long/short slot states match ordinary Metal byte for byte; managed
peak is 8,589,922,714 bytes, below the 8 GiB cap.

The option stays off by default. A subsequent profile found that post-window
decode spends about one second in expert reads versus 48 ms in allocation.
Known-route file-cache advice now remains available for single-token projections
after an opted-in window. In a reversed-order unprofiled server pair, long-request
decode improves from 7.38 to 8.32 tokens/s (12.70%); the short follow-up improves
from 8.02 to 8.87 tokens/s (10.62%). TTFT remains about 124 s, and all logits/slots
and the 8 GiB peak remain exact. This does not remove the 645 post-window buffer
allocations or reduce logical decode bytes. A warmed prefix-only diagnostic had
no stable gain; the result is scoped to this server series, not universal cache
conditions. Option/policy/overlap disable, 7 GiB and never-enabled controls issue
zero additional advice calls. See [decode evidence](atomic-engine-qwen38-decode-advice.json).
Admission failure falls back before state mutation. A request for 8 to 7 GiB
from another thread during a real projection completes the current window,
frees its resources, and resumes the rest sequentially with exact logits/state;
growth back to 8 GiB also passes. Windowed P11 now passes SSE cancellation at
the server batch boundary, recovery, stop/EOS, and explicit file-based rollback.
Two additional long prompts and a same-server repeat match ordinary Metal logits
and serialized slots byte for byte at the fixed 8 GiB engine cap. The windows
remain opt-in. Native pressure, automatic checkpoint rollback, longer soak and
multi-slot execution remain open. See [window lifecycle evidence](atomic-engine-qwen38-window-p11.json)
and [the initial window comparisons](atomic-engine-qwen38-window-server.json).

P00 is complete for the initial local development baseline: checkout provenance,
reproducible targeted build, native committed-token accounting, a real request,
a short warmed A/A comparison, memory/I/O observations, and scoped legacy risks.
This does not qualify throughput, cold SSD performance, or a total-memory cap.

M1 is now demonstrated for the initial Apple M4 Max / Ornith 35B profile:
original-GGUF prefill/decode runs with a 5 GiB engine budget, including resident
nonexpert weights, state, main graph buffers and a reserved runtime envelope.
Microbatches 32 and 128 match resident logits; the complete-answer probe ends at
EOG. Manual P04 shrink/grow preserves logits and serialized state and releases
real backend memory. These results do not qualify other architectures, larger
contexts or general performance.

P05 now adds opt-in automatic pressure control: deterministic replay precedes
runtime integration, and a real-model signal replay preserves full logits/state
while releasing Metal buffers. Shared CLI/server configuration and macOS signal
subscription are connected. Native warning/critical delivery under system stress
remains unqualified; see [policy, reproduction and limits](atomic-engine-pressure.md).

See [the current memory contract, live API and reproduction commands](atomic-engine-memory.md).
The sections below retain the source/store and first graph-integration history;
the older expert-only cap is still available as a distinct diagnostic mode.

P06 now provides demand traces and an opt-in, accounted ClassLRU policy.
The two-workload replay and live-control decision are in
[the cache evaluation](atomic-engine-cache.md); LRU remains the default.

P10A now qualifies a second architecture on Metal: LFM2.5-24B-A2B (`lfm2moe`)
preserves resident full logits and state for short decode and useful prefill
with live shrink/grow. See [the scoped support profiles and evidence](atomic-engine-second-architecture.md).
The [LFM streaming profile](atomic-engine-lfm-performance.md) then removes
mixed-size allocation churn and evaluates upstream-derived ClassLRU on ATET2
traces without changing the default policy.

The synchronous prefill path now reuses completed expert allocations and batches
independent vector nodes for one expert. See [the phase profile and acceptance
results](atomic-engine-prefill-performance.md); the historical timings below
precede these optimizations.
The [LFM GPU follow-up](atomic-engine-lfm-gpu.md) now groups up to four independent
waves per submission, admitting all workspace and retaining leases through
completion. Near the working floor it reduces the group to one.
The [LFM read follow-up](atomic-engine-lfm-reads.md) removes physical staging and
the subsequent upload copy for CPU/shared Metal expert slots while preserving
the same conservative I/O headroom, file validation and publication boundary.
The [LFM overlap follow-up](atomic-engine-lfm-overlap.md) reads up to four known
subsequent demand experts during GPU matrix-wave execution. It is enabled by
default for LFM/Metal and falls back when budget headroom is insufficient.
The [Qwen overlap qualification](atomic-engine-qwen-overlap.md) extends that
default to `qwen35moe` on Metal after matched on/off, all-hot and live controls
on Ornith 35B with IQ4_XS. Other Qwen architectures remain opt-in.

Structured measurements, model identity hints, source hashes and build flags are
in [the evidence record](atomic-engine-evidence.json).

### LFM2.5 24B partial wide prefill

The single-slot Metal profile with BF16 KV, a 4096-token context and batch,
and an 8-token microbatch uses wide prefill by default for fresh prompts from
320 tokens at fixed budgets of 8–24 GiB. For a partial block it binds resident
expert weights to the graph and stages the missing layers. The usual memory
admission and fallback still apply. At 16–18 GiB the final banks can serve
decode for the same request. Set `LLAMA_LFM_PARTIAL_WIDE_PREFILL=0` before
starting the server to restore the previous partial-block behavior.

The [handover comparison](atomic-engine-lfm-handover-8-24gib.json) covers
858 input tokens and 64 greedy outputs at nine budgets. Using the wider graph
at 64 input tokens changed a greedy token in a separate control at 18 GiB.

### Qwen3.6 35B with a resident verifier at 24 GiB

With the BF16 DFlash2 sidecar alongside the Q4_K_M target, start the qualified
single-slot Metal profile with the following command.

```sh
LLAMA_QWEN35_RESIDENT_VERIFY=exact python3 tools/expert-source/run-qwen35-metal-profile.py 24
```

The target keeps all 40 expert layers resident and passes prompt features to the
drafter in batches of up to 512 tokens. Verification retains its eight-row
microbatch limit. The `exact` setting uses the ordinary attention kernel.
Set the variable to `1` to allow grouped attention on qualified BF16 verify
graphs on Apple GPU family 9. This kernel uses half operands and float
accumulation. On the 4000-token fixture the grouped run preserved the entire
greedy continuation, while the exact run also preserved every verification
logit bit for bit. Leave the variable unset to select the calibrated defaults.

The [24 GiB qualification record](atomic-engine-qwen36-resident-verify-24gib.json)
includes the three individual client timings per mode, physical and ledger
peaks, per-block Splash replay, and the full-vocabulary comparison.

### Grouped attention with the calibrated Qwen memory profiles

At exactly 26 GiB, the resident verifier can use the same 40-layer target
layout with an adjacent BF16 DFlash2 sidecar. Select grouped attention with
`LLAMA_QWEN35_RESIDENT_VERIFY=1 python3 tools/expert-source/run-qwen35-metal-profile.py 26`.
This opt-in uses the single-slot fixed-budget Metal profile and six proposals.
The [912-token handover comparison](atomic-engine-qwen36-handover-resident-26gib.json)
records both control configurations, client timings, memory peaks and output IDs.

Use `LLAMA_QWEN35_GQA8=1 python3 tools/expert-source/run-qwen35-metal-profile.py 8`
to select grouped attention independently of expert residency. The same flag
works with the calibrated 10–24 GiB budgets. At 8 GiB it handles one serial
decode row at a time. With DFlash2 it handles the target's multi-row draft
verification. The existing profile selects the number of resident layers and
the prefill batch. The Metal kernel requires the qualified Qwen head geometry,
BF16 K/V and 4096 physical KV rows. Other shapes use ordinary attention.

The [budget sweep](atomic-engine-qwen36-gqa8-budget-sweep.json) compares both
attention kernels at equal residency and includes full-vocabulary diagnostics
at 8 and 12 GiB.

### Qwen hybrid prefill at 8–24 GiB

On a qualified wide prefill, each Metal layer graph borrows its resident expert
weights when available. The staging banks load the missing layers. The
20–24 GiB profiles can retain the last two banks for decode; the next
uncached prompt releases these before admitting its prefill workspace.
Set `LLAMA_QWEN35_HYBRID_PREFILL=0` to
compare against the original wide staging path.

The fixed-budget Qwen Metal profile uses fast prefill by default at 8–24 GiB.
Start the server with the budget alone:

```sh
python3 tools/expert-source/run-qwen35-metal-profile.py 12
```

Fresh shorter prompts qualify for wide prefill from 64 tokens, or 128 on
the 10 GiB four-layer DFlash2 profile. At 24 GiB the qualified BF16 draft
configuration uses all 40 resident layers. Changing the prefill geometry can
change the highest-scoring token even with greedy decoding. The
[known issue and reproducer](atomic-engine-qwen36-fast-prefill-known-issue.md)
describe the logit reversal and the observed effects on complete answers.
Pass `--no-fast-prefill` or set `LLAMA_QWEN35_FAST_PREFILL=0` before
startup to restore the previous profile. Explicit per-feature diagnostic
overrides take precedence. The [Qwen and LFM default timings](atomic-engine-qwen36-lfm-fast-default-8-24gib.json)
cover the same handover text across nine budgets.

At 24 GiB the 912-token handover measured 109.0 decode tokens/s with 44 of
103 draft proposals accepted. The earlier 156.8 tokens/s came from the
4000-token fixture with a 26 GiB budget and 40 direct layers during decode
(49 of 79 proposals accepted). A subsequent 24 GiB production run on that
same 4000-token fixture measured a median of 151.0 tokens/s with the grouped
resident verifier. These measurements span different builds. They do not
isolate the cost of a verification step across prompts.
[Four-layer experiment](atomic-engine-qwen35-four-moe-direct-ab.json) and
[24 GiB production measurements](atomic-engine-qwen36-resident-verify-24gib.json).

The [8–24 GiB comparison](atomic-engine-qwen36-hybrid-prefill-8-24gib.json)
records client timings, physical peaks, logical expert reads and output IDs.

### Compact Qwen draft at 8 GiB

The launcher looks for the qualified Q4_0 DFlash2 sidecar with BF16
selector tables beside the standard target filename and in the local
workspace model cache. It verifies the recorded size and SHA-256 before
automatic selection. If the file is absent, 8 GiB uses serial decoding.
For a direct server invocation, supply the compact draft with `-md`.
Set `LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=0` to disable drafting explicitly.

One direct expert layer leaves room for four prefill workers and a single
pass over the expert weights on the 4000-input workload. On the
qualification prompt the compact draft reproduced the BF16 speculative
continuation; the serial continuation differed. The [low-budget comparison](atomic-engine-qwen36-low-budget-draft.json)
records the individual requests, accepted proposals, memory peaks and the
10 and 12 GiB controls.

## P00: checkout and reusable components

Project and upstream base are `31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`.
The upstream `master` ref was checked and matched. Origin is
`git@github.com:AtomicBot-ai/atomic-engine.git`; upstream is
`https://github.com/ggml-org/llama.cpp.git`. Work is on
`codex/expert-weight-source`; implementation changes are uncommitted. No upstream
update is mixed into this change. Existing user deletions are preserved.

The hardware is Apple M4 Max (Mac16,5), 51,539,607,552 bytes RAM. The build uses
Darwin arm64, Apple clang 21.0.0, CMake 4.3.0, Release, static libraries, embedded
Metal kernels, and no BLAS. Metal graph optimization is at its default setting.
No concurrent build or inference ran during the comparison requests.

Useful code in the current checkout:

- `llama-model-loader`: GGUF parsing, split discovery, tensor shapes and loading.
- `llama-graph::build_moe_ffn`: actual router IDs, shared branches, activation,
  scales/biases and expert reduction. Only routed matrix operations are replaced.
- `llama-moe-cache`: an existing cache of host-resident expert copies. It is not
  disk streaming or a total-memory limit, and cannot be combined with this path.
- GGML backend allocation, scheduler copies and synchronization: reused for
  physical buffers, CPU/Metal transitions and completion-safe lifetime.
- Existing native sampler, batch and memory APIs: reused by the diagnostic runner.
  Server/SSE counters and speculative decoding are not involved.

The separate legacy experiments remain untouched. Inspected source families are
pinned to `41ec4c4e94fd5ff6c258691f35f2fcd0d3dde892` and
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`. The former contains compact layer pools,
parallel reads, shared-event publication, generation guards, next-layer preview,
Top4 prefetch and early shared FFN; the latter has a synchronous bounded Metal
control. Their observer/patch ABI and fixed pool geometry do not match this base.
No code from these experiments was copied. Their ownership lessons and failure
cases inform the new source/store integration; historical performance percentages
are not transferred. No local experimental branch with those features exists in
this checkout.

Scoped follow-up ownership:

- P03 allocator/accounting: historical cache-plus-aux overflow is addressed by
  explicit expert/workspace scope now; admit all remaining owners before M1.
- P07 execution/preview: requalify preview and shared-first for batched graphs;
  old guards disabling them are not evidence that current batches are supported.
- P08/P09 speculation/state: requalify hybrid-state rollback and generation retirement
  on this base before enabling MTP or trees. Prior evidence is configuration-bound.
- P00 measurement: native committed IDs are authoritative; SSE text chunks can
  merge incomplete UTF-8 tokens and cannot supply the same counter.
- P06 benchmark: match Metal graph settings and build hashes when comparing an
  accelerator. The current resident path is the numerical control, not an
  equal-total-cap performance competitor.

## Source, store and ownership

`llama-expert-source` indexes original GGUF metadata and owns pinned file handles.
Explicit complete shards retain their own offsets. Reads preserve original packed
quantization blocks; no sidecar, repacking or precision reduction is introduced.
Supported storage is contiguous `[input, output, expert]` with complete blocks.
Shard numbering/counts, tensor totals, duplicate names, shapes and file bounds are
checked. Secondary shards may omit architecture metadata. File size and timestamps
are checked around reads. Files must remain immutable; these checks are not
cryptographic authentication. A failed/changed source is permanently invalidated;
a partial read is never published. OS page caching remains enabled.

`llama-memory-budget` has move-only tickets for reserved, allocated and retiring
bytes. Admission happens before payload allocation; freeing the resource precedes
returning its charge. UMA aliases share one ledger. The underlying ledger also
supports separate host/device domains, but the inference bridge is qualified only
for CPU and Apple Metal, not discrete GPUs.

`llama-expert-store` owns individually releasable expert backend buffers. Misses
admit metadata arena, backend allocation and read staging together. Publication
follows a complete direct host read, or a staged upload and synchronization.
Direct publication does not drain unrelated pinned GPU work. LRU eviction considers only
unpinned entries. Leases retain the generation and synchronize before unpinning.
Pinned shrink entries remain charged until their last lease ends. Internal store
shrink/grow is implemented and tested; that is not an active-session resize API.

The older expert-only managed cap includes expert payload buffers, their tensor arenas and staging,
plus the executor's tensor arena, backend workspace and host gather/scatter buffer.
The bridge rounds backend reservations to 64 KiB. Source/control metadata,
backend pipelines, kernel file cache, resident nonexperts, KV/recurrent state and
main scheduler scratch are outside this cap. That older setting is not a total-memory promise. The new engine budget adds
model/context/scheduler admission and the explicit runtime envelope described
in the memory contract.

## P02 L2: graph integration and acceptance

The implementation extends `llama_model_params` with `expert_streaming_size`;
zero preserves resident loading. Recompile clients using this struct. The loader
keeps routed tensors as metadata only and disables whole-file mmap. Nonexpert
weights still load normally. Original router/shared/state graphs remain in place.

`llama-expert-executor` replaces only routed projections with a CPU custom-op
bridge to a separate synchronous compute backend. It receives actual router IDs,
groups selected rows by expert, acquires a lease per referenced expert, and computes bounded
waves. The batch union never has to fit in cache. The inner operation is the
original `MUL_MAT_ID`, with a local zero ID for the loaded expert. Metal matrix
waves use at most 32 rows and preserve the upstream matrix/vector selection;
other requests use vector rows. Padding consumes reserved workspace, not extra
experts. Unused outputs are discarded and selected rows scatter to original slots.

Preserving surrounding backend placement is essential: allowing the CPU bridge
to pull MoE activations/reduction onto CPU changed long-prefill logits even when
all committed IDs matched. These operations now stay on their layer's original
backend. This, together with matching indexed matrix kernels, restored bitwise
agreement in the measured request. Future upstream kernel-selection changes must
revisit this bridge's policy and its numerical comparison.

Integrated steps and invariants:

1. Keep original expert weights out of model allocations and execution graphs.
2. Admit the largest expert plus fixed workspace before computation; reject an
   impossible cap rather than silently falling back to resident loading.
3. Preserve routed IDs, input strides, per-slot activations, shared branch and state.
4. Hold every expert lease through its submitted row waves and synchronize before eviction.
5. Catch callback failures inside worker threads, reject the entire evaluation,
   and permanently reject further evaluations on that context. Discard the context;
   rollback/recovery after failed computation is not implemented.
6. Compare native logits/IDs against resident execution and observe actual
   misses, bounded high-water and allocation counters on the same configuration.

The capability gate accepts file-backed Qwen3, Qwen3.5 MoE and LFM2 MoE, CPU/MTL0,
one sequence/default context, and F32/F16/Q4_0/Q4_K/Q5_K/Q6_K/IQ4_XS expert storage.
Qwen3.5 MoE and LFM2 MoE have scoped real full-model Metal evidence. Multi-device/tensor
split, MTP, LoRA, folded weights, custom expert precision policies and full tensor
validation are rejected. Fused storage is understood by the source/graph but is
not qualified by the real checkpoint comparison. CUDA, Windows/Linux, multiple
concurrent contexts remain unqualified. The feature stays opt-in; the shared
`--engine-budget-mib` CLI explicitly selects total-budget streaming.

## Reproduction

Configure once; build only targets needed for the selected check:

```sh
cmake -S . -B build-metal \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_OPENSSL=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
  -DGGML_BLAS=OFF -DGGML_CCACHE=OFF
cmake --build build-metal --target llama-expert-infer -j4
```

Set `MODEL` to the local original checkpoint; models are not downloaded. The
native runner uses greedy sampling with no chat-template processing. It records
committed output IDs (excluding EOG), full request wall time, separate prefill time,
optional full logits, backend buffers, OS RSS/footprint and process-charged reads.
A fixed output limit is a diagnostic prefix, not necessarily a completed answer.

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --output /tmp/resident-aa.json --logits /tmp/resident-aa.f32 \
  --tokens 8 --ctx 512 --ubatch 16 --warmup 1 --repeat 2

build-metal/bin/llama-expert-infer --model "$MODEL" \
  --prompt-file docs/atomic-engine-prefill.txt --tokens 16 --ctx 512 --ubatch 32 \
  --output /tmp/resident.json --logits /tmp/resident.f32

build-metal/bin/llama-expert-infer --model "$MODEL" --store-mib 64 \
  --prompt-file docs/atomic-engine-prefill.txt --tokens 16 --ctx 512 --ubatch 32 \
  --output /tmp/streamed.json --logits /tmp/streamed.f32
cmp /tmp/resident.f32 /tmp/streamed.f32
```

The completed resident A/A had identical eight-token traces and bitwise-identical
logits. Request times were 0.101351 and 0.102252 seconds, about 0.9% apart. This
estimates only the order of short warm-request noise, not a confidence interval.

The matched 190-token prefill/16-token decode comparison took 0.9540 seconds
resident versus 19.8681 seconds streamed; prefill was 0.7023 versus 16.5173 seconds.
This is a slow correctness baseline, not a speedup or equal-total-cap comparison.
The streamed run had 68,355 expert misses, 38,824,181,760 logical read bytes,
66,692,912 peak managed bytes under a 67,108,864-byte cap, and zero cache hits.
The cache cannot retain the cross-layer working set in this tiny-cap condition.

Resident model buffers total 20,114,934,272 bytes; streamed nonexpert model buffers
total 2,666,629,632 bytes. Metal allocated size after the request was
19,684,589,568 versus 2,302,033,920 bytes. These are allocations/mappings, not
independent quantities to add to RSS. Streamed process footprint was
3,714,813,240 bytes; it is not the 64 MiB managed-cap measure. The evidence record
retains context/compute buffers and OS counters separately. Physical process reads
are OS charges, not model-only or NAND bytes; cache state was not controlled.

## Targeted checks and remaining work

Completed checks are limited to the new behavior:

- Source: seven cases / 121 assertions across selective runs, including original
  bytes across shards, packed formats, invalid layouts, changed/truncated files,
  and concurrent cursor safety.
- Store: six CPU cases / 48 assertions; one Metal projection case / seven
  assertions. Includes failed admission/read rollback and controlled delayed
  completion, LRU/pins and shrink/grow.
- Executor: five CPU cases / 50 unique assertions; two Metal routing cases / 42
  unique assertions. Shared and per-slot strided input, batch unions over capacity,
  padded and multiple waves, invalid IDs, poisoned source and impossible floor.
  Only the affected matrix-wave case was rerun after extending its edge coverage.
- Real source/store sample: both Ornith 122B shards, 294 slices,
  503,709,696 logical bytes, matching sample digest `9ab0cd2acd41857c`.
  An 8 MiB store probe released all managed allocations; Metal allocation dropped
  by 2,162,688 bytes at final shrink. This was not full 122B inference.
- Real Ornith 35B prefill/decode: final full logits and committed IDs match the
  resident control. Earlier divergent projection/placement attempts are retained
  as diagnostic failures, not accepted equivalence evidence.

No full test suite, aggregate verify, full coverage or broad matrix ran. P03 admission and manual P04 controls now have their own targeted evidence in
the memory contract. P06 profiling and P05 control now have separate targeted
evidence. P10A now qualifies the second architecture on the same backend. Native
pressure-event qualification remains open. P11 is qualified for the fixed-budget
Qwen/Metal release profile below. Known-route read/GPU
overlap is qualified for LFM2 MoE and Qwen3.5 MoE. Next-layer preview now has
read-only capture and bounded offline replay evidence in
[the P07 evaluation](atomic-engine-preview.md). The subsequent
[Qwen runtime preview](atomic-engine-qwen-preview-runtime.md) preserves measured
outputs and budget but does not demonstrate acceleration; it remains an explicit
experiment, off by default. The [cost follow-up](atomic-engine-qwen-preview-cost.md)
reduces predictor and speculative I/O work but still does not pass the request
latency gate. The [optimization audit](atomic-engine-optimization-audit.md)
binds preview to actual graph geometry, eliminates measured excess read bytes,
and reduces contiguous-row gather cost; it makes no whole-request speedup claim.
Shared-first and
chain speculation remain open.

The [known-expert Metal dispatch optimization](atomic-engine-single-expert-map.md)
is enabled for the qualified Qwen3.5 MoE and LFM2 MoE streaming matrix waves.
CPU-prepared maps remove two GPU dispatches and their internal dependency barrier
per wave while retaining the original matrix kernel and existing memory footprint.
Measured request reductions are 32–35% on Qwen and 2–3.4% on LFM; exact outputs
and the shared memory cap are preserved in the targeted profiles.

The [bounded route index](atomic-engine-route-index.md) removes repeated ID scans
while preserving expert/row order and GPU grouping. Budgeted Qwen3.5 MoE Metal
contexts attempt its 22,530-byte admission after mandatory allocations; LFM stays
off by default because its request-level improvement was not established. Qwen's
later comparison pairs were 2.8–4.7% faster, with I/O variation contributing to
the observed difference. Exact outputs, the added floor and live resizing have
targeted qualification; the original scan remains available.

The [exact early-read experiment](atomic-engine-early-reads.md) moves the next
independent projection's first four actual experts under the source's tail GPU
group. It preserves full outputs and fixed/live budgets on the targeted profiles,
but remains opt-in: the eligible GPU windows total less than 0.3% of request time
and the matched comparisons do not establish an attributable speedup. Its optional
binding workspace adds 30,728 bytes to the working floor. Main-graph shared-FFN
scheduling is a separate, unqualified hypothesis.

The [shared-FFN scheduling probe](atomic-engine-shared-schedule.md) demonstrates
earlier exact-ID availability and ordinary demand reads before shared GPU
completion, with exact Qwen outputs and the same 5 GiB cap. Its measured shared
GPU service is only 28–30 ms/request and the serial/overlap control does not
establish a speedup. The prototype was archived and active source restored; no
shared-schedule mode is shipped. This closes the current three-direction
optimization wave without claiming completion of the engine roadmap.

## P11 fixed-budget release profile

P11 is complete for the checked Ornith 1.5 35B A3B Q4_K/IQ4_XS GGUF
(Qwen3.5 MoE, original packed weights) on macOS M4 Max Metal (`MTL0`). The
qualified server settings are a fixed 5 GiB managed budget, one context and
one slot, context/batch 512, microbatch 32, eight CPU threads, and greedy text
completion. The budget accounts for engine allocations and a runtime envelope.
System page cache, OS memory pressure and process RSS have separate semantics.

- Qualified HTTP surface: first and repeated requests, EOS and literal stop
  strings with token accounting, SSE cancellation and slot reuse, same-server
  slot save/erase/restore, and fixed-budget metrics.
- Qualified engine controls: the [memory contract](atomic-engine-memory.md)
  records resident numerical controls, SSD streaming, live manual resize and
  below-floor rejection through the embedding interface. Resident runs use their
  own memory configuration. The fixed HTTP profile does not expose live resize.
- Included acceleration: bounded known-route overlap, prepared Metal expert
  maps, and the budget-admitted route index. Generic fallbacks are covered by
  their focused executor checks. Preview and early adjacent reads remain opt-in
  experiments. Shared-first is archived.
- Outside this release profile: automatic pressure-driven resize, multiple
  budgeted slots, speculative decoding, multimodal inputs, auto-fit, RAM prompt
  cache and context checkpoints. Other weights, backends, and sampling setups
  need their own qualification. LFM retains separate engine evidence.

The server rejects any other slot count before loading weights when a budget is
configured. An ordinary two-slot server reached readiness without budget gauges.

```sh
build-metal/bin/llama-server -m "$MODEL" -c 512 -b 512 -ub 32 -np 1 \
  -t 8 -tb 8 -ngl 99 -fit off --engine-budget-mib 5120 \
  --cache-ram 0 --ctx-checkpoints 0 --no-warmup --metrics \
  --host 127.0.0.1 --port 8080
```

Submit the contents of `docs/atomic-engine-completion.txt` to `/completion`
with `temperature: 0`, `n_predict: 32`, `return_tokens: true`, `cache_prompt:
true`, and `id_slot: 0`. The first and repeated requests both returned `8` with
token IDs `[23, 248046]`, `tokens_predicted: 2`, and `stop_type: eos`. The EOS
token is counted in the IDs and generation count. Both calls processed 199
prompt tokens with `cache_n: 0`. The recurrent profile can fall back to full
prefill because the checkpoint cache is disabled under the budget.

An SSE client closed its connection after the first token of a 96-token
request with EOS ignored. The server logged cancellation and released slot 0
with 201 tokens in its prompt state, including the original 199 prompt tokens.
The next ordinary completion on that slot returned the same `8` and EOS.
All sampled `/metrics` usage was 5,368,077,426 of 5,368,709,120 admitted bytes.

The automatic configuration also started and completed the same request with
valid advisory headroom and zero pressure decisions. This qualifies its HTTP
startup and metrics. Existing embedding resize/replay and controlled read-error
evidence retains its original scope.

A non-streaming completion with `stop: ["8"]` returned empty content,
`stop_type: word`, `stopping_word: "8"`, and one generated token (ID 23).
On the same server, `save`, `erase`, and `restore` on `/slots/0` reported 200
tokens each. Saving the restored slot again produced a byte-identical
69,966,312-byte file before any further inference. The next completion matched
the original `8` and EOS, with full prefill on this recurrent profile. The
sampled managed budget stayed within 5 GiB. This is a same-server round trip.

A 3000 MiB startup request failed allocation admission before HTTP readiness.
With the 5 GiB profile, changing the timestamp of a disposable copy of the GGUF
after readiness triggered expert source identity validation on demand. The server
returned HTTP 500 (`Compute error.`) and released the slot without emitting a
completion. The context becomes unusable after a source error. Restart the
server with intact weights to create a new context. The existing short-read
fixtures separately cover partial data and lease cleanup.

## Automatic pressure follow-up

An automatic-mode server then completed a request before and after an independent
process touched up to 4 GiB of anonymous memory in 512 MiB steps. Each step was
followed by an idle metrics sample. The host's reported available memory was
73–78%, with no native warning or critical notification and no budget decision.
This bounded load verified continued service and exact response tokens. Native
shrink and recovery in response to actual OS notifications remain unqualified.
The checked scenarios required no runtime source changes.

`tools/expert-source/p11-native-pressure.py` runs the remaining check inside a
dedicated macOS environment with Metal and the same GGUF. Give it the local
server executable, model file, output path, an explicit load limit in MiB, and
`--isolated-mac`. It warms a single automatic-budget server and increases real
anonymous memory load in 256 MiB steps. It stops at shrink, at the load limit,
or at the configured system-availability floor. The JSON report and server log
capture the observed native level, applied cap, freed bytes, recovery and
completion. A passing run requires warning/critical with actual shrink and
release, then normal and recovery to 5 GiB, with the reference answer preserved.

## P08 entry check

The qualified Ornith 35B and the available Qwen3.6 35B have no `nextn` metadata
or draft tensors. The two-shard Ornith 122B has one declared NextN layer and
the layer-48 NextN tensors in its second shard. That layer contains 1,737,697,280
packed bytes, including 1,660,944,384 expert bytes. A draft path that loads the
whole layer would need to account for that payload in addition to its state and
verification workspace. The existing 122B source/store evidence is a slice
probe. It does not establish full model inference or MTP compatibility.

The current MTP integration expects target and draft contexts plus matching
hidden-state rows. Budgeted model loading rejects MTP and the qualified context
disallows recurrent rollback slots. P08 is unavailable for the fixed 5 GiB
release profile.

### Ornith 122B target floor and draft admission

The two-shard Q4_K/IQ4_XS checkpoint completed a full 199-token prefill and
greedy decode on M4 Max Metal with context and batch 512, microbatch 32 and one
sequence. The target ran all 48 trunk layers, leaving 144 expert tensors on
disk. The 10 GiB control returned token 23 (`8`) and EOG 248046. It read
143,007,940,608 logical expert bytes and took 40.16 seconds for the request.
Logical reads can include repeated expert slices and filesystem cache hits.

The engine floor was 9,023,134,786 bytes. At 8605 MiB, context admission rejected
the working set. At 8606 MiB, the full request completed in 52.01 seconds with
peak managed use of 9,023,135,154 bytes. Saved logits, committed token IDs,
EOG and serialized-state digest `012b8232a95ab2b8` match the 10 GiB streaming
control. This is the lowest whole-MiB cap for the exercised configuration.
The comparison uses two streaming caps because a fully resident 122B model
does not fit the host's physical RAM.

The separate draft GGUF retained layer 48 plus `token_embd.weight`,
`output_norm.weight` and `output.weight` from the original shards, with the
original metadata except the split fields. All 23 tensor shapes, types and
payload hashes match the source. The temporary file is
`/tmp/atomic-engine-p08-122b-mtp-only.gguf`. Loading it with MTP enabled,
mapping disabled and extra buffer types disabled charged 3,358,785,536 bytes
for weights and 43,515,904 for draft-context buffers on the target's ledger.
The target floor with draft allocations held was 12,425,436,226 bytes. The
draft context was destroyed before its model, and the drafter charge returned
to zero.

The opt-in `llama-expert-infer` diagnostic used a 12 GiB shared cap with
`--rollback-probe --draft-admission-probe "$DRAFT_GGUF"`. At position 199 it
charged a 161,199,973-byte target snapshot, decoded an intentionally rejected
token 0, and restored the snapshot before decoding accepted token 23. The
serialized state digest matched before and after restore, and the completed
request matched the baseline logits, tokens, EOG and final state digest.
Peak managed use for the combined run was 12,884,877,410 of
12,884,901,888 bytes. The snapshot is an in-memory diagnostic for one rejected
token.

### Ornith 122B MTP verification probe

Run the same 199-token fixture with `--budget-mib 12288 --mtp-probe
--draft-admission-probe "$DRAFT_GGUF"` in `llama-expert-infer`. The opt-in
probe saves the target after ordinary prefill, then replays the prompt with
all NextN rows exposed. Pairing each prompt token with the preceding target
hidden row fills draft KV. The first token uses a zero row. The 3072-float
final target row and seed token 23 produce the actual draft proposal 248046.

Restoring the draft snapshot reproduces its greedy proposal and full logit
row exactly. The target verifies the seed and proposal in one two-row batch
and selects 248046 as well. A second batch with an intentionally wrong
candidate (token 0) leaves the verifier's prefix choice unchanged and
exercises rejection. Restoring the target returns its serialized state
digest and sequence position to the saved values. The draft snapshot was
412,389 bytes and the target snapshot was 161,199,973 bytes. Their charges
and the transferred row workspace share the drafter ledger category.

Capturing all hidden rows directly in the primary prefill changed the first
logits by up to 1.14441e-5 in the initial probe. The final probe isolates the
hidden capture in the replay. Two-row verifier logits also differ bitwise
from the serial row despite choosing the same token. Both tentative batches
are restored. Committed logits, IDs, EOG and final state digest match the
original streaming run bitwise after continuation through serial decode.
The shared ledger peaked at 12,884,888,295 of 12,884,901,888 bytes and the
drafter charge returned to zero after teardown.

This checks one real greedy proposal, draft rollback and batched target
evaluation. The diagnostic replay does
extra prefill and is unsuitable as a throughput measurement. P09 depends
on a qualified P08 path. See the [P10B memory notes](atomic-engine-memory.md#resident-cuda-model-and-context-diagnostic)
for the resident CUDA qualification and remaining expert streaming work.

### Ornith 122B non-EOG batch commit check

The diagnostic `--mtp-batch-commit-probe` uses the same 12 GiB cap and draft
GGUF. It compares two serial target decodes with one two-token target batch
from the same snapshot. Both logit rows and the complete serialized states
must match byte for byte before the batch can be retained. The serial state
comparison reuses one admitted RAM buffer and an anonymous temporary file.
On a mismatch, the probe restores the target before normal serial decode.

The 27-token prompt asks the model to repeat the sentence “The quick brown
fox jumps over the lazy dog.” The output starts with the model's reasoning
prefix. The actual draft proposed token 198 after seed 248068, and both
target evaluations selected 198, followed by 760. The two batch logit rows
differed in 240,148 and 248,320 of 248,320 values respectively, with maximum
absolute differences of 0.000461578 and 0.00396156. The serialized states had
the same size (157,017,973 bytes) and different contents. The commit gate
rejected the batch.

After restore and serial continuation, IDs `[248068, 198, 760, 1156]`,
all four saved FP32 logit rows and final state digest `e878e8134b0ae901`
matched an independent serial request with the same prompt and cap.
The combined run peaked at 12,884,882,695 managed bytes.

### Localizing the non-EOG batch difference

The opt-in `--mtp-trace-layers` callback reads the first token at selected
boundaries. On this fixture, layer 0 attention norm agrees bitwise between
the serial and two-row paths. The layer 0 QKV and beta projections already
differ before convolution, recurrent attention or routed expert execution.
The original projections are Q8_0. Metal selects a one-row matvec for serial
decode and can select the few-row MMA kernel from two input rows. The trace
log records both Q8_0 kernel variants. This locates a divergence before the
recurrent update, without assigning every downstream difference to one kernel.
The callback inserts synchronization boundaries, so the commit control runs
without it. In the uninstrumented two-row run, the first differing serialized
state byte is at offset 28,795 and the logit differences match the earlier probe.

With `--ubatch 1`, the same two-token verifier call executes two one-token
microbatches. It retains the actual non-EOG proposal 198 from the batch.
Both logit rows and the entire 157,017,973-byte state match the serial control
from the same snapshot. The final four logit rows and token IDs match a
separate sequential run at ubatch 1, with final state digest
`1d8895db872e44ee`. Compare only within that microbatch setting: the prefill
graph differs from the ubatch-32 fixture. Combined peak is 12,884,892,498 of
12,884,901,888 managed bytes.

The one-token control admits both state comparison buffers in RAM. The
two-row control uses one admitted comparison buffer and a temporary file.
The layer trace uses one comparison buffer and digests, and never commits.
An initial link failed for lack of disk space. A first single-row run also
failed to write its temporary state file, and an initial trace was denied
when it tried to admit two comparison buffers. The final targeted controls
completed after reducing or relocating those diagnostic allocations.
This establishes an exact logical batch for the tested fixture when internal
execution is serial. A bitwise-exact jointly executed multi-row verifier
remains open.

### Q8_0 row matvec in the joint verifier

A temporary Metal probe ran each row of eligible Q8_0 projections through
the existing one-row matvec kernel, retaining a joint two-token microbatch
for the rest of the graph. The override was active only during verification
and counted 373 changed projections on the 27-token Ornith 122B fixture.
The MTP proposal and verifier still selected non-EOG token 198.

In the uninstrumented verification run, maximum absolute logit differences
from serial dropped from 0.000461578 and 0.00396156 to 0.000404358 and
0.00256062. There were still 241,658 and 248,319 differing values out of
248,320 per row. The first differing state byte remained at offset 28,795,
so the commit gate rejected the batch. Restoring the snapshot and decoding
serially preserved all four output rows, token IDs and final state digest
against the independent serial control. The peak was 12,884,882,695 of
12,884,901,888 managed bytes.

With the row override and a separate diagnostic trace, the first token's
layer-0 QKV, beta, convolution output and linear-attention output matched
the serial checkpoints. The first remaining mismatch among the sampled
checkpoints was the layer-0 routed MoE reduction output (`ffn_moe_out-0`).
The sampled linear-attention output covers the first token only. These
observations narrow the remaining divergence without establishing that the
recurrent state update or every expert projection matches.

The experimental backend and runner edits were restored after the runs.
For reproduction in a separate checkout, apply
`docs/atomic-engine-p08-q8-row-probe.patch` to the recorded base and add
`--mtp-q8-rows` to the existing batch commit probe. Add
`--mtp-trace-layers` only for the activation trace. The patch is a diagnostic,
and the joint non-EOG commit remains unqualified.

### Layer-0 routed MoE divergence and router row probe

With the Q8_0 row override, the detailed first-token trace agrees at
`attn_residual-0` and `attn_post_norm-0`. The first differing routed
MoE checkpoint is `ffn_moe_logits-0`, a 3072-to-256 F32 router matrix
multiplication. Expert IDs and the raw gate, up, SwiGLU and down outputs
agree. The router softmax, selected weights and weighted expert output
differ before reduction.

A second temporary override uses the same one-row Metal matvec on each
of the two input rows of the F32 router. The uninstrumented verifier
reports 48 affected routers plus the 373 Q8_0 projections. Token 198 and
the next greedy choice 760 match serial. The maximum logit differences
fall from 0.000404358 and 0.00256062 with the Q8_0-only override to
0.000284195 and 0.00200474. Both logit rows and the serialized state still
differ, with the first state difference at byte offset 28,795. The
verifier rejects direct commit. After snapshot restoration, the four
output rows, IDs and final state digest match the independent serial run.
Peak managed use is 12,884,882,695 of 12,884,901,888 bytes.

A separate trace with both overrides matches the first token through
every sampled routed MoE boundary in layer 0, including `ffn_moe_out-0`.
The next sampled mismatch is `l_out-0`, after the shared branch and
residual. The trace does not identify a unique operation in that interval
or establish equality for the second token or recurrent state.

The temporary experiment was reverted and the runner rebuilt. To
reproduce it in a separate checkout, apply
`docs/atomic-engine-p08-router-row-probe.patch` to the recorded base.
This patch includes the Q8_0 override and extended trace. Run the existing
batch commit probe with `--mtp-q8-rows --mtp-router-rows`, adding
`--mtp-trace-layers` only for the trace.

### Shared-expert gate and exact joint commit

With the Q8_0 and F32 router overrides, the first differing shared-branch
checkpoint is `shared_expert_gate-0`, the 3072-to-1 F32 projection.
The shared expert's up, gate, SwiGLU and down outputs already match for
the first token. The mismatched scalar gate carries through sigmoid,
multiplication and the final FFN/residual sums.

Adding the one-row Metal matvec to the shared scalar gate produced an
exact joint commit with `ubatch=32`. The uninstrumented verifier changed
373 Q8_0 projections, 48 F32 routers and 48 F32 shared gates. It committed
non-EOG proposal 198 from the joint microbatch without restoring and
redecoding the pair. Both logit rows and all 157,017,973 serialized state
bytes match the serial control. Final IDs `[248068, 198, 760, 1156]`, four
FP32 output rows and state digest `e878e8134b0ae901` match the independent
serial run. Peak managed use was 12,884,882,695 of 12,884,901,888 bytes.

The patched activation trace matches every sampled first-token checkpoint
through `l_out-1`. The trace run also matches the first logit row, while its
second differs by up to 0.00137472. This shows sensitivity to instrumentation
that splits encoder fusion boundaries. The instrumented path never commits.

This is one exact two-token verifier result for the saved fixture. It
does not qualify multi-round acceptance or speed. The probe includes a
serial control and hidden-state replay. Keep it in a separate diagnostic
checkout using `docs/atomic-engine-p08-shared-gate-row-probe.patch`, with
`--mtp-q8-rows --mtp-router-rows` and the existing commit-probe options.
The working backend and runner were restored and rebuilt after the probe.

### Consecutive exact target commits and retained draft prefix

With the same 27-token prompt and an eight-token output limit, the diagnostic
accepted three separate two-token verifier batches at output offsets 0, 2 and
4. The MTP proposals were 198, 1156 and 728. Each batch started from the
previous exact joint commit. Both logit rows and full serialized target state
matched the two serial decodes from the same snapshot at every round.

Two diagnostic variants used either draft KV rebuilt from the committed
prefix or draft KV retained across rounds. The retained variant decoded 27
prompt rows initially, then only two new draft rows at each later round.
The rebuilt and retained draft states differ after the first round. Both
variants proposed the same tokens on this fixture and passed the target's
bytewise commit gate. All eight output logit rows, committed IDs and final
target state digest `467c791e099663f2` match the independent serial run.

Both variants peaked at 12,884,900,231 managed bytes under the
12,884,901,888-byte cap. Draft allocations returned to zero on teardown.
The comparison buffers nearly fill the diagnostic budget. The retained
draft variant still replays the target prefix to capture hidden rows and
runs the serial control at each verification. Request times were 11.41 s
serial, 43.39 s with draft rebuild and 42.77 s with draft retention.
They measure the probes including replay and controls. A production
implementation needs a way to preserve hidden rows without replay and
establish when it can safely omit the serial comparison. Then compare
request time at the same cap, including rejected proposals.

For reproduction in a diagnostic checkout, apply
`docs/atomic-engine-p08-multi-round-probe.patch` to the recorded base.
Use the same prompt as the non-EOG commit check, eight output tokens,
and `--mtp-multi-round-probe --mtp-q8-rows --mtp-router-rows` with the
existing draft and budget options. Add `--mtp-retain-draft-prefix` for
the retained draft KV variant. Active Metal and runner sources were
restored after the three targeted runs.

### Carry committed NextN rows between rounds

After the first target replay, the diagnostic keeps the last three committed
NextN rows in a 36,864-byte ring charged to the shared ledger. A draft token
uses the preceding target row. Catching up on the two tokens of the previous
commit and proposing from the new seed therefore needs at most three rows.
Masked NextN extraction uses the decode's existing output positions.
Accepted pairs publish their rows after the bytewise comparison, and a
serial fallback takes its row from the committed serial decode.

On the same eight-token fixture, the original prefill was replayed only in
the first round. All three pairs committed. Each pair's two NextN rows,
two logit rows and serialized target state matched its serial control
byte for byte. All eight output logit rows and token IDs matched the
independent serial request, as did its final state digest
`467c791e099663f2`. This fixture had no rejected proposal.

The checked run took 23.97 s versus 42.77 s in the earlier retained-draft
probe, with logical expert reads down from 112,055,943,168 to
66,468,839,424 bytes. These are individual diagnostic runs. The sequence
still includes a bootstrap replay, serial verification for every pair and
draft rollback checks. The independent serial request took 11.41 s.

Managed memory peaked at 12,884,899,809 of 12,884,901,888 bytes and the draft
charge returned to zero. macOS reported a physical footprint peak of
13,365,127,368 bytes, above the managed cap. The cap applies to the ledger.
The individual run qualifies hidden carry on this fixture, while runtime
acceleration and exactness across other inputs still require evidence.

Apply `docs/atomic-engine-p08-hidden-carry-probe.patch` to the recorded base
in a diagnostic checkout and add `--mtp-carry-hidden` to the retained-draft
command above. The patch also includes the two Metal row overrides used by
the earlier probes.

### Reject a verifier batch and resume hidden carry

The eight-token fixture was run with an injected wrong non-EOG suffix at
offset 2, after the first exact commit. The draft generated 1156, and the
negative control substituted 0 in the actual two-row target evaluation.
The verifier's first logit and NextN rows matched the serial seed decode
byte for byte and selected 1156. The suffix rows and target state differed,
as expected for the wrong token. The target snapshot was restored and its
digest matched the preceding exact commit. Both contexts returned to
position 28 before the serial fallback advanced the target.

At offset 3, the draft caught up with one token and used the NextN row from
the fallback decode. Its seed hidden digest matched the serial control.
The pairs at offsets 3 and 5 both committed with bytewise logit, NextN and
target state checks. Across the request, all eight output logit rows and
token IDs matched the independent serial run, as did the final state digest
`467c791e099663f2`.

A diagnostic variant moved the comparison outside the measured request.
It skips the duplicate serial target decodes, the serial comparison file
and the repeat draft proposal. The verifier chooses acceptance from its
first logit row, and the rejection restores the target before a serial
decode. The run retained the one-time target bootstrap replay and snapshot
rollback. It followed the same acceptance sequence, produced the same
seed hidden digests and matched the independent serial output byte for
byte, with an identical final state digest.

For this fixture, the rejection run took 25.73 s with inline comparisons
and 21.87 s with offline comparison, versus 11.41 s serial. Logical expert
reads were 74.10 GB, 58.39 GB and 29.98 GB, respectively. These are single
diagnostic measurements. Even after removing the duplicate controls, the
request is slower than serial. The target bootstrap replay still reads
the prompt's experts again. Peak managed use in the offline run was
12,884,888,999 of 12,884,901,888 bytes, and the draft charge returned to
zero. Its macOS physical footprint peaked at 13,361,391,840 bytes.

Apply `docs/atomic-engine-p08-rejection-probe.patch` after the hidden-carry
patch and add `--mtp-reject-step 2` to the same fixture command. For the
offline run, apply `docs/atomic-engine-p08-offline-control-probe.patch` on top
and add `--mtp-offline-control --state-digest --logits "$LOGITS"`. The two
variants cover one injected rejection after a successful pair with greedy
sampling. The offline diagnostic's acceptance rule is only exercised on
this fixture. Both variants were removed from active sources after the
targeted runs.

### Capture draft bootstrap rows in the target prefill

On the 27-token rejection fixture, extracting dense NextN rows during
the initial target prefill eliminated the replay. The existing unmasked
mode moves the last-layer crop from before the FFN to after the output
norm. That changes the shape of the output token's computation. The direct
variant selected the same IDs and produced the same final state digest,
but 248,317 values differed in the first logit row, with a maximum absolute
difference of 0.0019989013671875. The other seven rows matched bytewise.

A separate last-layer branch retained the ordinary early crop and logit
path for the output token. The dense NextN branch shares the last-layer
attention and its incoming residual with that path, then runs the final
FFN and output norm for all rows. This branch is used only for unmasked
capture in a managed Qwen3.5 MoE graph with multiple tokens and one output.
It adds one last-layer FFN for that microbatch instead of replaying the
whole prompt. Both requests in this probe have one output token in the
initial batch.

The split variant matched all eight logit rows, committed IDs and the
final state digest of the independent serial request. All four MTP
rounds used the same seed hidden digests and followed the same
accept/reject sequence as the replay control. It took 12.81 s with
37.06 GB of logical expert reads, versus 21.87 s and 58.39 GB with
bootstrap replay under the same offline control. The sequential request
took 11.41 s with 29.98 GB of logical expert reads. These are individual
diagnostic measurements. Peak managed use was 12,884,875,938 bytes of the
12,884,901,888-byte cap. The macOS physical footprint peaked at
13,360,163,064 bytes.

The 199-token, multi-microbatch fixture also matched both serial logit
rows, token 23, EOG 248046 and final state digest `012b8232a95ab2b8`.
The draft proposed EOG after bootstrap and the runner finished via the
serial decode. The run took 45.59 s and read 146.08 GB of logical expert
data, with a peak of 12,884,875,938 managed bytes. The older replay
diagnostic took 94.15 s and included additional verifier controls.
Its timing is not a like-for-like replay comparison. The original
serial run took 40.16 s at a different cap (10 GiB).

For reproduction, apply `docs/atomic-engine-p08-one-prefill-probe.patch`
on top of the hidden-carry, rejection and offline-control probes. Add
`--mtp-one-prefill` and set `LLAMA_ATOMIC_MTP_SPLIT_PREFILL=1` in that
process. Leaving the environment variable unset selects the direct
capture control. The long fixture uses the saved completion prompt and
omits `--mtp-reject-step`. The model-graph hook, runner and Metal overrides
were restored after measurement. The split path remains a diagnostic.
The short fixture still favors serial generation at the same managed cap.

### Retain the draft seed and checkpoint only partial target state

The draft's proposal decode consumes the target's already committed seed.
It does not consume its own proposed suffix, so the draft can retain the
seed KV on both acceptance and rejection. The next round appends only
any accepted suffix that is missing from the draft. After rejection and
serial fallback, the draft is already aligned through the seed and the
next proposal starts without a catch-up decode.

The target takes a sequence checkpoint using the existing partial-state
API before verification. On rejection it restores recurrent state first,
then truncates the suffix from ordinary attention KV. This follows the
existing checkpoint order for hybrid memory. The checkpoint was
156,304,252 bytes at each round versus 156,968,773 to 157,091,773 bytes
for full state. Most of the payload is recurrent state. The variant also
eliminates per-round full state hashing and draft serialization.

In the eight-token rejection fixture, the target returned to position 28
while the draft retained the committed seed at position 29. At the next
round, the draft consumed zero catch-up rows and produced the same proposal
and seed hidden digest as the full-checkpoint control. The acceptance
sequence remained commit, reject, commit, commit. A second run omitted the
injection and accepted all three proposals. Both runs matched all eight
logit rows, committed IDs and final state digest `467c791e099663f2` from
the serial request. Peak managed use was 12,884,875,938 bytes of
12,884,901,888, and the draft charge returned to zero. macOS physical
footprint peaked slightly above the managed cap in both runs.

Individual request times were 12.27 s with rejection versus 12.81 s for
the full-checkpoint rejection control, and 11.69 s with all proposals
accepted. The serial request at the same cap took 11.41 s. Even with all
proposals accepted, the diagnostic read 33.36 GB of logical expert data
versus 29.98 GB serial. These measurements do not establish acceleration.

Apply `docs/atomic-engine-p08-compact-checkpoint-probe.patch` on top of
the one-prefill probe and add `--mtp-compact-checkpoint`. Keep
`--mtp-reject-step 2` for the rejection run and omit it for the acceptance
run. Active sources were restored and the runner rebuilt after the probes.

### Attribute expert reads and share the draft vocabulary weights

The accepted-pair fixture with a separate draft read 33,363,197,952
logical expert bytes versus 29,982,031,872 in the 12 GiB serial run.
The extra 3,381,166,080 bytes comprise 443,154,432 during prefill and
2,938,011,648 after it. Expert cache occupancy at the end of prefill
was 454,754,304 bytes with the draft versus 3,861,446,656 serial.
The MTP decode used 1,851 expert submissions versus 2,016 serial,
while its target expert reads increased from 8.65 GB to 11.59 GB.

Both the token embedding and LM head in the extracted draft were
810,516,480-byte Q8_0 copies of the target's tensors. Their shapes,
quantization and raw payload hashes matched. The opt-in probe skips
loading those two copies and points the draft at the already charged
target tensors before creating the draft context. The target owns the
tensors and outlives the draft context and model. The probe assumes
the draft was extracted from the same target and requires external
payload verification before use.

Draft weight charge fell by 1,621,032,960 bytes and expert cache occupancy
after prefill rose to 2,075,262,976 bytes within the same 12 GiB ledger.
All three pairs committed with the same proposals as before. The eight
output logit rows, IDs and final state digest matched serial. Peak
managed use was 12,884,893,442 bytes, macOS physical footprint peaked at
12,654,765,680 bytes and the draft charge returned to zero on teardown.
The request read 33.34 GB, down by 27.72 MB, and finished in 11.65 s
versus 11.69 s with the duplicated tensors. These individual runs show
the memory saving. They do not establish a latency gain.

A second serial control at 10,589 MiB held 2,080,571,392 expert bytes
after prefill, within about 5.3 MB of the shared-weight run. It matched
the original serial output and read 30.92 GB, of which 9.58 GB came
after prefill. Its 2,439 post-prefill cache hits compare with 14 in the
shared-weight MTP decode, which read 11.56 GB after prefill. Matching
prefill occupancy alone therefore leaves 2.42 GB of extra reads,
including 1.98 GB in decode. This control does not separate verification
access order from transient checkpoint reservation. A per-round cache
trace can isolate which misses remain after accounting for both.

Apply `docs/atomic-engine-p08-shared-vocab-probe.patch` on top of the
compact checkpoint probe and add `LLAMA_ATOMIC_MTP_SHARE_VOCAB=1` for
the accepted-pair command. Keep the same verified draft file. The probe
is isolated from the active loader and runner. The cache-matched serial
control used the restored runner, the same prompt with an eight-token
limit and `--budget-mib 10589` in place of `--budget-mib 12288`.

### Measure expert eviction at verifier checkpoint boundaries

On the same shared-weight fixture, the runner sampled cache counters
before and after each checkpoint reservation and after each verifier.
The first temporary checkpoint evicted 89 expert slices. The next two
reservations fit without any eviction or budget resize. All three
verifier rounds recorded zero hits, with 1,851, 1,923 and 1,875 misses.
They read 3.15, 3.28 and 3.19 GB of expert slices, respectively. The
14 decode hits came after the last verifier. One reservation-time
eviction event therefore does not account for continued verifier misses.

An alternative charged and retained the same 156,304,252-byte host
checkpoint after prefill and reused it for each verifier. It moved the
evictions to request setup. The verifier hit counts, miss counts and
logical bytes stayed identical. Keeping the reservation through the
final serial decode lost 14 hits and added 24,182,784 bytes of reads.
Individual request times were 11.68 s for temporary reservations and
11.57 s for retention, versus 11.41 s for the 12 GiB serial baseline.
These runs show no read reduction from retaining the checkpoint and do
not establish a latency gain.

Both runs committed the same three pairs, matched the eight serial logit
rows byte for byte and matched final state digest `467c791e099663f2`.
The managed peak was 12,884,893,442 bytes in both runs and the draft
charge returned to zero. Physical footprint peaks were 12,651,865,688
bytes with temporary reservations and 12,499,560,048 with retention.
The retained-workspace variant was exercised on the all-accepted fixture.

Apply `docs/atomic-engine-p08-checkpoint-cache-probe.patch` on top of the
shared vocabulary probe. The existing accepted-pair command emits the
temporary-reservation counters. Add `--mtp-retain-checkpoint` for the
retention control. Sources were restored after the probes. A trace
comparison of serial and verifier access order can now focus on expert
reuse distance and working-set size.

### Replay batched decode cache access and qualify ClassLRU

The same 27-token prompt was traced with shared-weight MTP at 12 GiB
and the 10,589 MiB serial cache control. The LRU replay matched all
recorded hits and allocation sizes for both traces. Request phases were
split at the recorded prefill projection count for reuse-distance
analysis. The replay tool's default one-token classification includes
two-token verifiers in its prefill bucket, so policy savings below are
whole-request totals unless a measured runtime phase is specified.

Byte reuse distance sums the minimum allocation sizes of distinct
expert slices accessed since the previous access to the same slice.
Across post-prefill reuses, its median was 7.71 GB for MTP and 4.35 GB
for serial. MTP had 5,121 of 5,148 reuses beyond the observed capacity
after accounting for the slice itself. Serial had 3,917 of 6,357.
The median cache-plus-staging capacity was 1.92 GB for MTP and 2.08 GB
for serial. The distance is a lower bound on the allocation pressure
when staging, larger reused buffers or pinned leases matter.

The existing ClassLRU replay predicted a 725,483,520-byte saving on
MTP and a 1,298,399,232-byte increase on the cache-matched serial trace.
An untraced Metal run with ClassLRU matched the MTP replay's 464 hits,
19,138 misses and 32,609,992,704 logical read bytes. All measured read
savings occurred after prefill. The third verifier gained 225 hits.
All three proposals were accepted. The eight logit rows and final
state digest matched the independent serial run. Managed use peaked
at 12,884,876,846 bytes under the 12 GiB cap. Request time was 11.55 s
versus 11.68 s in the untraced MTP LRU control and 11.41 s serial.
These are individual runs. The read savings alone do not qualify
end-to-end acceleration or a policy default change.

Extending the same prompt to 16 output tokens gave six accepted pairs
and two natural rejections, at offsets 6 and 7. The next pair committed
at offset 8. All 16 logit rows matched the 12 GiB serial control byte
for byte, along with IDs and final state digest `974b57d3751adee8`.
The MTP run spent 15.25 s and read 27.05 GB after prefill, versus
14.13 s and 18.77 GB after prefill for serial. Peak managed use was
12,884,893,598 bytes. This pair of runs supplies no acceleration
evidence for the longer decode. Keep the MTP path as a diagnostic.

The two rejected verifier batches accounted for 5,602,541,568 logical
bytes before the serial fallbacks. Subtracting those bytes as a
bookkeeping sensitivity check leaves 21,442,461,696 post-prefill bytes,
still 2,670,624,768 above serial. This is not a predictor simulation:
skipping a batch would change cache history and subsequent reads.
It shows that the measured read gap also involves accepted work.
Keep serial as the default for this profile. Tree verification remains
deferred while chain speculation lacks an end-to-end win here.

Apply `docs/atomic-engine-p08-trace-probe.patch` on top of the checkpoint
cache probe and add `--expert-trace "$TRACE"` to the shared-weight MTP
command. The serial trace uses `--budget-mib 10589`. Replay both with
`tools/expert-source/replay-cache.py`. For the live MTP candidate, use
`--cache-policy class-lru` with tracing disabled. The longer comparison
sets `--tokens 16` on the candidate and on the 12 GiB serial control.
The diagnostic changes were removed and the runner rebuilt.

### Stop speculation when the round cost exceeds serial decode

The 16-token diagnostic calibrated serial decode at 372.489 ms per
advanced token from the saved 12 GiB control. It stopped using MTP after
two consecutive rounds whose elapsed time per committed token exceeded
that calibration. The cycle includes draft, checkpoint, target
verification and any synchronous serial fallback. The first two rounds
both accepted, costing 884 ms and 750 ms for two tokens each. The runner
released the draft context and weights after the second pair. The 36 KiB
hidden-row buffer stayed charged until the request ended.

The adaptive request took 14.51 s versus 15.25 s with fixed MTP and
14.13 s serial. Post-prefill reads fell from 27.05 GB to 19.89 GB,
versus 18.77 GB serial. All 16 output logit rows matched serial byte
for byte, with the same tokens and final state digest
`974b57d3751adee8`. The managed peak was 12,884,899,362 bytes.
The threshold was fitted to this fixture's serial control. This
diagnostic has no periodic retry and provides no cross-request
qualification. Serial remains the default for this profile, and
the tree experiment remains deferred.

Apply `docs/atomic-engine-p08-adaptive-cost-probe.patch` on top of the
trace probe and add `--mtp-serial-ms 372.489` to the untraced
16-token ClassLRU command. Active sources were restored and the
runner rebuilt after the comparison.
