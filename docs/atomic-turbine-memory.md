# Engine memory budget and live control

## Contract and ownership

`llama_model_params.memory_budget_size` enables disk expert streaming with a
single model-wide UMA ledger. It is mutually exclusive with the older
`expert_streaming_size`, which still means expert/workspace bytes only.
Rebuild clients and backend plugins after the model-parameter and internal buffer
layout additions; mixing old backend binaries is unsupported.
The initial qualification is Apple M4 Max, original Ornith 35B Qwen3.5 MoE,
one context/sequence, context 512 and microbatches 32/128. It is not a portable
OS RSS limiter or an automatically qualified support matrix.

P10A adds [LFM2 MoE on the same Metal backend](atomic-turbine-second-architecture.md):
context 512, microbatch 32, LRU, a 4 GiB initial cap and live 3456/4608 MiB
targets. Its short-convolution and attention state remain in the shared ledger.

The admission invariant is:

```
rounded physical backend allocations
+ explicitly owned expert arenas and host read/workspace buffers
+ fixed 2 GiB runtime envelope
<= applied engine budget
```

The envelope is reserved before model metadata loading. It remains a reservation,
not a fabricated allocated-byte counter. It covers tokenizer/source/graph metadata,
C++ control objects, host-only scratch, loader temporaries, thread stacks and
backend/runtime allocations not exposed as GGML buffers. The first measured
5 GiB run's quiescent footprint residual above explicit payload was about 0.82 GiB;
the envelope leaves more than 1 GiB of margin for the tested profile. Process peak
footprint is measured separately; it is never added to Metal allocations. Kernel
file cache, other processes, and caller-owned buffers are outside this engine
contract. Buffered reads are not a controlled cold-SSD benchmark.

Major physical buffers receive a ticket in `ggml_backend_buft_alloc_buffer`
before the allocator runs. The ticket travels with the buffer; the free callback
returns it only after physical backend destruction, including when freed on a
different thread. Failed allocations cancel reservations. Parent multi-buffers
and views do not acquire a second ticket for their children. Qualified CPU/Metal
buffer reservations round up to 64 KiB to cover backend page rounding. Backend
layouts requiring mapped or repacked ownership are not silently admitted.

Allocation scopes label model buffers as weights, context/KV/recurrent/output
buffers as state, and scheduler buffers as scratch. Reallocation counts old and
new buffers while both live. Allocation pressure may evict completed expert
entries before retrying admission; it does not alter weights or truncate state.
Expert buffers already have explicit tickets, so their nested allocation scopes
suppress interception and avoid double charging. Their ledger is now the same
ledger used by the model and context.

The context is constructed only if the total working floor fits. Floor is the
currently fixed engine allocations and runtime envelope, plus the largest expert,
its metadata and simultaneous read staging. It does not count reusable cache
entries as fixed memory. CPU/shared Metal slots now accept direct expert reads;
the same staging headroom remains reserved for conservative admission but is
not committed as a physical allocation. Other buffers retain staged upload.
Only one active context may claim a budgeted model;
failed construction and destruction return that claim. Decoder graph growth,
output buffers and state/device helper allocations use the same admission hook.

The exposed experimental profile requires an explicit context no larger than
4096, batch no larger than 4096, microbatch no larger than 256, at most 16 CPU
threads, one sequence, and no rollback/embedding mode. These are implementation
bounds, not evidence that every point in that range has been qualified. MTP,
LoRA/control-vector adapters, training, no-alloc fitting, mapped weights, custom
buffer overrides and multi-device execution are outside this profile. A second
architecture beyond the two measured profiles, device and larger context require
separate qualification.

## P03/P04 L2 implementation decisions

The concrete integration points are the GGML buffer allocator/free path,
`llama-memory-contract`, model loading, context construction, scheduler/output
allocation, the existing expert executor/store, and native diagnostic runner.
The original file parser, router, attention/recurrent state and numerical kernels
are reused. The legacy allocator remains unchanged when the observer is absent.

The integrated sequence is admission at physical allocation, shared model/context
ownership, floor reporting, synchronized control boundaries, and measured release
with a native logits/state comparison. An allocator hook was chosen over charging
buffer totals after loading: post-hoc totals cannot prevent a transient over-cap
allocation. A fixed measured envelope was chosen for hidden runtime objects rather
than claiming exact process-wide allocation interception.

Invariants: no resident fallback, no double-counting aliases, reservation before
payload allocation, resource release before ticket release, no evicting active
leases, no change to the active context or original quantization on resize, and
no publication of a partial read. The synchronous executor retains each expert
until all its row waves complete; it has no asynchronous read publication queue.
Pending reads finish before the graph boundary applies a new target.

The optional ClassLRU history is also admitted as runtime memory and raises the
reported minimum; see [cache policy evidence](atomic-turbine-cache.md).

The bounded vector micrograph arena adds 27,280 admitted bytes on Metal. Reused
expert allocations retain their original tickets and are still individually
freed on shrink; [current prefill/live evidence](atomic-turbine-prefill-performance.md)
records the updated floors and real release.

The [LFM allocation optimization](atomic-turbine-lfm-performance.md) can reuse a
larger completed victim for a smaller expert. Excess capacity stays physically
allocated and fully charged until eviction/shrink; trace payload and required
allocation are distinct from the actual retained allocation. No idle pool sits
outside admission or release.

The [grouped Metal executor](atomic-turbine-lfm-gpu.md) adds up to 3,231,904
admitted workspace bytes on the measured LFM profile. Its live minimum therefore
increases by that amount; it does not raise the user's cap. Workspace sizing can
fall back at construction, and the live group size drops to one when retaining
multiple maximum-size experts plus read staging would not fit. No budget change
is applied while an in-flight group still owns expert leases.

The [bounded LFM read overlap](atomic-turbine-lfm-overlap.md) reserves space for
up to four additional maximum-size expert allocations alongside the current GPU
group and one full-payload I/O reservation. These are normal cache entries and
leases, not a separate unaccounted queue. Direct CPU/shared Metal reads finish
before publication without waiting on unrelated pinned GPU work. The caller
then waits before resetting the graph or releasing its leases, including on a
read exception. No new payload pool, workspace or thread is allocated; the floor
is unchanged. Low headroom disables optional overlap at each projection. All
reads and GPU work still complete before applying a live resize.
The same unchanged admission rule is now enabled by default for `qwen35moe`
on Metal; [Qwen controls](atomic-turbine-qwen-overlap.md) cover the 5 GiB
microbatch-128 profile and microbatch-32 live shrink/grow with original IQ4_XS
weights. This qualification does not enable other Qwen architecture IDs.

## Live API

Opt-in [automatic pressure control](atomic-turbine-pressure.md) reuses this
same admission and release path. Explicit manual requests suspend the policy;
they never compete with automatic recovery.

The experimental C++ extension declarations are in `src/llama-atomic.h`:

- `llama_memory_budget_request(ctx, target)` queues a target and returns a
  monotonically increasing receipt ID. It is safe to call during evaluation.
  The latest pending request wins; earlier pending IDs are superseded.
- `llama_memory_budget_apply(ctx)` applies a queued target while idle. Do not
  race this call with evaluation. The engine also applies pending targets after
  each synchronized inference graph, including between prefill microbatches.
- `llama_memory_budget_get(ctx, &snapshot)` reports requested/applied/minimum,
  current use, lifetime high-water, cumulative completed releases, category
  totals, runtime envelope, latest receipt/completion IDs and status. Query idle.

States are idle, pending, applied and rejected. The requested field retains the
last attempted target. A below-floor target is rejected before changing the
ledger cap or cache; the old applied limit and session remain valid. The caller
must distinguish the attempted requested value from the still-active applied cap.
Grow changes admission capacity immediately and allocates nothing by itself;
subsequent misses warm the cache. Shrink frees unleased buffers at a completed
graph boundary. This implementation has no partial-apply state because one
serialized context completes its synchronous I/O/GPU work before applying.

The request call does not promise a deadline: pending work lasts until the next
graph boundary. A caller can impose a waiting deadline without reclaiming live
buffers. An idle session needs the explicit apply call. Concurrent evaluation on
one context and racing apply/query/destruction remain unsupported. Multiple
rapid requests are coalesced, not treated as independent completed resizes.
A read/compute error poisons the context as before; discard it rather than claim
transactional state recovery. Pressure-driven policy is a later package.

## Reproduce only the affected checks

Use the Metal configuration in `atomic-turbine.md`, then build selected targets:

```sh
cmake --build build-metal --target llama-expert-infer test-memory-contract test-expert-executor -j4
build-metal/bin/test-memory-contract
build-metal/bin/test-expert-executor 'shared engine budget resize preserves computation|impossible working set rejected'
build-metal/bin/test-expert-executor --backend MTL0 'shared engine budget resize preserves computation'
```

The contract fixture checks denial before allocation, old/new overlap, cross-thread
release, aliases, allocation failure rollback, nested explicit admission, reclaim,
runtime floor and single-context ownership. The executor fixture checks two
shrink/grow cycles, rejected targets, unchanged results/state charges and separate
demand warming/cache hits. The existing delayed-completion store case covers
retirement while a backend consumer still owns a lease.

For the real checkpoint, set `MODEL` to its existing local path. The fixed-budget
control uses the same runner and raw prompt as the earlier numerical control:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 5120 \
  --prompt-file docs/atomic-turbine-prefill.txt --tokens 16 --ctx 512 --ubatch 32 \
  --state-digest --output /tmp/engine-control.json --logits /tmp/engine-control.f32

build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 5120 \
  --prompt-file docs/atomic-turbine-prefill.txt --tokens 16 --ctx 512 --ubatch 32 \
  --state-digest --prefill-budget-mib 4800 \
  --resize-at 2:6144 --resize-at 2:4704 --resize-at 4:6144 \
  --resize-at 8:4800 --resize-at 10:1 --resize-at 12:6144 \
  --resize-at 14:4704 --resize-at 15:5120 \
  --output /tmp/engine-resize.json --logits /tmp/engine-resize.f32
cmp /tmp/engine-control.f32 /tmp/engine-resize.f32
```

The prefill control callback queues its request after the first layer's down
projection; application waits for the full graph boundary. Duplicate step numbers
queue opposing requests before one apply, exercising coalescing. The 1 MiB target
is intentionally impossible. JSON events distinguish lifetime releases from the
release delta of each individual resize. State digests cover native serialized
model/context state; output token IDs come directly from the native sampler.

A second microbatch regime uses `docs/atomic-turbine-completion.txt`, which contains
explicit Qwen message delimiters and a closed thinking prefix. It asks for a
complete answer to the notes, not an arbitrary bounded reasoning fragment. Run
with `--ubatch 128 --tokens 32`, first resident and then `--budget-mib 5120`, and
compare the complete logits, committed IDs, EOG and state digest. The resident
control is numerical evidence, not an equal-cap performance competitor.

All observations and limitations are retained in `atomic-turbine-evidence.json`.
No aggregate verify, full suite or coverage matrix is required or run here.

## Recorded result

M1 is demonstrated for this first profile. At microbatch 32, the 190-token prompt
and 16-token decode matched all 3,973,120 recorded logits against the resident
reference. The total ledger peaked at 5,368,263,968 bytes under 5,368,709,120 bytes;
observed process footprint peak was 4,100,689,448 bytes. At microbatch 128, the
199-token complete-answer prompt returned `8` and EOG 248046, matching resident
logits and the native state digest. Ledger peak was 5,368,589,472 bytes and
footprint peak 4,003,450,360 bytes. The 3000 MiB negative probe was rejected while
admitting nonexpert model buffers, before inference and without a resident fallback.

Manual P04 control is demonstrated on the same active session. A request queued
inside prefill applied automatically at a graph boundary. Opposing queued targets
coalesced to the latest receipt. Two growth/shrink cycles, a near-floor target and
an impossible target left all logits and the final serialized-state digest equal
to the fixed-budget control. Final managed use returned to exactly the control's
5,367,673,776 bytes; there was no accumulated managed allocation in these cycles.

The largest recorded shrink freed 1,408,788,304 managed bytes; Metal allocated size
fell by 1,407,909,888 bytes and process footprint by 1,447,363,064 bytes in 46.94 ms.
Grow itself allocated zero bytes; later computation recorded 1,255 cache hits.
The impossible 1 MiB request released nothing and left the applied 4800 MiB cap
and session intact. The lifetime peak reached the temporarily applied 6 GiB cap;
it must not be compared against a later lower cap as though it were current use.

Request times were 20.30 seconds for the 32-token-microbatch control, 19.30 seconds
for the resize trace, and 8.84 seconds for the complete-answer microbatch-128 run.
These single runs establish behavior, not a statistically qualified speed result.
The narrow tests comprised four allocation-contract cases (16 assertions), the
shared-resize/floor CPU cases (19), shared resize on Metal (18), and the existing
delayed-completion retirement case (7). Earlier unrelated cases were not rerun.

## P10B discrete-memory prerequisite

The ledger already supports independent host and device caps. A focused
expert-store fixture uses CPU storage with direct access disabled to exercise
the staged transfer path under separate caps. It verifies host admission
failure rolls back both domains, host staging contributes to the host peak
and is freed after upload, and the resident expert stays charged to the
device domain until the final lease completes a pending device shrink.

The fixture exercises accounting and ownership on the CPU backend. The
active model allocation observer admits CPU and Metal buffer types and
charges them to the UMA ledger. The hardware probe results below cover
the underlying CUDA operations. Managed CUDA inference remains open.

### CUDA device probe

Build the standalone probe on a Linux host with the CUDA toolkit and a
discrete NVIDIA GPU. It needs no model and is outside the default build.

```sh
cmake -S . -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TOOLS=ON
cmake --build build-cuda --target llama-discrete-probe -j
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY build-cuda/bin/llama-discrete-probe CUDA0 > discrete-probe.log 2>&1
```

The probe allocates 2 MiB of pinned host storage and 1 MiB of device
storage through ggml. It checks the actual buffer types so a pageable
fallback cannot pass as pinned memory. It queues H2D and D2H on the
same backend stream and waits on an event recorded after the copies.
The independent caps remain pending while the buffers own their
charges. After event synchronization, the probe checks every copied
byte, frees each buffer and verifies the two caps reach zero.
Driver-reported free memory is sampled before allocation, during
ownership and after release. Those samples include activity outside
the probe and are informational.

Keep the registered device and its `VMM:` line from the backend startup
log with the probe result. That line reports capability for the build
and device. The copy probe uses ordinary backend buffers. A missing
device returns status 2, and a failed probe invariant returns status 1.

On the Apple host, `--cpu-control` verified byte equality and the
separate ledger peaks of 2 MiB and 1 MiB with synchronous copies.
The default invocation reported `device unavailable: CUDA0` and
returned status 2.

On Ubuntu x86-64, the native CUDA 13.5.37 build ran on a GeForce RTX
4070 Laptop GPU (compute capability 8.9) with driver 595.99.02. The
probe confirmed async copies, events and the actual pinned host buffer
type. The 1 MiB round trip matched byte for byte. Ledger peaks were
2 MiB host and 1 MiB device, and both caps reached zero after event
completion and release. Driver-reported free VRAM went from
8,031,436,800 bytes to 8,029,339,648 and returned to 8,031,436,800.
The single round-trip sample includes first-use work and is unsuitable
for throughput estimates. The native build used an isolated CMake
project containing ggml and the two probes, with CUDA FlashAttention
disabled and only architecture 89 targeted.

### CUDA virtual memory lifecycle probe

The separate Driver API probe requires Linux and CUDA Toolkit 11 or
later. It links only the CUDA driver and the fork's memory ledger.
It can be built independently of the ggml CUDA backend.

```sh
cmake -S . -B build-vmm -DLLAMA_BUILD_TOOLS=ON -DATOMIC_TURBINE_CUDA_VMM_PROBE=ON
cmake --build build-vmm --target llama-cuda-vmm-probe -j
build-vmm/bin/llama-cuda-vmm-probe 0
build-vmm/bin/llama-cuda-vmm-probe 0 --no-vmm
```

The VMM path reserves twice the aligned virtual range it backs with
physical VRAM. Each of two cycles maps the same address, fills and
reads back every byte, and requests a zero VRAM cap. The ledger keeps
the retired allocation charged through synchronization and unmap.
The probe releases the generic handle after mapping, so successful
unmap releases the remaining reference to the backing allocation.
The second cycle exercises growth and reuse of the same virtual range
before the reservation is freed. The pageable readback buffer uses a
separate 1 MiB RAM charge.

`--no-vmm` forces the ordinary `cuMemAlloc`/`cuMemFree` path, which
also runs automatically when the device reports no VMM support.
Inspect the reported `mode` to identify which path executed. An error
from a VMM operation on a device that advertises support fails the
probe. Reported peaks and releases are ledger values. A missing or
integrated device returns 2, and an API or invariant failure returns 1.

Local checks used the CUDA 12.8 headers and a CPU-side Driver API
control to exercise call ordering and ledger assertions for both
allocation paths. The default CMake configuration excludes the
probe, and enabling it for a macOS target reports the Linux requirement.

The CMake target was also cross-linked on Apple Silicon for x86-64
GNU/Linux using Zig 0.15.2 and the NVIDIA CUDA 12.8.90 driver stub.
All 21 imported CUDA symbols resolved against the stub. The resulting
ELF depends on `libcuda.so.1` and carries no stub search path or
`libcudart` dependency. This build used the glibc 2.36 target and
exercised only the standalone VMM target.

On Ubuntu x86-64 with a GeForce RTX 4070 Laptop GPU (8,188 MiB) and
driver 595.99.02, the cross-linked binary passed both allocation paths.
VMM reported a 2 MiB allocation granularity and used a 4 MiB virtual
range. The peak ledger charge was 2 MiB and the two cycles released
4 MiB in total. The forced `cuMemAlloc` path had a 1 MiB peak and
released 2 MiB in total. Both paths checked all device bytes and
released the full charge after each cycle.
The native CUDA 13.5.37 build reproduced both outcomes on the same
GPU. This completes the early P10B device probe.

### CUDA expert projection with separate RAM and VRAM caps

The focused expert-store fixture loads one original FP32 expert slice
from a small GGUF and computes a matrix-vector projection on CUDA.
The store uses its existing synchronous staged upload. The RAM ledger
covers read staging and tensor metadata. The VRAM ledger covers the
expert buffer and the explicitly allocated graph input and output
buffers. Internal CUDA workspace and driver pools are outside the
fixture's ledger.

The test requests a VRAM shrink after submitting the graph, with the
expert still leased. The shrink stays pending until lease release
synchronizes the backend and frees the expert. Graph output remains
readable and matches the four reference values exactly. Both tracked
domains return to zero after teardown.

```sh
cmake -S . -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build-cuda --target test-expert-store -j
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY build-cuda/bin/test-expert-store \
  --backend CUDA0 'discrete CUDA expert projection and shrink'
```

On the same Ubuntu RTX 4070 Laptop GPU with CUDA 13.5.37, the selected
case passed all 19 assertions. The run used an isolated CMake project
containing ggml and this test target, with architecture 89 and CUDA
FlashAttention disabled. The case skips on non-CUDA backends, as
confirmed on macOS. Full model inference still needs a device allocation
classifier and managed caps for model, state, graph and backend workspace.

### Discrete allocation observer

An explicit pair of backend buffer types can be supplied to the existing
allocation observer. The selected device buffer type is charged to VRAM.
The backend's pinned type and the standard CPU buffer type are charged to
RAM. The observer compares buffer type identities, and an unlisted type is
rejected even if its display name matches a listed type. The UMA path
retains its existing classification.

The observer covers allocations made through ggml's observed buffer
allocator. The default CUDA legacy and VMM pools allocate outside it.
The discrete classifier is available to explicitly scoped callers;
model startup and context initialization still select the UMA contract.

On the Ubuntu RTX 4070 Laptop host the focused test passed all 13
assertions. The test admitted CPU and pinned buffers under a 128 KiB
RAM cap and a device buffer under a separate 64 KiB VRAM cap. It
checked independent denials at each cap, rejected an unlisted buffer
type, and verified that a device state allocation kept a VRAM shrink
pending until release. The ordinary allocation dispatch case passed
its 15 assertions on macOS.

```sh
cmake --build build-cuda --target test-memory-contract -j
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY -u GGML_CUDA_NO_PINNED \
  build-cuda/bin/test-memory-contract --cuda-device CUDA0 'discrete CUDA buffer accounting'
```

### Bounded staging for discrete model loading

The internal GGUF loader can accept an explicit discrete staging owner.
For buffered file loads, it uploads device tensors in synchronous
chunks through one reusable 1 MiB CPU buffer charged as host I/O
before reading. The selected device buffer type must match the
destination. Host tensors still read directly into their allocations.
The ordinary loader path retains its existing upload selection,
including its pinned and pageable paths.

The bounded path rejects mapped, direct-I/O and lazy input modes, and
checks file ranges before copying. Its synchronous transfers let the
same staging buffer serve successive chunks without event ownership.
An insufficient host budget fails before the first read. Destruction
releases the staging allocation and its charge, including on unwind.

A local GGUF fixture passed through the internal loader into a Metal
buffer and matched all 2 MiB + 17 bytes with a 1 MiB host staging peak.
The focused staging case passed all 11 assertions on CPU and on the
Ubuntu RTX 4070 Laptop GPU. It checks admission failure, retry after
a read error, exact CUDA upload, and final release. The resident
diagnostic below uses this loader option.

### Opt-in CUDA workspace admission

An internal CUDA proc-address hook allows a fresh discrete NVIDIA backend
to select an allocation-observed pool. It uses ordinary device buffers
for CUDA kernel workspace, so the scoped observer reserves VRAM before
allocation and the buffer releases the ticket after `cudaFree`. This
variant frees each block on return to the pool. It disables CUDA graph
capture and graph optimization for that backend so freed workspace
cannot be referenced by a captured graph. The standard backend retains
its pool selection, caching and graph behavior.

An observed graph submission without an installed allocation scope
fails. A denied workspace allocation returns `GGML_STATUS_ALLOC_FAILED`
and synchronizes before the caller can retry. This is a bounded
correctness path. Its synchronous frees have not been benchmarked.

The pool covers explicit kernel scratch allocations routed through
`ctx.pool()`. The opt-in path also accounts for the explicit cuBLAS
handle workspace, described below. Opaque driver and library
allocations remain outside the ledger. The resident diagnostic below
uses this hook. End-to-end CUDA expert streaming remains open.

On the Ubuntu RTX 4070 Laptop GPU, the focused case passed all 10
assertions with the CUDA graph build option enabled. A zero VRAM cap
denied the quantized projection's workspace. Growing the cap allowed
the same graph to retry, match the reference output and return its
workspace charge. The ordinary allocation-dispatch case on macOS
also passed its 15 assertions.

```sh
cmake --build build-cuda --target test-memory-contract -j
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY build-cuda/bin/test-memory-contract \
  --cuda-device CUDA0 'CUDA workspace pool admission and retry'
```

### Explicit cuBLAS workspace admission

The observed CUDA backend reserves the explicit cuBLAS workspace
through ggml before creating the first cuBLAS handle. The backend keeps
the buffer and its VRAM scratch charge until teardown. When admission
fails, no handle is installed. The next submission can retry using
the same backend after the caller raises the VRAM cap. The ordinary
backend continues to allocate its per-stream workspaces directly.
The observed path uses its existing single-stream restriction.

On the RTX 4070 Laptop GPU the focused case passed all 12 assertions.
It denied the 4 MiB workspace with a zero VRAM cap, then ran the same
cuBLAS graph after raising the cap and matched every reference value.
A second submission reused the original workspace at the exact cap.
The zero-cap shrink remained pending until backend teardown freed the
workspace. This covers the explicit handle workspace. Driver and
library-internal allocations still require separate qualification.

```sh
cmake --build build-cuda --target test-memory-contract -j
env -u GGML_CUDA_ENABLE_UNIFIED_MEMORY build-cuda/bin/test-memory-contract \
  --cuda-device CUDA0 'cuBLAS workspace admission and lifetime'
```

### Resident CUDA model and context diagnostic

`llama-expert-infer` accepts a CUDA device name with a pair of RAM and
VRAM caps in MiB. The diagnostic uses buffered GGUF loading and attaches
the two-domain ledger before allocating weights. CPU and CUDA host
buffers charge RAM. The selected device's buffers charge VRAM. The
context uses the same ledger for KV and recurrent state, graph buffers,
explicit CUDA kernel scratch and the explicit cuBLAS workspace.
Optional state digest capture also reserves its serialization buffer in RAM.
The runner records cap, usage, peak and release counters after model
load, context creation, inference and teardown. The last snapshot retains
the 2 GiB host runtime reservation as long as it holds the contract.

The profile is limited to one discrete CUDA GPU, fixed caps, resident
weights and one independent inference context. It excludes the upstream
host MoE cache, adapters and additional accelerator backends. Opaque
CUDA driver and library allocations remain outside the ledger. The
explicit workspace pool uses synchronous frees and disables CUDA graphs.

Local checks covered argument rejection for incomplete caps, an unavailable
CUDA device and a CPU device selected under this profile. A regular CPU
Qwen3-0.6B request completed four decode steps and emitted logits and a
state digest. The runtime-floor and ordinary allocation-dispatch cases
passed. The Metal control could not initialize a command queue in the
local session.

On Ubuntu with an RTX 4070 Laptop GPU, both runs used the same
Qwen3.5-4B-Q4_K_M GGUF, `CUDA0`, 33/33 offloaded layers, a requested
128-token context, a 16-token microbatch, four CPU threads and the
prompt `The capital of France is`. The CUDA library had graph support
compiled in and CUDA Flash Attention compiled out.

- All 3,973,120 logits bytes matched the ordinary CUDA run. Both
  committed token IDs `[11751, 13, 198, 32]` and produced serialized
  state digest `ab0006a3a2a03d1f`.
- With caps of 3,072 MiB RAM and 4,096 MiB VRAM, ledger peaks were
  2,597.563 MiB and 2,677.750 MiB respectively. The RAM peak includes
  the 2,048 MiB runtime reservation and the diagnostic state copy.
  Model load released the 1 MiB staging buffer. Teardown reduced VRAM
  usage to zero and RAM usage to the held runtime reservation.
- At 64 MiB VRAM, the ledger refused the weight allocation. At
  2,620 MiB VRAM, weights fit and the ledger refused a 52,690,944-byte
  state buffer during context initialization. Both cases exited with
  failure without an inference report.

This qualifies one resident request for the explicit allocation
contract. Live cap changes and performance qualification remain open.

### CUDA expert streaming diagnostic

Add `--cuda-expert-streaming` to the fixed-cap CUDA runner for a model
supported by the existing expert source. The profile requires an explicit
microbatch of at most 32 tokens. Model loading leaves routed matrices on
disk. The existing CPU graph callback reads one expert at a time, submits
the projection on a separate CUDA backend and holds the lease until that
submission has completed. Expert payload and CUDA scratch charge VRAM.
Tensor arenas, I/O staging and gather/scatter buffers charge RAM. Scratch
admission can evict idle experts while the current expert remains pinned.
Both CUDA backends enable the observed allocation pool.

For indexed CUDA matmul, the number of token rows affects kernel selection.
A one-row replay of each route matched the resident Q4_0 projection at
one and three tokens and diverged at 32. Keeping the original microbatch
dimension for each expert, with zero padding for unused rows, passed all
15 assertions in the targeted CUDA executor case. This includes exact
projection outputs at 1, 3 and 32 tokens, explicit scratch admission and
release, domain accounting and teardown.

On the RTX 4070 Laptop GPU, a synthetic two-layer Qwen3 MoE GGUF with
16 experts per layer and Q4_0 routed weights passed a 59-token prefill and
four-token decode with `--ctx 128 --ubatch 32 --gpu-layers 99 --ram-mib 2100
--vram-mib 6 --cuda-expert-streaming`. The ordinary CUDA run and the
streaming run matched all logits bytes, token IDs `[69, 69, 69, 69]` and
state digest `5ce4c75d25133901`. The cache had 75 misses, 21 hits and 63
evictions. VRAM peaked at 6 MiB and returned to zero. RAM peaked at
2,049.125 MiB including the 2 GiB runtime reservation, which was the
remaining RAM charge after teardown. A four-expert fixture at 5 MiB VRAM
denied scratch during prefill and failed the request cleanly.

The diagnostic uses synchronous per-expert submission and LRU with fixed
caps.

#### Trained LFM2-8B-A1B projection

The RTX 4070 Laptop GPU comparison used the unchanged first four experts
from `blk.2.ffn_down_exps.weight` in the
[official Liquid AI Q4_0 GGUF](https://huggingface.co/LiquidAI/LFM2-8B-A1B-GGUF)
at revision `145e98524a2ed11d35a45e0080febcee063749b6`. The projection
has 1,792 inputs and 2,048 outputs. Two slots per token select cyclically
through the four experts using identical inputs on both paths.

- At 1, 3 and 8 tokens, all projected floats matched the resident CUDA
  `MUL_MAT_ID` output byte for byte.
- At 9 tokens, 24,734 of 36,864 values differed in their bits. Maximum
  absolute difference was `2.98023224e-7`. Differences persisted at
  12, 16, 24 and 32 tokens, with a maximum of `3.57627869e-7`.
- The projection test peaked at 9.50 MiB managed VRAM and returned it
  to zero after teardown. RAM returned to the 2 GiB runtime reservation.

On this GPU Q4_0 switches from indexed MMVQ to indexed MMQ at nine
tokens. Preserving token count preserves that boundary. MMQ still
chooses its tile configuration and Stream-K work partition from the
expert and route dimensions, which differ between the resident and
per-expert graphs. At nine tokens, increasing the expert axis from
four to 32 in the resident graph while retaining identical active
weights, inputs and routes changed 6,975 output values. An identical
four-expert repeat matched exactly. The first differing decision
inside MMQ remains to be isolated. Full trained-model inference,
other quantization types, performance and live cap changes require
qualification beyond the profile below.

#### Hybrid LFM2 graph with trained expert weights

A two-layer `lfm2moe` fixture combined a dense short-convolution block,
an attention/MoE block, deterministic synthetic tensors and the same
four trained Q4_0 down-projection experts. On the RTX 4070 Laptop GPU,
ordinary CUDA, budgeted resident CUDA and budgeted CUDA streaming
produced identical logits bytes, four committed tokens and state
digest `5e9b77d1db3d620a` for a 64-token prefill at `--ubatch 8`.

The streaming run also matched at 2,304 MiB RAM and 24 MiB VRAM caps.
Its peaks were 2,051.22 MiB RAM including the runtime reservation and
23.56 MiB VRAM. It recorded 24 misses, 21 evictions and 21 buffer reuses.
Teardown released all tracked device buffers.

#### Full LFM2-8B-A1B GGUF

On the RTX 4070 Laptop GPU, the full 24-layer official Q4_0 GGUF
(SHA-256 `63cd0f1be2f595293550d8dd5772782f211964d2979ad6113d7d32adafa9f785`)
was run with all layers offloaded, `--ctx 256 --ubatch 8`, and four
greedy decode steps after the 11-token prompt
`Finish this sentence: The capital of France is`. The ordinary
CUDA run and the streaming run matched all 1,048,576 logits bytes,
tokens `[5242, 521, 768, 3771]` and state digest `59cc66a3e874afcd`.

At caps of 2,560 MiB RAM and 1,024 MiB VRAM, the streaming ledger
peaked at 2,156.46 MiB RAM and 1,022.63 MiB VRAM. It returned to the
2 GiB RAM reservation and zero VRAM after teardown. The cache
recorded 2,034 misses, 234 hits, 1,703 evictions and 4,198,957,056
bytes of logical expert reads. In this single request, measured
prefill was 638 ms with streaming and 46 ms with ordinary CUDA.
Wider MMQ batches remain open.

#### Full LFM2.5-24B-A2B hybrid CUDA comparison

The AD-Q4_K_M GGUF copied to Ubuntu matched the source SHA-256
`6cfa4cafd31b97cb008041fe9fc76f3d4c3793ec044cf996695fbff4d1d23140`.
The ordinary and streamed profiles used the RTX 4070 Laptop GPU,
`--gpu-layers 14 --ctx 256 --ubatch 8`, the nine-token prompt
`Finish this sentence: The capital of France is` and a four-token
generation limit.

- Ordinary inference committed `[4741, 20, 309, 278]` with state digest
  `08217f45765d4354`. Streaming committed `[4741, 22]` before EOG with
  digest `f13e78d53ddbcb7f`. The first logit row already differed in
  all 128,000 values, with maximum absolute difference 3.40375.
- Streaming peaked at 2,897.07 MiB RAM and 2,048 MiB VRAM against
  caps of 4,096 and 2,048 MiB. It recorded 6,810,501,120 bytes of
  logical expert reads and returned to the 2 GiB runtime reservation
  and zero VRAM after teardown. Single-run prefill took 1,744 ms
  with streaming and 206 ms with ordinary inference.
- At `--ubatch 1` the first logit row still differed in all 128,000
  values, with maximum absolute difference 3.45753. The second
  committed token remained different.

Both paths offloaded 14 of 41 layers, but the streaming executor
evaluates every paged expert on CUDA. The ordinary run keeps the routed
projections of host-resident layers on CPU. The result excludes the
wider CUDA MMQ batch as a requirement for the divergence and leaves
the first differing operation open. This hybrid streaming profile
does not qualify for numerical equivalence.

#### Host-layer expert placement and CUDA fusion

The runner's `--tensor-trace` records hashes of named contiguous
F32/I32 prefill tensors. At `--ubatch 1`, the first differing sampled
tensor was `ffn_moe_gate-2`, the first routed gate projection of
LFM2.5-24B, in a CPU-resident layer. The ordinary `MUL_MAT_ID` had
hash `29a893fb3f7e6856` and the CUDA-streamed result had
`df44dc5d55718e60`. The diagnostic callback synchronizes between
sampled operations and interrupts backend fusion, so use untraced runs
to assess request equivalence.

With `--cuda-experts-match-layer`, experts in CPU-resident layers use
the generic CPU indexed matmul and retain the same disk source.
Expert payloads and CPU work allocations charge the host ledger.
Idle CPU expert buffers are released after each projection. Experts
in GPU layers use the existing observed CUDA path.

With default fusion, matching layer placement restored all four
reference token IDs at microbatches 1 and 8. The first logit row
still differed, with maximum absolute errors of 0.09853 and 0.19348,
respectively, and the state digests still differed. With
`GGML_CUDA_DISABLE_FUSION=1` on both paths and tensor tracing off,
all 2,048,000 logit bytes, the four token IDs and the serialized
state digest matched at each microbatch. The streamed peaks were
2,903.85 MiB RAM at microbatch 1, 2,952.07 MiB RAM at microbatch 8,
and 2,048 MiB VRAM at both. The checked caps were 4 GiB RAM and
2 GiB VRAM. Teardown left the runtime reservation and zero VRAM.

This qualifies the matching-placement profile against the unfused
reference for this prompt. Bitwise equivalence to the default fused
CUDA graph and localization of the remaining fusion difference are
still open.

#### Selective CUDA fusion control for LFM2.5-24B

The first differing output sampled inside the CUDA backend with fusion
active was `l_out-26`. The resident graph fused
`ffn_moe_weighted-26 -> ffn_moe_out-26` (one multiply, four views and
three adds) on CUDA. In the matching-placement streamed graph this
fusion did not run. Skipping just that reduction in the resident
reference restored the `l_out-26` hash in both the one-row and
eight-row prefill. The backend capture ran after kernel submission
without a scheduler evaluation callback. CUDA graph capture was
disabled for that readback, and the sampled runs retained the same
logits as their respective uninstrumented runs.

- At `--ubatch 8`, skipping only the layer-26 weighted reduction
  matched the existing streamed profile through all 2,048,000 bytes
  of logits, four committed tokens and state digest
  `12ba591590321567`. Fusion logs still recorded 238 resident and
  339 streamed fusions.
- At `--ubatch 1`, the same selective skip matched the first logit row,
  all four token IDs and final state digest, but 127,178 values in
  the second row differed by up to 0.003154. After correcting the
  layer-26 split, the next differing sampled branch was
  `ffn_moe_out-38`. A layer-38-only Top-K skip on both graphs was
  insufficient.
- At `--ubatch 1`, skipping the layer-26 reduction and CUDA Top-K
  routing fusion on both sides matched all 2,048,000 logit bytes,
  token IDs and state digest `b5f2f4bda28a6a21`. Other fusions stayed
  active (348 resident, 672 streamed), including RMSNorm,
  convolution-output plus residual, and applicable expert fusions.

Both exact comparisons used 4 GiB RAM and 2 GiB VRAM caps on the
streamed side. The streamed ledger peaked at 2,903.85 MiB RAM
for batch 1, 2,952.07 MiB RAM for batch 8, and 2,048 MiB VRAM
for both. Teardown left zero VRAM and the fixed RAM reservation.

Apply `docs/atomic-turbine-p10b-fusion-probe.patch` to a disposable
build. Run the same pair of resident and matching-placement streamed
requests above, setting `ATOMIC_P10B_SKIP_WEIGHTED_26=1` in both
processes. For the one-row comparison also set `ATOMIC_P10B_SKIP_TOPK=1`.
The layer name is specific to this offload split. The controls alter
the diagnostic reference and do not qualify bitwise equality against
the original default-fused reference. The runtime sources retain their
original fusion policy.

#### Reproducing the default CUDA fusions

The streaming diagnostic `--cuda-boundary-reduction` requires
`--cuda-expert-streaming --cuda-experts-match-layer` and a partially
offloaded LFM2 MoE model with op offload. It places the expert
weighting and sum of a CPU layer on the device of the next layer.
At the CPU-to-GPU boundary this lets the CUDA backend recognize
the same reduction as in resident inference, while the routed
matrix projections still run on the original layer's device.
The boundary is derived from placement. In the 14/41 offload
profile it is layer 26.

In an instrumented CUDA build this placement reproduced the
layer-26 weighted reduction and retained Top-K fusion, including at
layer 38. Enabling the resident allocation dependencies alone did
not recover the boundary reduction or numerical equality. Adding
them to the successful placement change had no effect on output
for this fixture. The final diagnostic reuses existing scheduler
and CUDA fusion behavior with no CUDA source changes.

On the same nine-token request and RTX 4070 Laptop GPU, the final
runner matched the original fully fused resident control at both
microbatches 1 and 8. All 2,048,000 logit bytes matched in each
run, with matching committed IDs and state digests
`ca782525397a5846` and `08217f45765d4354`, respectively.
Against the 4 GiB RAM and 2 GiB VRAM caps, ledger peaks were
2,903.85/2,048 MiB at microbatch 1 and 2,952.07/2,048 MiB at
microbatch 8 (RAM/VRAM). Teardown returned VRAM to zero and RAM
to the held runtime reservation. This evidence covers the tested
prompt, offload split and microbatches. All other profiles retain
their prior settings unless the diagnostic is explicitly selected.

#### Live CUDA budget change between decode calls

The `--cuda-resize-probe` runner option starts from 4,096 MiB RAM
and 2,048 MiB VRAM. After selecting the first token it requests
3,200 MiB RAM and 1,024 MiB VRAM, then evaluates that token under
the reduced caps. After selecting the second token it restores the
original caps and resumes evaluation in the same context. The
control runs between decode calls after synchronization and
uses the existing store eviction order. It checks the fixed
allocations, one expert working set and a caller-supplied transient
allowance in each domain (64 MiB for this probe) before requesting
either cap. This allowance is specific to the checked profile.

The LFM2.5-24B-A2B 14/41 offload, microbatch-8 request on the RTX
4070 Laptop GPU checked both 1 MiB cap rejections before shrink.
Rejection preserved the caps and residency. Shrink completed in
36.4 ms and reduced tracked VRAM from 2,047.31 to 1,023.94 MiB.
It released 1,023.38 MiB VRAM and 0.187 MiB host metadata in the
ledger. The CUDA device query reported 1,622 MiB more free VRAM
over the same interval. The two readings count different scopes.

The grow boundary had 541.375 MiB tracked VRAM in use. Grow
changed the cap without allocating any additional buffers. Subsequent
decode repopulated the cache to 988.44 MiB. All 2,048,000 logit
bytes, four committed tokens and state digest `08217f45765d4354`
matched both the fixed-cap streamed and ordinary CUDA controls.
Teardown returned tracked VRAM to zero and RAM to the held runtime
reservation. The host-cap test validates continued admission under
a lower limit. It does not establish substantial host RSS reclamation.
Automatic pressure and interleaved resize callers remain outside this
diagnostic.

#### Prefill resize with a partially read expert

`--cuda-prefill-resize-probe` attaches a one-shot callback to the first
CUDA expert cache miss. The source reads the first half into the admitted
staging buffer, the callback records the pending shrink, then the same
read finishes and validates the source before upload and publication.
The callback does not evict, resize or acquire another lease. At the
first synchronized microbatch boundary, the probe consumes the request
and checks that staging was released. It grows at the second boundary.
The ordinary source path retains one complete read per expert.

On the RTX 4070 Laptop GPU with the full LFM2.5-24B-A2B AD-Q4_K_M,
14/41 offloaded layers, context 256 and microbatch 8, the 33-token
prefill crossed five boundaries. Charged RAM I/O was 2.1875 MiB during
the split read and 0.5 MiB at the shrink boundary. The second microbatch
ran at 3,200 MiB RAM and 1,024 MiB VRAM. The original 4,096/2,048 MiB
caps applied from the third microbatch through decode.

Shrink released 949.5 MiB of tracked VRAM in 39.3 ms. The CUDA query
reported 1,424 MiB more free VRAM across that interval. Grow changed
admission limits without allocating buffers. The live run peaked
at 2,950.10 MiB RAM and 2,048 MiB VRAM. Teardown returned VRAM to zero
and RAM to its held runtime reservation.

The ordinary CUDA, fixed streaming and resizing streaming runs matched
all 2,048,000 logits bytes, tokens `[4741, 22, 440, 5141]` and state
digest `c2627eda8a0b6a83`. This checks deferred resize while a real
expert payload is partially read through the synchronous loader.

#### Worker read, deferred resize and cancellation

Add `--cuda-worker-read complete` or `--cuda-worker-read cancel` to the
prefill resize probe. The acquisition thread retains the staging buffer
and its ticket while a separate worker reads from the original GGUF.
The worker pauses after the first half. The acquisition thread records
the budget request while the worker is waiting, then signals it to
complete or cancel. Both budget changes are requested this way and
applied at synchronized microbatch boundaries after the read joins.
`cancel` cancels the first (shrink) read. The partial buffer is destroyed
without upload or cache publication, and the original demand is retried
from the expert's file offset. The grow read completes in both modes.

The same LFM2.5-24B profile ran 33 prefill tokens in five microbatches
and selected four decode tokens. Both modes matched all 2,048,000 logit
bytes, token IDs and state digest `c2627eda8a0b6a83` from the resident
and fixed-cap controls. At each request, RAM I/O was 2.1875 MiB
while the worker was paused and 0.5 MiB at the application boundary.
Cancelled and completed runs had equal cache hits, successful misses,
logical bytes and submissions. Cancellation incurred one additional
buffer allocation for the unpublished attempt.

Shrink released 949.5 MiB tracked VRAM in both modes (45.6 ms with
completion, 40.5 ms with cancellation). The contemporaneous CUDA
free-memory increase was 1,424 MiB. Grow allocated no buffers at the
boundary, and subsequent inference used the restored headroom.
Both runs peaked at 2,950.10 MiB RAM and 2,048 MiB VRAM against
the starting 4 GiB and 2 GiB caps, with zero tracked VRAM after teardown.
This tests cooperative cancellation before the second file read. Cancel
does not interrupt an OS syscall already in progress. No background
worker or split read is activated without the diagnostic.

#### Cancellation of a submitted Linux read

Configure the runner with `-DATOMIC_TURBINE_IO_URING_PROBE=ON` and
liburing development files. Add `--cuda-uring-read` to
`--cuda-prefill-resize-probe`. This submits an unlinked
`IORING_OP_READ` with `IOSQE_ASYNC` for the full expert range from
the open GGUF. After submission, the probe records the shrink request
and sends a separate cancellation for that read. The staging ticket
stays live until both completion entries arrive. The store discards
the unpublished attempt and retries that same expert. For grow, the
probe submits another asynchronous read and lets it complete.
The budget changes apply at synchronized microbatch boundaries.

On the RTX 4070 Laptop GPU and the same 33-token LFM2.5-24B profile,
the first 1,769,472-byte read returned `-ECANCELED` and cancellation
returned zero. The read had no completion entry visible before the
cancellation was submitted. The request for grow read all 1,769,472
bytes. After microbatch one, the caps changed from 4,096/2,048 MiB
(RAM/VRAM) to 3,200/1,024 MiB. The original caps were restored after
microbatch two, and prefill continued through microbatch five.

Shrink released 949.5 MiB of tracked VRAM, and the CUDA device query
reported 1,424 MiB more free VRAM. The ledger peaked at 2,950.10 MiB
RAM and 2,048 MiB VRAM, with zero tracked VRAM after teardown. All
2,048,000 captured logit bytes, token IDs `[4741, 22, 440, 5141]`
and serialized state digest `c2627eda8a0b6a83` matched both the
resident CUDA and fixed-cap streaming controls.

This result establishes cancellation after kernel submission on this
profile. The diagnostic observes request completion, with no
measurement of whether the storage device started servicing the read.
The build option defaults to OFF, and the runner opens a ring only
when explicitly requested.
