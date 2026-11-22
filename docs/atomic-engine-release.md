# Atomic Engine fixed-budget server profiles

These are the P11 profiles with targeted runtime evidence. Use the listed GGUF,
backend, memory caps and batch geometry together. Each profile covers one
text slot with greedy sampling. For a split checkpoint, pass the first shard
and keep the remaining shards alongside it.

The budget covers admitted engine allocations and the runtime reserve.
The OS file cache and total process footprint are measured separately.
Use `/metrics` while the server is idle to inspect
`llamacpp:engine_memory_budget_{applied,used,high_water}_bytes` on Metal.
CUDA exposes separate `llamacpp:engine_memory_ram_*` and
`llamacpp:engine_memory_vram_*` gauges. Compare used bytes with the applied
cap. The high-water gauge records the peak managed usage.

## Metal

### Build the server

From the repository root on Apple Silicon with the macOS SDK available,
configure a fresh Release build and build `llama-server`:

```sh
cmake -S . -B build-metal-release \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF -DGGML_CCACHE=OFF
cmake --build build-metal-release --target llama-server -j 6
```

Set `MODEL` to the local GGUF or first shard for the profile and `SLOTS` to a
writable directory. The commands below serve HTTP on loopback.

### Inkling 2 Mini AD-Q4_K with DSpark-3

Set `MODEL` to the local `Inkling-2-Mini-AD-Q4_K.gguf`. Use the same published
AtomicChat Q8_0 draft as IQ2_M and IQ3_S. The command uses the Metal build above and a 20 GiB budget
plus 100 MiB for KV. [Download the tested draft](../README.md#run-inkling-2-mini-on-apple-silicon).

```sh
./build-metal-release/bin/llama-server \
  -m "$MODEL" -md .work/models/atomicchat-dspark-3095098/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-ubatch 32 \
  --engine-budget-mib 20580 -fit off -c 4096 -b 1024 -ub 1024 -np 1 \
  -fa on -ctk q4_0 -ctv q4_0 -ctkd q4_0 -ctvd q4_0 -ngl all -ngld 99 -t 8 -tb 8 \
  --temp 0 --cache-ram 0 --ctx-checkpoints 0 --no-warmup --metrics \
  --ui --alias inkling-2-mini --host 127.0.0.1 --port 8080
```

For 8 GiB plus the same KV allowance, change `--engine-budget-mib 20580`
to `--engine-budget-mib 8292`.

The profile uses a 4096-token context, 1024-token prefill blocks, mode 3,
greedy sampling, a draft microbatch of 32 and loopback port 8080. Expert banks load through `pread`.
As the budget grows, the engine keeps the repeated core resident first,
then admits outer layers while reserving workspace for streamed reads.
At 20 GiB plus the 100 MiB KV allowance, the profile attempts to keep
all 24 physical layers' expert banks resident. If the full set cannot
be admitted, partial residency uses the regular streaming reserve.
The budget measures engine allocations. Physical memory must also cover
the OS and other processes.

### Qwen3.8 fixed-budget server

**Qwen3.8 Flash Next GSQ-RCO IQ3_XXS (`qwen4exp`)** on Apple M4 Max.
Use 8 GiB, context 4096, batch 256, microbatch 8 and eight CPU threads.
The selected server profile covers a 3481-token prompt and its repeat,
a short follow-up, and exact logits and slot state against ordinary Metal.
Short completion, non-thinking chat and natural EOS have separate evidence
at the effective context-256 geometry described in the family plan.

```sh
mkdir -p "$SLOTS"
build-metal-release/bin/llama-server -m "$MODEL" -ngl 99 -c 4096 -b 256 -ub 8 \
  -np 1 -t 8 -tb 8 -fit off --engine-budget-mib 8192 --no-warmup \
  --cache-ram 0 --ctx-checkpoints 0 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 8080
```

After `/health` returns `{"status":"ok"}`, send one short request:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say only OK."}],"chat_template_kwargs":{"enable_thinking":false},"temperature":0,"seed":42,"max_tokens":16,"id_slot":0,"cache_prompt":true,"return_tokens":true,"verbose":true}'
```

The fresh server build on M4 Max returned `OK` and EOS (`[3793, 248046]`).
A follow-up carrying that assistant turn and `Repeat your previous answer
exactly.` returned the same IDs. It reused 17 of 37 input tokens from the
live slot. The managed peak was 8,589,668,826 bytes under the 8,589,934,592-byte
cap. This smoke check covers the build and launch instructions above. See
the [model profile](atomic-engine-qwen38.md) for numerical controls and
the long-request results.

Wide Metal prefill with overlapped expert reads is the automatic path for
qualified streaming models at fixed budgets with `-c 4096 -b 4096 -ub 8
-np 1 -ctk bf16 -ctv bf16`. It applies to fresh text batches of at least
1024 tokens in multiples of 32 and falls back to sequential streaming
when its workspace does not fit. On the qualified Flash Next profile, a
fresh prompt longer than eight tokens with one output row also qualifies
at shorter or unaligned lengths. The graph uses the actual input tokens.
The [912-token handover measurements](atomic-engine-flash-next-partial-wide-8-16gib.json)
compare this path with sequential streaming at 8 through 16 GiB.
Use `--no-engine-prefill-window` for sequential streaming.
Wide and sequential graphs can produce different
logits, generated tokens and slot bytes. The
[wide prefill notes](atomic-engine-prefill-layer-batch.txt) describe
admission, read staging and numerical checks.

For the qualified LFM2.5-24B Metal profile, fixed budgets from 16 through
18 GiB keep the final complete prefill banks for decode. At exactly 12 GiB,
set `LLAMA_LFM_RETAIN_PREFILL_TAIL=1` to opt in. The banks keep their original
budget reservations and are released before the next prompt or a budget
change. At fixed budgets from 20 through 24 GiB, a fully resident MoE can
feed the wide prefill graph directly. If any layer is missing, the usual
streaming prefill remains available. See the
[LFM calibration](atomic-engine-lfm-residency-calibration-8-24gib.json).

Qualified Qwen3.6-35B-A3B Q4_K_M keeps the last complete prefill banks at
fixed 22 GiB without a draft and at fixed 20, 22, and 24 GiB with the
calibrated BF16 DFlash2 draft. At fixed 24 GiB without a draft, wide prefill
binds all 40 banks directly when they are resident. Default activation
requires the measured resident prefix and, with DFlash2, room for verification.
Set `LLAMA_QWEN35_PREFILL_RESIDENCY=0` to use the original path or `=1` to
try residency at other qualified fixed budgets between 8 and 24 GiB.
The [Qwen3.6 grid](atomic-engine-qwen36-prefill-residency-grid.json) records
the comparisons. The [LFM rollout measurements](atomic-engine-prefill-residency-rollout.json)
include production checks and the live resize.

Add `--engine-prefill-window` to also enable the legacy prefill windows.
The opt-in profile covers long text completion, stop, forced EOS, disconnect
during a prefill slice, same-slot recovery, and explicit file-based rollback
and continuation. Cancellation takes effect at the server slice boundary.
On the 3481-token fixture, windows reduced measured TTFT by about 25% and
logical expert reads by 51.5% while preserving exact logits and state.
The follow-up decode experiment measured about 12.7% higher TPS versus
the earlier window implementation on that server sequence. See the
[family plan and steady-state measurements](atomic-engine-qwen38.md) and
[window lifecycle evidence](atomic-engine-qwen38-window-p11.json).

At context 4096, the window path requires at least 320 uncached prompt
tokens and successful workspace admission. An ineligible request uses
sequential budgeted streaming. If a manual resize
is pending during window execution, the engine completes the current window,
releases its workspace, applies the request at the synchronized boundary, and
uses sequential streaming for the remainder. This resize check uses the
embedding API. A failed window after state mutation makes that context
unusable and requires a fresh context.

At context 1024 and a fixed budget of at least 18 GiB, the opt-in path
can process a batch of 65–256 tokens as a single layer-major window.
Use `-c 1024 -b 256 -ub 8 --engine-budget-mib 18432 --engine-prefill-window`
with the same single-slot Metal setup. Its 246-token comparison processed
30 eight-token microbatches and a six-token tail. First-token time was
22.25 s with the window off and 10.06 s with it on. Both runs streamed
the original GGUF shards within the cap. All 64 logit rows, token IDs and
the saved slot matched. The sequential pair used an uncontrolled macOS
file cache.

#### Experimental parallel reads during a window

Set `LLAMA_EXPERT_PARALLEL_READS=1` on the Qwen4Exp Metal server with
`--engine-prefill-window` to batch known demand misses for the next group.
After submitting the current group to Metal, the store admits the next
group in demand order and reads up to four original GGUF slices concurrently
into their unpublished slots. The submitted group retains its leases,
and the next group receives its leases only after the reads and checks finish.
The mode applies to eligible windowed prefill and reads the original GGUF
shards. Unset the variable to keep serial demand reads during Metal work.

One sequential M4 Max pair at 8 GiB on the 3617-token prompt reduced time
to first token from 145.62 to 125.69 s. Both modes read the same 827 GB of
expert slices and completed 57 windows. Logit hashes, token IDs and saved
slots matched. The OS file cache was uncontrolled, and the second run
recorded 92 GB of process disk reads during prefill versus 136 GB in the
first.

#### Experimental DFlash2 decode

The local Flash-Next DFlash2 draft can run against the streamed Qwen4Exp
Metal target with context 1024, microbatch 8, one text slot, and a fixed
18 GiB engine budget. The target, draft, recurrent rollback rows, and
32 MiB of draft slot-file staging share the engine ledger. Use the
matching DFlash2 GGUF and the normal single-slot launch settings with
windowed prefill disabled. Add:

```sh
export LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=1
mkdir -p "$SLOTS"
build-metal/bin/llama-server -m "$MODEL" -ngl 99 -c 1024 -b 256 -ub 8 \
  -np 1 -t 8 -tb 8 -ctk bf16 -ctv bf16 -fit off --engine-budget-mib 18432 \
  --no-warmup --cache-ram 0 --ctx-checkpoints 0 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 8080 \
  --spec-type draft-dflash -md "$DRAFT_GGUF" -ngld 99 --spec-draft-n-max 5
```

On the 246-token input / 64-token greedy output fixture, the target
accepted 41 of 110 draft proposals. The 19 rejection rounds removed all
69 rejected positions. All 64 output IDs matched the serial control.
Full logits differed starting at the second output, and the target slot
bytes differed from the serial control. The saved speculative slot was
byte-identical after restore and re-save in a fresh speculative server.
Decode after the first token measured 6.38 tokens/s serial and 5.03 tokens/s with
DFlash2, while logical decode reads rose from 12.37 GB to 23.31 GB.
File cache state was uncontrolled. The default remains serial decode.

### Qwen3.6-35B-A3B fixed-budget Metal server

The regular `llama-server` selects a startup profile for the fully offloaded
Qwen3.6-35B-A3B Q4_K_M text model on Metal. It uses a fixed engine budget
of 8 through 24 GiB inclusive, a single slot, BF16 KV, context and logical
batch 4096, and microbatch 8. For the layer ceiling, intermediate budgets use
the lower measured row. The ledger enforces the full requested byte cap,
and allocation admission can reduce the layer count.

| Engine budget (GiB) | Startup choice with adjacent BF16 draft | Direct layers without draft | Direct layers with BF16 DFlash2 |
|---:|:---|---:|---:|
| 8 | Serial | 2 | Legacy opt-in |
| 10 | DFlash2 | 5 | 4 |
| 12 | DFlash2 | 8 | 9 |
| 14 | DFlash2 | 18 | 13 |
| 16 | DFlash2 | 22 | 18 |
| 18 | DFlash2 | 22 | 22 |
| 20 | DFlash2 | 31 | 26 |
| 22 | DFlash2 | 31 | 31 |
| 24 | DFlash2 | 40 | 40 with six proposals and eight threads for generation and prompt processing, otherwise 36 |

At 8 GiB, the Metal profile launcher automatically selects the qualified
compact draft when available and uses one resident target layer. The
adjacent BF16 draft alone does not activate this compact profile.
The serial column uses the measured no-draft candidates. At 24 GiB, 35
and 40 layers had nearly identical mean total request time. All 40 layers
gave faster decode. The sweep and individual requests are in the
[fixed-budget measurement record](atomic-engine-qwen35-fixed-grid.json).

The server selects DFlash2 automatically from 10 through 24 GiB when a
compatible BF16 `qwen35-dflash2.gguf` is next to the target GGUF. With a
missing or incompatible sidecar, it selects the serial profile. Use
`--spec-type draft-dflash` to require the adjacent sidecar, or supply a
compatible draft file with `-md`. The
server checks the draft selector and target-layer mapping and reports an
error if an explicitly requested draft is unavailable or incompatible.
Use `--spec-type none` for serial execution and `LLAMA_QWEN35_PROFILE=0`
to bypass the automatic profile. The historical explicit 8 GiB DFlash2
experiment uses a supplied draft, `--spec-type draft-dflash`, and
`LLAMA_EXPERIMENTAL_BUDGET_DFLASH2=1`.
The [default-policy recheck](atomic-engine-qwen36-dflash-default-recheck.json)
measures both modes at the budgets affected by this startup choice. The
[current defaults and handover timings](atomic-engine-qwen36-lfm-fast-default-8-24gib.json)
include the 40-layer resident verifier at 24 GiB. To restore the earlier
Qwen profile, set `LLAMA_QWEN35_FAST_PREFILL=0` before startup.
[Greedy output differences and reproduction](atomic-engine-qwen36-fast-prefill-known-issue.md).

```sh
build-metal/bin/llama-server -m "$QWEN_MODEL" --engine-budget-mib 24576 \
    -fit off -ngl 99 -np 1 -c 4096 -b 4096 -ub 8 -ctk bf16 -ctv bf16 \
    -t 8 -tb 8 --cache-ram 0 --ctx-checkpoints 0 --no-warmup
```

On the automatic DFlash2 path the default draft maximum is six and the
draft microbatch is 128. Hybrid verification streams the missing expert
layers. The 40-layer profile verifies directly with resident target weights. Both target and draft share the ledger. A live resize uses the
existing synchronized release path and retains the current decoding mode. To prepare the draft from
`incoai/Qwen3.6-35B-A3B-DFlash2`, use `convert_hf_to_gguf.py --outtype bf16
--target-model-dir TARGET_TOKENIZER_DIR`.

At 24 GiB, passing a Q4_0 DFlash2 draft with BF16 predecessor and successor
codebooks via `-md` selects 37 resident target layers. Prepare the compact draft from the converted BF16 file and add
`-md qwen35-dflash2-q4_0-selector-bf16.gguf` to the server command above.

```sh
build-metal/bin/llama-quantize --tensor-type 'selector_(predecessor|successor).weight=bf16' \
    qwen35-dflash2.gguf qwen35-dflash2-q4_0-selector-bf16.gguf Q4_0
```

The [24 GiB compact draft measurements](atomic-engine-qwen35-compact-draft-24gib.json)
record uncached 4000-token prompts, 64-token greedy outputs, engine ledger
peaks, and the effect of increasing the layer count to 38.

The wide prefill graph writes the selected target layer inputs to the
existing device feature buffer. The draft reads these rows to seed its
KV cache. The draft weights, feature buffer, target rollback snapshots
and up to 128 MiB of draft slot-file staging share the engine budget.
For a four-worker wide prompt, the engine first tries one 320 MiB Metal
scratch buffer for layer-local temporaries across those workers. It
keeps layer outputs and draft features live and reserves both complete
expert banks before applying prompt state. If that admission fails, the
engine retries with private worker scratch and the usual bank admission.
Set `LLAMA_EXPERIMENTAL_PREFILL_DFLASH2_SHARED_SCRATCH=0` to select that
control directly.

An unadmitted wide graph falls back to sequential prefill. Batched
verification can change target logits and greedy tokens relative to
single-token decode. Use `--spec-type none` for the serial control.

Set `LLAMA_EXPERIMENTAL_DFLASH2_ADAPTIVE=1` alongside the budgeted DFlash2
opt-in to choose how many draft tokens to send to the target for greedy
requests. `--spec-draft-n-max` remains the upper limit. The policy combines
selector confidence and observed acceptance at each position, then chooses
a contiguous prefix based on expected committed tokens and the estimated
cost of verifying it. It models an extra verification row as half the
cost of the anchor and tries the full allowed prefix every eighth draft
to collect feedback. The policy resets per prompt and counts only the
accepted prefix and the first rejection as observations. Discarded rows
never enter the target batch, so they do not route or read target experts.
The existing target verification and rollback still apply to sent rows.

Add `LLAMA_EXPERIMENTAL_DFLASH2_TIMED_COST=1` to fit the extra-row cost
from the last 32 eligible target decode batches within each greedy request.
The timer covers target decode through its existing synchronization,
including expert reads. Prompt batches, partial batches and output-limited
rounds are excluded. The fit starts at 0.5, uses that prior while timing
samples are sparse, and limits the influence of stalled reads.

#### Experimental combined FP32 Metal MoE

Set `LLAMA_EXPERIMENTAL_METAL_MOE=1` to combine streamed Qwen4Exp
up/gate, SwiGLU and down operations in one Metal graph per group of at
most 16 experts. The intermediate activations stay in the graph. It uses
the existing FP32 vector matmul and SwiGLU kernels, with the original GGUF
weights; it is not a new fused shader or a reduced-precision path. Known
expert reads also use the existing batched reader for each projection.
The combined change therefore measures both graph batching and batched
reads, not the isolated contribution of either one.

This path accepts at most eight token rows and 32 routes per row, with
separate gate/up weights and unclamped SILU. The graph hook requires
outputs for every input row, as in decode and speculative verification.
The normal last-output-only prefill remains on the existing path. LoRA,
rotations, explicit precision overrides, unsupported shapes, and projection
scales or gate/up biases retain the ordinary graph. Router weights, expert
reduction, shared FFN, attention and recurrent state stay unchanged.

Workspace, graph metadata and all simultaneously held weight leases are
charged to the engine ledger. Groups shrink to fit the remaining budget;
an opt-in initialization that cannot fit the minimum working set fails
explicitly. Do not combine this experiment with prefill windows, explicit
non-LRU policies, projection observers, demand traces, preview or early
reads. These combinations are rejected by their capability/configuration
boundary. Unset the variable to restore the ordinary path without the
additional workspace, batched reads or GPU work.

The qualified real-model configuration is the DFlash2 command above with
`LLAMA_EXPERIMENTAL_DFLASH2_ADAPTIVE=1`, the ordinary two-shard Flash-Next
IQ3_XXS GGUF, 18 GiB, context 1024 and microbatch 8. KV cache types remain
as specified in that command; FP32 here describes expert arithmetic and
intermediate activations, not the KV cache or quantized weight storage.

One sequential off/on M4 Max pair on the 246-token input / 64-token output
fixture measured 25.85 / 25.70 s to the first token and 5.59 / 7.65 tokens/s
after it (1.37x decode throughput). Expert graph submissions fell from
7404 to 2468, while logical decode reads were 17.74 / 17.76 GB. Neither run
reused a prompt prefix; the OS file cache was uncontrolled. Both runs used
the same diagnostic binary, including full-logit capture.

All 64 complete logit rows, output IDs and 133884412-byte slot files were
byte-identical. Both accepted 34 of 48 proposals; seven rejection rounds
removed the same 14 positions. Erase/restore/re-save preserved each slot
file, including its draft state. The combined path executed 1392 times
during decode and zero times during measured prefill, and both engine
high-water marks stayed below 18 GiB. These are one-pair request timings,
not an isolated kernel speedup or a repeatability estimate.

#### Experimental resident-first MoE pipeline

Add `LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE=1` alongside
`LLAMA_EXPERIMENTAL_METAL_MOE=1` to overlap known-demand expert loading
with the combined FP32 Metal graph. The initial group selects experts
whose up, gate and down slices are already resident and pins them without
I/O. After submitting that group, the calling thread acquires the next
group while Metal executes. The existing batched reader performs the
reads. No permanent worker or second graph workspace is created.

Both groups retain their leases until GPU completion. Read errors drain
the submitted graph before any referenced lease can be released or reused.
The next group never writes the current group's weights or graph scratch.
Outputs scatter to their original route/token positions; routing weights
and the final reduction order are unchanged. This is known-demand overlap
within one MoE, not a predictor for future layers.

The group width is bounded by capacity for two complete up/gate/down
groups plus the existing read headroom in the shared ledger. If two
experts cannot fit, execution falls back to the sequential combined path
without increasing the minimum memory requirement. All existing combined
MoE qualifications and incompatible-mode checks still apply. The off path
does not inspect residency or start pipeline reads; only the configuration
flag/counters and one dispatch branch are added to the combined executor.

The private streaming counters `moe_resident_starts` and
`moe_pipeline_groups` report resident starts and acquisitions between
submission and explicit completion wait. Profile `overlap_read_bytes`
counts reads issued in that interval; it does not measure the intersection
of actual GPU execution and I/O. The acquisition and compute wall timers
overlap and must not be summed as independent phases.

Portability beyond the initially qualified profile is tracked in
[MoE Portability](moe-portability-followup.txt).

One sequential pipeline-off/on M4 Max pair with the same 246-token input,
64-token output, adaptive DFlash2, 18 GiB, context 1024 and microbatch 8
measured 7.61 / 8.96 tokens/s (1.176x). First-token latency was 27.27 /
26.51 s; neither measured prefill used the pipeline. Explicit expert Metal
wait time fell from 1.590 to 0.712 s, with 17.763 / 17.753 GB of logical
expert reads and 9.790 / 9.645 GB of process disk reads during decode.
Pipeline mode issued 2921 expert graphs versus 2468 in the control: splitting
resident work adds submissions even as it reduces blocking waits.

During decode, 1343 MoE calls started with a resident group and 1529 next
groups were acquired between submit and wait; 16.382 GB of logical reads
were issued in those intervals. All 64 full logit rows, token IDs and the
133884412-byte target-plus-draft slot matched. Both modes accepted 34 of
48 proposals, and seven rejection rounds removed the same 14 positions.
Slot erase/restore/re-save was byte-identical, and both ledger peaks were
19327288610 bytes, below the 19327352832-byte cap. The pair used the same
profiling/logit-capture binary, without concurrent GPU work or compilation.
OS file cache was uncontrolled; these results are not a repeated-trial
estimate or a qualification of other models and budgets.

#### Experimental selective expert reclamation

Set `LLAMA_EXPERIMENTAL_EXPERT_RECLAIM=1` to replace allocation-failure
cache flushing with size-aware reclamation in the Qwen4Exp Metal target.
This option is independent of the combined MoE and pipeline flags; the
qualified comparison below enables both of those flags in both modes.
Demand tracing and other backend/architecture profiles retain the old
allocation-reclamation callback.

Context scratch/state allocation scopes pass the full rounded reservation
and memory domain through a private callback. The store checks the current
ledger target and evicts unpinned victims in the existing policy order
until the reservation fits. It does not allocate weights, perform reads,
retire pinned entries, change the budget or reserve the new buffer itself.
The allocator retries admission against the ledger after reclamation.
If pinned or non-cache allocations prevent admission, the request fails
without falling through to a full-cache flush. Requests larger than the
entire target do not pointlessly evict the cache.

Ordinary `trim()`, explicit resize and lifetime cleanup retain their
previous semantics. The existing no-argument callback is retained for
the disabled path. The shared allocation scope gains two pointer-sized
fields and checks the additional callback only after failed admission;
context boundaries obtain an empty capability when this mode is disabled.
No worker, I/O, GPU work or extra cache workspace is added by that path.
The private streaming counters `reclaim_calls`, `reclaim_evictions` and
`reclaim_bytes` count only selective reclamation; freed bytes include the
victim's backend buffer and tensor metadata.

In one sequential off/on M4 Max pair with the combined MoE pipeline and
adaptive DFlash2 enabled in both modes, 246 input / 64 output tokens,
18 GiB, context 1024 and microbatch 8, the same 5963776-byte allocation
started with 19322501122 bytes charged against a 19327352832-byte cap.
Full reclamation freed 10956220080 bytes. Selective reclamation evicted
two projection buffers, freeing 2097888 bytes: existing free capacity plus
these victims covered the reservation. No pinned weights were retired.

Decode measured 5.86 / 6.87 tokens/s (1.173x); logical expert reads were
17.753 / 15.144 GB and process disk reads were 17.159 / 14.115 GB during
decode. First-token latency was 28.61 / 30.77 s, so the full request did
not become faster in this pair. Selective reclamation ran once during
decode and never during measured prefill. OS file cache was uncontrolled,
and these samples do not establish repeatability or a whole-request gain.

All 64 complete logit rows, output IDs, target-plus-draft slot bytes and
slot erase/restore/re-save results matched. Both modes accepted 34 of 48
draft proposals; seven rejection rounds removed the same 14 positions.
Both ledger peaks were 19327288610 bytes, below the 18 GiB cap. Five
focused admission/cache tests passed 62 assertions, including preservation
of pinned entries, pending lower targets, impossible admissions, ordinary
allocation dispatch, legacy reclamation and explicit lifecycle trimming.

#### Experimental packed expert reads

For the Qwen4Exp Metal budgeted profile, build an optional local pack
from the original GGUF shards supplied in split order:

```sh
python3 tools/expert-source/pack-qwen4exp-experts.py \
  --output "$PACK" "$MODEL" "$MODEL_SHARD_2"
export LLAMA_EXPERT_PACK="$PACK"
```

Use the same server command and window flags. The pack adds about 40 GiB
of disk use for the IQ3_XXS checkpoint. One demand read fetches an expert's
gate, up, and down slices into staging charged to the same engine limit.
The reader checks the source shard identities and each packed record's CRC
before publishing its weights. Rebuild the pack after replacing or copying
the source shards. Unset `LLAMA_EXPERT_PACK` to use the original reader.
A missing pack or insufficient staging admission also uses GGUF reads.

In one sequential M4 Max comparison on the 3617-token prompt at 8 GiB,
both modes used the prefill window. Packing reduced prefill demand reads
from 1,422,456 to 474,152 at about 827 GB of logical reads in each mode.
Time to first token rose from 140.45 to 236.12 s and decode fell from
7.99 to 5.08 tokens/s. The logit hashes, token IDs and saved slots matched.
The packed mode added about 178 MiB to the engine floor. The macOS file
cache was uncontrolled across the two runs.

### Ornith fixed-budget server

**Ornith 1.5 35B A3B Q4_K/IQ4_XS (Qwen3.5 MoE)** on Apple M4 Max.
Use 5 GiB, context and batch 512, microbatch 32 and eight CPU threads.
Greedy HTTP completion covers first and repeated requests, EOS/stop, SSE
disconnect and reuse, file-based save/erase/restore, impossible budget and
controlled source modification. The profile uses bounded known-route
overlap, prepared expert maps and the budget-admitted route index.
Manual shrink/grow has separate embedding API coverage. See the
[P11 checks and failure handling](atomic-engine.md#p11-fixed-budget-release-profile).

```sh
mkdir -p "$SLOTS"
build-metal-release/bin/llama-server -m "$MODEL" -ngl 99 -c 512 -b 512 -ub 32 \
  -np 1 -t 8 -tb 8 -fit off --engine-budget-mib 5120 --no-warmup \
  --cache-ram 0 --ctx-checkpoints 0 --metrics --slots \
  --slot-save-path "$SLOTS" --host 127.0.0.1 --port 8080
```

LFM2.5-24B on Metal has separate engine qualification at an initial 4 GiB
budget, context 512 and microbatch 32, including manual resize.
Its HTTP profile is qualified on CUDA below. See
[the Metal engine profiles](atomic-engine-second-architecture.md#qualified-support-profiles-and-limits).

## CUDA

Both selected profiles use Ubuntu, one RTX 4070 Laptop GPU with 8 GiB of
physical VRAM, separate 4096 MiB RAM / 2048 MiB VRAM engine caps, four CPU
threads, microbatch 8 and a single text slot. Use the qualified CUDA build
and matching backend libraries. Select the device explicitly and retain
partial offload.

- **LFM2.5-24B-A2B-AD-Q4_K_M**, 14 GPU layers, context and batch 256.
  Short HTTP completion, stop/EOS, SSE recovery and slot round-trip pass.
  See the [launch command and evidence](atomic-engine-cuda-server.md#reproduction).
- **Qwen3.8 Flash Next GSQ-RCO IQ3_XXS**, 22 GPU layers, context 4096 and
  batch 128. Short raw completion, stop, SSE recovery and file-based slot
  restore pass. Selected non-thinking chat cases pass with natural EOS,
  including a long two-turn dialogue that reuses 3473 of 3493 input tokens
  from the live slot and matches ordinary CUDA logits and state byte for
  byte. A selected [two-turn thinking dialogue](atomic-engine-qwen38-cuda-chat.md#two-turns-with-thinking-and-preserved-history)
  preserves reasoning, reuses 93 of 122 tokens and matches ordinary CUDA
  logits and slots byte for byte. A selected [single tool call and result](atomic-engine-qwen38-cuda-chat.md#single-tool-call-and-result-in-the-same-slot)
  reuses 322 of 349 tokens from the live slot with exact logits and state.
  A 3481-token raw HTTP prompt passes across 28 server batches,
  with identical output IDs and byte-identical full slot files. A fresh
  server restores the long slot and reuses 3484 cached tokens for an exact
  continuation. The long engine comparison separately verifies full logits.
  See the [launch command](atomic-engine-qwen38-cuda-server.md#reproduction-and-remaining-boundaries),
  [chat evidence](atomic-engine-qwen38-cuda-chat.md),
  [HTTP prefill](atomic-engine-qwen38-cuda-http-prefill.md),
  [HTTP restore](atomic-engine-qwen38-cuda-http-restore.md) and
  [long engine comparison](atomic-engine-qwen38-cuda-long-prefill.md).
  The [same-slot long series](atomic-engine-qwen38-cuda-http-series.md)
  passes an exact repeat and a one-token edit with full re-evaluation.

### CUDA disk prefix checkpoints (opt-in)

P11 is complete for this extension to the single-slot Qwen3.8 CUDA profile
above. Add `--engine-prefix-checkpoints "$PREFIX_DIR"` to that profile's
launch command, with `PREFIX_DIR` set to an existing writable directory,
and send requests with `cache_prompt=true`. The server owns the private
checkpoint files for the life of its process. Plan for two full prefix
states and space for a staged write during replacement. Public slot files
remain the interface for saving state across server restarts.

The checked 3481-token repeat reused 3456 tokens and had 4.794 s TTFT with
integrity checking. The selected one-token edit reused 1664 tokens. Missing,
truncated and same-size corrupted files each caused a full prefill in the
same slot. Full sampling logits and complete saved slots matched ordinary
CUDA in the selected comparisons within the 4 GiB RAM / 2 GiB VRAM engine
caps. See [launch and qualification details](atomic-engine-qwen38-cuda-prefix.md#opt-in-server-qualification)
and [checkpoint integrity](atomic-engine-qwen38-cuda-prefix.md#checkpoint-integrity).

## Release boundaries

Keep `-fit off`, `--cache-ram 0` and `--ctx-checkpoints 0` for these
profiles. File-based slot actions require `--slots` and a writable
`--slot-save-path`. Budgeted multi-slot operation, automatic checkpoint
rollback, prolonged soak, multimodal inputs and speculative decoding
remain outside the release profiles. Native macOS warning/critical
pressure delivery is pending an isolated test environment. Replay and
manual memory controls have separate engine evidence. CUDA resize
qualification is scoped to the engine at context 2048. The checked
Qwen3.8 CUDA HTTP profile at context 4096 uses fixed caps.

Treat additional GGUFs, devices, context geometry and server workloads as
new qualification cases. Known-route demand overlap is distinct from
next-layer preview. Preview and tree/MTP verification remain experimental
or deferred. The fixed-budget checks require only the relevant cases
from the linked reports.

#### Experimental compact expert buffers

Set `LLAMA_EXPERIMENTAL_EXPERT_COMPACT=1` to use per-slice allocation
sizes in the Qwen4Exp Metal executor. This is independent of combined MoE,
its resident-first pipeline and selective reclamation; the comparison
below enables all three in both modes. Other architectures/backends and
an unset flag retain their existing allocation policy.

Instead of rounding all small projections to the largest power-of-two
slot, each gate/up/down slice uses its backend-required allocation rounded
to 64 KiB. This retains the existing generic granularity, covering Metal
page rounding and keeping every physical buffer and tensor arena charged.
The executor's minimum working set and pipeline admission use the reduced
maximum slice allocation. The replacement policy still chooses the same
unpinned victim; compact mode only reuses that buffer when its size equals
the required size. An oversized victim is freed before normal bounded
admission retries, preventing large buffers from accumulating behind small
slices. The tradeoff is potentially more backend allocations.

Lease pinning, staging headroom, padding initialization, failed-read
cleanup, synchronized retirement and explicit resize semantics are
unchanged. The original GGUF bytes and FP32 routed arithmetic are unchanged.
No new backend hook, public ABI, worker or workspace is introduced. Disabled
mode adds one constructor environment lookup, one stored boolean and one
short-circuited check when evaluating an otherwise reusable victim; it
performs no feature I/O or GPU work. Cache-payload diagnostics used for the
comparison are confined to the external probe binary, not production code.

In one sequential fixed/compact M4 Max pair on the same ordinary two-shard
Flash-Next GSQ-RCO IQ3_XXS GGUF, 246 input / 64 output tokens, 18 GiB,
context 1024 and microbatch 8, first-token latency was 28.46 / 20.92 s
and decode was 10.39 / 11.32 tokens/s (1.089x). Both modes used adaptive
DFlash2 with maximum draft length 5 and no prompt-prefix reuse. A separate
15-input / 8-output warmup preceded each measured request.

At the end of decode, cached projection slices increased from 10443 to
17383. Their useful GGUF payload was 6.100 / 10.132 GB in 10.950 / 10.948 GB
of physical buffers: buffer padding fell from 44.30% to 7.45%. These
percentages exclude tensor arenas (3.843 / 6.397 MB), which remain charged
separately. Remaining padding comes from the retained 64 KiB granularity.
Logical expert reads fell from 47.734 / 28.775 GB during prefill and from
15.144 / 8.252 GB during decode. Process disk reads during decode were
12.150 / 7.355 GB; these include non-expert reads and depend on OS cache.

The lower read volume did not translate proportionally into throughput:
decode backend allocations increased from 119 to 11548, and the profiled
allocation stage rose from 0.029 to 0.517 s. Reuses fell from 25794 to 2609.
Admission rose from 0.031 to 0.203 s while source-read time fell from
3.534 to 2.421 s. These overlapping pipeline intervals are not additive
whole-request timings. Selective reclamation still ran once during decode,
evicting two fixed buffers / four compact buffers for the same 5963776-byte
request; it did not flush the cache.

All 64 complete committed logit rows and token IDs matched exactly, as did
the 133884412-byte target-plus-draft slot. Erase/restore/re-save was
byte-identical in both modes. Both accepted 34 of 48 draft proposals;
seven rejection rounds removed the same 14 positions. Ledger peaks were
19327288610 / 19327352690 bytes, both below the 19327352832-byte cap.
Two focused CPU cache cases and one compact Metal pipeline case passed
91 assertions with no failures or skips, covering legacy oversized reuse,
compact mixed-size replacement, pinned and failed leases, FP32 reference
outputs, tight-budget fallback and read failure with GPU work in flight.
No aggregate suite was run. This one pair used an uncontrolled OS file
cache, fixed mode first, and no concurrent compilation or GPU workload;
it does not establish repeatability or qualify other models and budgets.

#### Experimental recycling of compact buffers

Set `LLAMA_EXPERIMENTAL_EXPERT_RECYCLE=1` together with
`LLAMA_EXPERIMENTAL_EXPERT_COMPACT=1` to retain already-evicted compact
buffers by allocation size. The executor qualifies Qwen4Exp Metal only,
with maximum expert payload at most 1 MiB. The recycling option does not
change GGUF bytes, FP32 routed arithmetic, the graph or draft acceptance.

The per-size bound covers up/gate slices in two 16-expert pipeline groups.
A separate admission path first looks for an unpublished buffer whose
physical and tensor-arena sizes exactly match the request. Otherwise it
uses normal admission and the existing victim policy. Unpinned victims of
other sizes may enter a bounded pool: 64 per size, 256 buffers total,
no buffer larger than 1 MiB. Overflow victims are freed. Pooled buffers
retain their original buffer and arena tickets; their capacity competes
with resident weights inside the same managed limit. No extra uncharged
weight cache or eager allocation is introduced. The fixed 256-slot
index, like entry control metadata, belongs to the existing runtime envelope.
The pool can cause a few extra policy evictions while it fills; it never
searches useful residents for a more convenient size-class victim.

Read staging still needs admission. If it cannot fit, other pooled buffers
are freed before useful weights. If there are no available victims, spares
are released before retrying fresh admission, so the pool cannot make an
otherwise-fitting request fail. Failed or cancelled reads destroy the
unpublished selected buffer instead of returning partially written weights
to the pool. Published leases, synchronized release and zero-tail handling
retain their existing contracts.

Selective reclamation, budget shrink and idle eviction free pooled buffers
before published weights. Explicit trim drains the pool. Demand tracing
drains it and uses the previous admission path while tracing is enabled.
Policy-history admission, minimum-budget reporting and graph/pipeline
capacity calculations count spares as reclaimable. `pooled_buffers`
and `pooled_bytes` describe unpublished charged storage separately from
`resident_bytes`; `pool_reuses` counts successful publications sourced from
the pool. Reclamation counters include freed spare buffers and their arenas.

Disabled mode retains the original admission loop. Structural overhead is
an additional constructor environment lookup, a null pool pointer, expanded
private statistics, and pool-presence checks at admission/reclamation and
cleanup boundaries. No pool index, buffer, worker, I/O or GPU work is created
when disabled. The ordinary backend and public ABI remain unchanged.

In one final sequential off/on M4 Max pair, both modes enabled compact
buffers, combined FP32 MoE, the resident-first pipeline, selective
reclamation and adaptive DFlash2 (maximum draft length 5). Conditions were
the same ordinary two-shard Flash-Next GSQ-RCO IQ3_XXS GGUF, 246 input /
64 output tokens, 18 GiB, context 1024, microbatch 8, one slot, no reused
prompt prefix, and a separate 15-input / 8-output warmup per mode.

Decode buffer allocations fell from 11548 to 3363 (70.88% fewer), and the
allocation stage fell from 0.496 to 0.175 s. Pool hits during decode were
9175; total reuses rose from 2609 to 10891. New allocations remain when
the bounded pool has no suitable spare. Prefill allocations fell from
40875 to 18431. The pool held 123.090 MB at first token and 93.766 MB at
request end, including arenas; all of it remained inside the engine cap.
This is a deliberate capacity/retention tradeoff, not allocation elimination.

Decode measured 12.28 / 12.82 tokens/s (1.044x), and first-token latency
was 19.70 / 19.09 s. Logical expert reads during decode were 8.252 / 8.308 GB
(0.69% more); process disk reads were 5.497 / 5.905 GB. During prefill,
logical expert reads were 28.775 / 28.897 GB. Some cache capacity is now
occupied by spares, so fewer allocations do not imply fewer reads. Stage
timers overlap with GPU execution and are not additive request timings.
A preliminary pool of two buffers per size, sixteen total, reduced decode
allocations only to 10417 and was replaced by the pipeline-sized bound.

All 64 complete committed logit rows, token IDs, candidate rounds and the
133884412-byte target-plus-draft slot matched exactly. Slot erase/restore/
re-save was byte-identical in both modes. Both accepted 34 of 48 proposals;
seven rejection rounds removed the same 14 positions. Both ledger peaks
were 19327352690 bytes, below the 19327352832-byte cap. Minimum-budget
reports matched at all three sampled phases, including while spares were
present. Selective reclamation ran once during decode in both modes; the
pooled mode freed four buffers for the 5963776-byte request, releasing
spares before published weights.

Focused CPU cases covered size-class reuse, complete physical accounting,
pinned/failed ownership, the pool bound, selective reclamation, shrink,
trace fallback and idle cleanup. The compact mode without recycling also
passed its existing focused cache case. After adjusting the bound and
working-capacity accounting, the two changed CPU cases (30 assertions) and
one Metal pipeline case (67 assertions) passed without failures or skips.
The Metal case covered exact FP32 reference outputs, cold/hot execution,
tight-budget fallback and a read failure after GPU submission. No full
suite was run. The final pair used the same diagnostic binary, fixed order
(off then on), no concurrent compilation/GPU workload and an uncontrolled
OS file cache. It does not establish a repeatable 4.4% speedup or qualify
other model/backend/budget combinations.

#### Experimental decode lookahead for the next MoE layer

Set `LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD=1` with the Qwen4Exp Metal MoE
pipeline. This option uses ordinary GGUF shards and requires a host-readable
F32 or BF16 next-layer router. It predicts from up to two current activation
rows, ranks the union of their four highest-scoring routes by normalized
score, and reads missing gate/up/down slices for at most two experts. The
actual routing, expert arithmetic and reduction use the existing graph.
Managed server prompt graphs do not launch lookahead.

The optional workspace is one charged staging buffer of up to six maximum
slices and one bounded dispatch group. A background job can continue across
the current MoE fence and the next layer's work. A real demand miss with the
same tensor and expert ID can copy a completed speculative slice into its
normally admitted weight buffer. A matching miss may wait for the bounded
batch to complete. Unmatched slices are discarded before workspace reuse.
Lookahead alone does not publish weights or change cache recency. The graph
owner drains the job at graph completion. An I/O failure clears the batch
and disables lookahead for the context. Actual demand still validates reads.

In the sequential off/on/off comparison on M4 Max (same 246 input / 64
output tokens, 18 GiB, context 1024, microbatch 8, and compact recycling,
point reclamation and adaptive DFlash2 enabled), decode was 13.71 /
14.51 / 14.19 tokens/s. First-token latency was 18.50 / 19.07 / 18.47 s.
The OS file cache was uncontrolled. The series establishes that prediction
and consumption ran with exact results, with modest decode throughput
differences in this workload.

The option read 1954560000 speculative payload bytes, including 1146700800
bytes consumed by actual demand misses and 807859200 unused bytes (41.33%).
Worker read time was 404 ms, of which 393 ms fell outside caller waits on
the job. This includes concurrency with both CPU and GPU work. Demand read
time plus waits on speculation fell from 2.003–2.054 s to 1.801 s, while
prediction took 376 ms and total logical expert reads increased from
8.308 to 9.120 GB. The workspace charged 5541152 bytes inside the cap.
All 64 logit rows, output tokens, the target-plus-draft slot, its restore
round trip, and seven rejection rounds covering 14 positions matched.
Two focused CPU cases and one Metal case passed.

#### Metal prediction and independent slice completion

With decode lookahead enabled, `LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD_METAL=1`
selects the experimental Metal predictor and independent read completion.
The qualified next-layer router, top-k and selected scores form a prefix of
the final MoE group graph. The CPU waits for the prefix command buffer,
merges up to eight ID/score pairs, and starts reads while the remaining
commands execute. The ordinary graph submission path keeps its existing
command-buffer split. The predictor's metadata and 128 KiB Metal buffer
are charged to the same budget.

Each speculative slice has an independent completion group. A demand miss
waits only for the matching slice. If the previous job still owns a slot
when new speculation would start, the new speculation is skipped. Graph
completion drains outstanding jobs before releasing or reusing storage.
The previous CPU predictor and batch-read behavior remain the default.

On the same Flash-Next 246-input / 64-output request, 18 GiB, context 1024,
and microbatch 8, the sequential CPU / Metal / CPU comparison measured
14.36 / 12.08 / 13.92 tokens/s. The first token arrived in 18.59 / 18.45 /
18.70 s. All 64 logit rows, output IDs, serialized target and draft slot,
slot restore/re-save and seven rollback rounds matched. Budget accounting
and the selected CPU and Metal tests passed. The OS file cache was
uncontrolled.

CPU prediction and candidate merge took 374–376 ms on the CPU controls.
The Metal path used 13 ms on the CPU merge and 268 ms preparing the
prediction and waiting for or reading the prefix. Speculative I/O waits
fell from 12–20 ms to 1.1 ms across six matching slice waits. Useful payload
remained about 1.14 GB and unused payload about 0.81 GB. The sum of GPU
command-buffer service in the MoE path increased from 0.63 to 1.24 s.
This includes the new kernels and effects of the split, which were not
isolated from each other. This implementation reduced I/O wait without
improving decode throughput.

#### Prediction in the target Metal graph

Set `LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD_GRAPH=1` together with
`LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD=1` to select target-graph prediction for
qualified Qwen4Exp Metal decode batches. This option takes precedence
over the experimental inner-graph Metal predictor. The target scheduler
places the next-router projection, top-k and selected scores on the
backend that can read the router weights. The compact IDs and scores are
ordinary inputs to the existing CPU MoE callback. The callback launches
bounded reads while the final expert group computes, with independent
slice completion. The inner MoE graph uses its original command boundary.

Managed prompt graphs omit predictor nodes. The graph reuse key separates
decode graphs with predictor nodes from other graph topologies. The
scheduler owns and charges the new tensors. Nonqualified router
configurations retain CPU prediction.

On Flash-Next with 246 input / 64 output tokens, 18 GiB, context 1024 and
microbatch 8, the CPU / target-graph / CPU sequence measured 13.86 /
14.38 / 14.15 tokens/s. The scheduler retained 98 splits per target
decode pass in all three runs. All 141 predictor operations per pass
were assigned to MTL0, with 1363 predictions consumed by the CPU callback
over the request. CPU prediction/ranking fell from 376–382 ms to 12 ms
of host candidate handling. This does not include the predictor's outer
GPU work. Prefetch waits fell from 13–19 ms to 1 ms, and logical useful
and unused prefetch bytes matched the CPU controls exactly.

All 64 logit rows, output tokens, the target-plus-draft slot, the
restore/re-save round trip, and seven rollbacks covering 14 positions
matched. All three runs respected the same cap. The focused Metal
regression case passed. This is a short sequential measurement with
uncontrolled OS file cache.

#### Shared demand queue for Metal decode

Set `LLAMA_EXPERIMENTAL_MOE_DEMAND_QUEUE=1` with the combined Qwen4Exp
Metal MoE pipeline on ordinary GGUF shards. The queue admits one wave of
up to sixteen experts in the existing projection and expert order.
Four workers share the demand queue. Each worker takes an expert and
reads its missing up, gate and down slices into pinned destinations.
A resident group can execute while the next wave reads. After the GPU
fence and readback, the caller claims all completed triples. The default
minimum is four ready experts. Set
`LLAMA_EXPERIMENTAL_MOE_DEMAND_MIN_READY=8` to compare a larger group.
The threshold is clamped to the number of experts remaining in the wave.

Queued misses are masked from other consumers until their triple is
validated and claimed. The queue drains before a worker destination is
released. Failed or unclaimed misses are retired. The GPU completion
fence still protects weights referenced by submitted graphs. The option
uses the existing two-wave budget bound and the original FP32 operations
and scatter positions. The ordinary path and the inner-graph Metal
predictor keep their existing loading behavior.

On Flash-Next at 246 input / 64 output tokens, 18 GiB, context 1024 and
microbatch 8, control / queue / control measured 14.28 / 15.67 / 14.67
tokens/s. The outer Metal lookahead and adaptive DFlash2 were enabled for
all runs. Logical expert reads including prefetch matched at 9120025600
bytes. The queue produced 2331 ready groups from 1433 waves, including
898 groups that left other experts in the same wave outstanding. Inner
submissions increased from 2825 to 3723. The OS file cache was uncontrolled.

All committed logits and output tokens, the target-plus-draft slot,
restore/re-save and seven rollbacks covering fourteen positions matched.
Both CPU fixtures and the targeted Metal pipeline case passed. Measured
peak engine memory stayed within the same cap in all runs.

A follow-up 4 / 8 / 4 / 8 comparison at the same budget, prompt and
context measured 15.95 / 15.73 / 14.38 / 15.68 tokens/s. An eight-expert
minimum reduced inner submissions from 3728–3731 to 3177. Blocking queue
wait was 1.05–1.18 s at four and 1.13–1.14 s at eight. The throughput
ranges overlap, so four remains the default. All committed logits,
tokens, slots with restore/re-save, and rollback positions matched.

#### Reusable MoE layer plan

Set `LLAMA_EXPERIMENTAL_MOE_REUSE_PLAN=1` on the shared demand queue path.
Within one layer evaluation, the plan keeps the FP32 operation tensors
and their scratch bindings at the maximum group size seen so far. Each
group binds only pinned, ready weights. An unchanged activation in a
scratch lane is reused across consecutive uses. Each submission still
builds the active graph dependencies for Metal fusion and encodes a new
set of Metal command buffers. The caller waits for GPU completion and
copies the output to its original route before reusing scratch or
releasing the leases. The plan uses the existing scratch reservation
and does not change the ready threshold.

With the minimum at four, the same Flash-Next run measured
16.11 / 16.29 / 16.41 / 16.04 tokens/s in off/on/off/on order.
Decode graph setup shrank from 46–48 ms to about 40 ms, with 10,964
route lanes reused per run. Inner Metal submissions stayed near 3,724,
and blocking demand wait ranged from 0.96 to 1.01 s. There was no
repeatable decode speedup in this sample. All 64 logit rows, output
tokens, slot save/restore, and seven rejected-draft rollback events
matched. The OS file cache was uncontrolled.
