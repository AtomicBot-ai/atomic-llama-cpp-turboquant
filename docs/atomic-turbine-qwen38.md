# Qwen3.8 Flash Next family support

## Scope and order

Qwen3.8 Flash Next is a required family in the support roadmap. Qualify
the local GSQ-RCO IQ3_XXS checkpoint first on Metal, then extend the
capability matrix using actual architecture, quantization and state
contracts. A checkpoint result does not qualify every family variant.
Metal is the primary platform for this family; CUDA is secondary.
The initial Metal gates preceded the now-qualified
[single-slot LFM2.5 CUDA server profile](atomic-turbine-cuda-server.md).
Qwen3.8 CUDA now has a scoped short-request backend qualification.
CUDA PLE row reads and the seven routed formats now pass the scoped
[operation checks](atomic-turbine-qwen38-cuda.md) at microbatches one and eight.
With token embeddings also streamed, the 4 GiB RAM / 2 GiB VRAM profile
passes a five-token prefill and four greedy outputs at 22 GPU layers.
All captured logits and serialized state match ordinary CUDA byte for byte;
the 2 GiB runtime envelope is unchanged. See the
[token-row result](atomic-turbine-qwen38-cuda-token-rows.md).
The [3,481-token CUDA control](atomic-turbine-qwen38-cuda-long-prefill.md)
now fits 4/2 GiB at context 4,096 and matches ordinary CUDA logits, output IDs
and complete state byte for byte. A gated copy of routing weights prevents
allocation reuse from disabling weighted-reduction fusion; ordinary execution
is unchanged. The subsequent [CUDA resize control](atomic-turbine-qwen38-cuda-resize.md)
passes synchronized shrink/grow and cooperative worker-read cancellation at
context 2048, with actual cache release and exact logits/state. Context 4096
safely rejects resize with the unchanged 64 MiB transient allowance and
remains a fixed-budget profile. The [CUDA server lifecycle](atomic-turbine-qwen38-cuda-server.md)
now passes short raw completion, stop strings, SSE cancellation and file-based
slot restore at context 4096 with exact matching-history state. File staging
is admitted by the shared ledger. The selected
[non-thinking chat/EOS control](atomic-turbine-qwen38-cuda-chat.md) also passes:
16 input tokens, `OK` and natural EOS, with identical template, output IDs
and full saved state. The selected [two-turn chat](atomic-turbine-qwen38-cuda-chat.md#two-chat-turns-in-the-same-slot)
reuses 17 live tokens on the second turn with exact full logits and slot files.
The selected [long HTTP prefill](atomic-turbine-qwen38-cuda-http-prefill.md)
also passes at 4/2 GiB: 3,481 tokens across 28 observed server batches,
identical output IDs and complete slot bytes, with the reserve unchanged.
The selected [long-slot restore and exact-prefix continuation](atomic-turbine-qwen38-cuda-http-restore.md)
also passes in fresh servers: 3,484 cached tokens reused, one newly evaluated
input token and complete slot bytes identical to ordinary CUDA. The
[same-slot long series](atomic-turbine-qwen38-cuda-http-series.md) passes the
exact repeat and a one-token edit through full re-evaluation, with matching
output IDs and complete serialized state. The [opt-in disk prefix profile](atomic-turbine-release.md#cuda-disk-prefix-checkpoints-opt-in)
is qualified at 4/2 GiB. The exact repeat reuses 3456 tokens and the selected
edit reuses 1664. Faulted checkpoint files trigger full prefill in the same
slot. Captured logits and complete slots match ordinary CUDA in these checks.
The selected [long non-thinking chat](atomic-turbine-qwen38-cuda-chat.md#long-two-turn-chat-over-sse)
also passes at 4/2 GiB: the follow-up reuses 3473 live tokens, evaluates 20
and reaches 3.731 s TTFT. Full logits and both slot files match ordinary
CUDA byte for byte. The selected
[two-turn thinking dialogue](atomic-turbine-qwen38-cuda-chat.md#two-turns-with-thinking-and-preserved-history)
preserves reasoning and reuses 93 of 122 tokens, with bitwise logits and
slot files at 4/2 GiB. The selected [single tool call and result](atomic-turbine-qwen38-cuda-chat.md#single-tool-call-and-result-in-the-same-slot)
reuses 322 of 349 tokens at the same caps, with exact logits and both slot
files. Parallel tool calls and longer thinking conversations remain open.
Other prompts/checkpoints and broader contexts remain unqualified;
microbatches above eight are rejected.

The single-slot Metal profile now also has a scoped context-4096 check:
a 3,481-token prompt, a full repeat, and a short follow-up at 8 GiB.
All 48 captured output-logit rows and three serialized states match
ordinary Metal byte for byte. See the longer-context series below for
timings and limits; this does not qualify every context-4096 workload.

The current Metal optimization gate now also passes compact layer-window
prefill, full-fixture chaining/tail, exact server logits/slots and synchronized
manual resize fallback. The feature is explicitly opt-in; the paired long TTFT
improvement is about 25%, with a small decode tradeoff. See
[the opt-in window result](#opt-in-full-prefill-layer-windows). The scoped P11
windowed HTTP lifecycle and two additional long prompts now pass at 8 GiB.
See the windowed lifecycle evidence below. Post-window read advice improves
the checked server decode by 12.7%. Buffer refill is a smaller measured cost.
Synthetic embedding repeatability remains outside the qualified text scope.

1. Inspect both original GGUF shards and source slices. Identify resident
   weights, routed formats, router semantics and hybrid state.
2. Connect original expert bytes to the existing bounded executor on Metal.
   Preserve upstream hyper-connections, QSA, PLE and recurrent state.
3. Establish a short full-logit and state comparison against ordinary
   execution where the hardware can fit the control. Record any limitation.
4. Qualify useful prefill, manual shrink/grow, rejected budgets and state
   restoration under an explicit total engine budget.
5. Reduce the PLE residency floor through separately admitted row reads.
   Validate original row bytes, row lifetime and numerical equivalence
   against the full-PLE streaming control before enabling it.
6. Qualify the single-slot text server, then CUDA using separate RAM/VRAM
   limits. DFlash2 drafting and the vision projector have separate gates.

## Initial checkpoint inspection

The two local shards contain 75,839,998,528 file bytes and declare
`general.architecture=qwen4exp`. The model has 48 layers, width 2,560,
512 experts with ten selected, shared experts, four hyper-connection
streams, mixed recurrent/QSA attention and a PLE module at layer one.
Upstream already has a model builder for these operations.

The 144 routed tensors occupy 42,912,972,800 bytes. Their mixed types
are Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL.
The largest expert slice is 921,600 bytes. Remaining tensor payload
occupies 32,916,001,280 bytes, including a 28.8 GB PLE table.
Expert streaming alone therefore requires a substantially higher
budget than the earlier 5 GiB Qwen profile. Start with a 36 GiB
engine budget and context 128, then use measured admission results.

Metadata and tensor extents were read from the local files. Manifest
checksums have not been independently verified.

## Initial Metal evidence

The fork admits `qwen4exp` through its existing streaming gate. Seven
additional expert formats are admitted only for this architecture on
Metal; backend operation support is still checked before execution.
The upstream model builder and quantized kernels provide the computation.
This family initially used the synchronous bounded executor. The
vector-prefill overlap qualification below now enables bounded demand
read/GPU overlap by default. Prepared expert maps and preview retain
their prior profile gates.

The 16 MiB store probe read first and last experts for every routed tensor:
288 misses, 167,628,800 logical bytes, digest `e029defa00304333`.
It peaked at 16,773,616 managed bytes and returned the store to zero.
Metal allocation dropped from 15,286,272 to 475,136 bytes at shrink.

With context 128, microbatch one and a 36 GiB engine budget, the five-token
prompt `The capital of France is` produced IDs `[11751, 13]`
(` Paris.`). The peak was 38,366,609,824 of 38,654,705,664 managed
bytes. This establishes full streamed execution through the real model.

A 33-token prefill at microbatch eight selected four tokens
`[271, 248068, 198, 760]`. Fixed-cap and live-cap streaming matched
all 3,973,120 captured logit bytes and state digest `f4b73d884f1f4807`.
The live run requested 34 GiB inside prefill, rejected 1 MiB after
the first output token, grew to 36 GiB after the second, then shrank
to 34 GiB after the third. The last shrink released 499,580,928 bytes
of Metal allocations. Grow allocated nothing at the boundary. Peak
managed use was 38,654,704,672 bytes, below the initial 36 GiB limit.
The observed live floor was 35,208,418,368 bytes; this is profile-specific.

The existing rollback diagnostic saved 119,055,260 bytes of hybrid
target state after the same prefill, evaluated rejected token zero,
then restored the snapshot byte for byte. Continuing with the accepted
token reproduced all four fixed-control logit rows, IDs and final state
digest. The snapshot was admitted through the shared ledger; managed
high-water reached exactly 38,654,705,664 bytes. Reproduce by adding
`--rollback-probe` to the fixed command with separate output paths.

The matched ordinary fully offloaded Metal control failed during prefill
with `kIOGPUCommandBufferCallbackErrorOutOfMemory` on the 48 GiB M4 Max.
This initially left ordinary numerical equivalence open; the
host-expert Metal control below resolves the checked profile. The
fixed/live comparison validates resize invariance within streaming.
These short requests establish neither output quality nor throughput.

## Bounded PLE row reads on Metal

Budgeted Qwen4Exp on one Metal device now keeps the PLE tensor as
metadata and reads its original IQ4_NL rows through the pinned GGUF source.
The synchronous row workspace is 90 bytes, admitted in the I/O category
before allocation and retained until executor destruction. Each row is
dequantized using the existing CPU type conversion into the scheduler-owned
output tensor. The graph's output allocation remains covered by the shared
scratch observer. File identity checks bracket each read; an I/O failure
invalidates the source and rejects the context's evaluation.

The source fixture checks exact packed bytes on the second shard,
destination guards, invalid row arguments and changed-file rejection
(one selected case, 12 assertions).

Only the budgeted Qwen4Exp Metal loader selects this path. The localized
model graph hook substitutes the gather for the metadata-only PLE tensor.
The ordinary graph still builds the upstream GET_ROWS operation. There
is one additional graph-construction check at the PLE gather boundary;
no row workspace or file reads are introduced in ordinary execution.
The fork-owned executor implements the read policy. Its CPU conversion
uses the existing ggml type-traits contract; upstream upgrades must retain
IQ4_NL row decoding equivalence with CPU GET_ROWS.

On the same 33-token, microbatch-eight request, the 8 GiB run matches
all 3,973,120 logit bytes, four output IDs and final state digest from
the saved 36 GiB full-PLE streaming control. Managed high-water was
8,589,934,570 bytes against 8,589,934,592; measured peak physical
footprint was 7,477,680,152 bytes. The observed live engine floor was
6,408,230,042 bytes. Resident model payload charges dropped from
32,916,111,360 to 4,115,922,944 bytes. These measurements qualify the
checked text profile. The ordinary-control follow-up below extends the evidence.

The 8 → 7 → 8 → 7 GiB live run requested its first shrink during
prefill and rejected a 1 MiB target after the first selected token.
All captured logits, IDs and final state matched the saved full-PLE
control. The final shrink released 554,237,952 Metal bytes; grow
allocated nothing at the boundary. Managed high-water was
8,589,934,442 bytes. A separate rollback run admitted a 119,055,260-byte
state snapshot, evaluated token zero, restored state byte for byte and
continued with matching output. Its drafter charge returned to zero,
with high-water 8,589,934,570 bytes. These were selected checks for
the new row-read path, without a broad test suite.

## Reproduction and next gates

Set `MODEL` to the first shard; the loader discovers the second.
Use the same binary and run each command sequentially.

```sh
prompt="Here is a short geography quiz about Europe. Answer the next question in one sentence, with no explanation. What is the capital of France? The capital of France is"
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 8192 \
  --ctx 128 --ubatch 8 --gpu-layers 99 --tokens 4 --prompt "$prompt" \
  --state-digest --output fixed.json --logits fixed.bin
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 8192 \
  --ctx 128 --ubatch 8 --gpu-layers 99 --tokens 4 --prompt "$prompt" \
  --state-digest --prefill-budget-mib 7168 \
  --resize-at 1:1 --resize-at 2:8192 --resize-at 3:7168 \
  --output live.json --logits live.bin
cmp fixed.bin live.bin
```

The family currently has experimental Metal text-streaming support.
The short single-slot HTTP scenarios and the ordinary Metal numerical
control below have passed. CUDA, DFlash2, vision, longer contexts and
other family variants remain separate gates.

## Single-slot Metal HTTP profile

The same local checkpoint passed focused server checks at an 8 GiB
engine budget on M4 Max. The requested context is 128; the server
reports an effective slot context of 256 after alignment. Batch is
128, microbatch eight, one slot and eight CPU threads. The profile
uses greedy text completion, fixed memory, zero RAM prompt cache
and zero context checkpoints.

```sh
mkdir -p "$SLOT_DIR"
build-metal/bin/llama-server -m "$MODEL" -c 128 -b 128 -ub 8 -np 1 \
  -t 8 -tb 8 -ngl 99 -fit off --engine-budget-mib 8192 \
  --cache-ram 0 --ctx-checkpoints 0 --no-warmup --metrics --slots \
  --slot-save-path "$SLOT_DIR" --host 127.0.0.1 --port 18138
```

Set `MODEL` to the first GGUF shard and `SLOT_DIR` to a writable
directory before running the command.

- `/completion` with prompt `The capital of France is`, temperature
  zero, `n_predict=2`, `return_tokens=true`, `cache_prompt=true`
  and `id_slot=0` returned ` Paris.`, IDs `[11751, 13]` and
  `stop_type=limit`. The repeated request returned the same result.
  Both processed the five-token prompt with zero cached tokens.
- Save, erase, restore and a second save on `/slots/0?action=...`
  reported six slot tokens. Both files were byte-identical at
  118,224,560 bytes, SHA-256
  `63c250ac60ee3a3bbd358f213cd1b8657d7b916a7d416a2c8a787348c71bdccd`.
  The subsequent completion reproduced the original answer.
- Adding `stop=["Paris"]` returned a single space, one generated
  token (11751), `stop_type=word` and `stopping_word=Paris`.
  The matched stop text was absent from returned content.
- An SSE request with `n_predict=64` and `ignore_eos=true` was
  disconnected after its first output event. The server recorded
  cancellation and released slot zero with seven prompt-state tokens.
  The next request reproduced ` Paris.` and its original IDs.
- A closed-thinking raw chat prompt asking `Say only OK.` returned
  `OK`, IDs `[3793, 248046]`, two generated tokens and
  `stop_type=eos`. The EOS token is included in token accounting.
- `/v1/chat/completions` with a user message `Say only OK.`,
  temperature zero, `max_tokens=16` and
  `chat_template_kwargs={"enable_thinking":false}` returned `OK`
  and `finish_reason=stop`. Usage was 16 prompt tokens and two
  completion tokens, matching the explicit raw-template request.

After these raw completion checks, metrics reported 8,588,340,218 used
bytes against 8,589,934,592 applied bytes, with policy mode zero. This
is an idle snapshot; the earlier engine diagnostics establish their
own lifetime peaks. Repeating startup at 5 GiB rejected model buffer
admission before HTTP readiness. The temporary server exited cleanly
after the checks.

These checks establish the listed HTTP lifecycle behavior. The ordinary
Metal numerical control below applies to the checked engine request.
Live resize remains an embedding
API capability; the fixed HTTP profile does not expose it.

## Ordinary Metal numerical control without full GPU residency

The upstream host-expert MoE cache provides a feasible ordinary control.
The diagnostic runner's `--native-moe-cache-mib 8192` selects CPU
storage for expert tensors, disables CPU repacking and configures the
existing context MoE cache with zero CPU share. These are upstream model
and context parameters, equivalent to the common options `--cpu-moe
--no-repack --moe-cache-mib 8192 --moe-cache-cpu-share 0`.
Expert matmuls still execute on Metal. Target router, PLE gather,
hybrid state and all other operations use the ordinary graph. Both
fork streaming budgets are zero in this control.

The original all-GPU mmap control exceeded available residency.
Disabling residency sets with the existing `GGML_METAL_NO_RESIDENCY=1`
setting also failed with Metal out-of-memory. A 2 GiB host-expert cache
completed, but its smallest groups had only 24 slots. At microbatch
eight and Top-10, upstream requires capacity for up to 80 selected
experts to use the cached path. The smaller control therefore changes
execution paths and does not establish the intended numerical baseline.
CPU overrides can also select CPU_REPACK, whereas cache fills copy
host payload directly; the final control explicitly preserves GGUF
packing.

With an 8 GiB upstream cache, all 48 layers belonged to ten cache groups,
each with at least 101 slots. The completed request reported 4,521 hits
and 4,964 misses, zero CPU misses and 8,300.88 MiB uploaded. It completed
the same 33-token prefill and four-token output at context 128,
microbatch eight and 99 requested GPU layers.

All 3,973,120 captured FP32 logit bytes matched the saved full-PLE
36 GiB streaming control and the bounded-PLE 8 GiB fixed, live-resize
and rollback runs. There were zero differing FP32 bit patterns and zero
maximum absolute logit error. IDs were `[271, 248068, 198, 760]`;
every final serialized-state digest was `f4b73d884f1f4807`.
State equality here is established by the recorded digest; the logit
comparison directly checks every byte.
The logit file SHA-256 is
`6a6b93404edab20fbb97536d95a36ecef1a3192296f04c5c0441addfb193f3e5`.

The native control reported 12,558,647,296 allocated Metal bytes and
9,199,634,040 peak physical-footprint bytes. Its 8 GiB option limits
only the upstream expert cache. Our 8 GiB limit covers the total
managed engine budget, so this comparison establishes numerical
equivalence, without an equal-memory performance claim.

Reproduce the ordinary side using the same `MODEL` and `prompt`
as the streamed commands above:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --native-moe-cache-mib 8192 --ctx 128 --ubatch 8 --gpu-layers 99 \
  --tokens 4 --prompt "$prompt" --state-digest \
  --output native.json --logits native.bin
cmp native.bin fixed.bin
cmp native.bin live.bin
```

The changes for this control are confined to the fork-owned diagnostic
runner. Larger microbatches require their own cache-capacity and
numerical checks. The all-GPU residency configuration remains outside
the available hardware envelope.

## Vector prefill read/GPU overlap

The 8 GiB, context-128, microbatch-eight profile above spent 3.5725 s
per request with phase timers enabled. Expert acquisition took 1.9156 s,
including 1.0959 s reading original bytes. GPU submission wait took
1.1338 s while recorded GPU command duration was 0.2080 s. The existing
overlap switch initially performed zero ahead acquisitions because it
only admitted matrix waves of at least 32 tokens.

Qwen4Exp now admits bounded reads while its grouped vector graph is
executing. It uses the existing maximum of four subsequent, already
routed experts. Current and ahead leases retain their buffers until
GPU completion and consumption; low headroom reduces the ahead count
to zero. It introduces no worker, predictor or numerical kernel.
The family gate selects vector batches; existing LFM/Qwen3.5 gates
continue selecting matrix waves. One-token decode has no added ahead
window. This lives in the fork-owned executor and is enabled by default
for Qwen4Exp on Metal. `--expert-read-overlap off` in the diagnostic
runner selects the original control.

With timers enabled, the request fell to 2.7861 s and GPU wait to
0.3729 s. There were 20,439 ahead acquisitions covering 11,950,054,400
read bytes during submitted work. The baseline and overlap runs had
identical 26,100 misses, 2,355 hits, 15,235,187,200 logical read bytes,
7,717 buffer allocations and 7,617 submissions. Timing counters nest:
compute duration includes overlapping reads and must not be added to
read duration to reconstruct total request time.

Four sequential requests without phase timers used OFF/ON/ON/OFF order.
Request times were 3.5190, 2.8016, 2.7662 and 3.5266 s; paired
reductions were 20.4% and 21.6%. Prefill times were 3.0537, 2.3376,
2.3039 and 3.0592 s. No compilation or other test GPU workload overlapped
these measurements. Page-cache state was uncontrolled and warm; this
establishes the improvement on this short request, not cold SSD or
long-context performance.

All four requests and the profiled candidate matched the ordinary Metal
control's 3,973,120 logit bytes, token IDs and serialized-state digest.
Fixed-cap high-water remained 8,589,934,570 bytes. The final default
also passed the 8 → 7 → 8 → 7 GiB prefill/decode sequence with a rejected
1 MiB target. It matched the same outputs and state, peaked at
8,589,934,442 bytes, and released 554,237,952 Metal bytes at the final
shrink. Grow allocated no additional Metal buffer at its boundary.

Two selected Metal executor cases passed 24 assertions. They cover cold
and hot routed results, no optional reads at the single-expert floor,
and source failure after GPU submission with completed synchronization,
zero surviving pins and returned payload charges. The selection was
`test-expert-executor --backend MTL0 'Qwen4Exp vector overlap.*'`.
The inference runner and server were rebuilt with the final default.

Reproduce phase profiling with `--profile-experts` and explicit
`--expert-read-overlap off` or `on` on the fixed 8 GiB command above.
Omit phase timers for request-latency comparisons. Remaining measured
costs include acquisition/admission and the source reads themselves.

## Cache admission and remaining read costs

With vector overlap already enabled, a focused profile measured
2.7629 s/request, 0.5093 s in cache admission and 1.0740 s in expert
reads. Added opt-in counters attributed 0.3165 s and 63,956,794
visited entries to victim selection. File transfer accounted for
1.0392 s; seek and source validation took 0.0065 and 0.0217 s.

The fork-owned store now maintains an intrusive LRU queue in acquisition
order. Hits move entries to the tail; completed new entries are appended
only after publication. Victim selection skips pinned entries from the
head. Release does not promote recency. Eviction, trimming and final
retirement unlink entries before destruction or reuse. ClassLRU retains
its existing class/age selection. No upstream cache or backend algorithm
is changed.

The queue adds two non-owning pointers per resident C++ entry and two
store head/tail pointers. These control objects are covered by the
existing 2 GiB runtime envelope; payload, tensor-arena and staging
charges are unchanged. It adds no independently allocated index or
retained pool. The explicit managed cap continues to include the same
runtime reservation.

On the same profiled request, victim selection fell to 0.0033 s and
23,743 inspected entries. Admission fell to 0.1919 s, while reads
remained 1.0630 s. Request time was 2.3442 s. Every reported store
counter matched the earlier control, including hits, misses, evictions,
logical bytes, allocations, reuses, submissions, floor and high-water.
Thus the improvement comes from finding the same victims more cheaply.

Two sequential unprofiled pairs used BEFORE/AFTER/AFTER/BEFORE order,
with the pre-change executable retained for the controls. Request times
were 2.7685, 2.3759, 2.3862 and 2.7681 s, giving paired reductions
of 14.2% and 13.8%. Prefill times were 2.3032, 1.9648, 1.9892 and
2.3037 s. Builds and GPU runs were serialized. File-cache state was
uncontrolled and warm; these are short-profile measurements.

A separate existing ClassLRU control reduced logical reads from
15,235,187,200 to 14,663,168,000 bytes and increased hits from 2,355
to 3,375. Its victim search cost 0.3772 s and the profiled request
took 2.7908 s, exceeding the optimized LRU's 2.3442 s. ClassLRU
remains opt-in. This check does not justify changing the default
eviction policy or altering the source's validation/packing.

All fixed-profile comparisons, ClassLRU and the final live-resize run
matched the ordinary Metal control's 3,973,120 logit bytes, four token
IDs and state digest `f4b73d884f1f4807`. The final default passed
8 → 7 → 8 → 7 GiB with the intervening 1 MiB rejection. Its peak
was 8,589,934,442 bytes; the final shrink released 554,237,952 Metal
bytes, and grow allocated none at the boundary.

Five selected store cases passed 37 assertions, covering recency across
out-of-order releases and hits, pinned eviction, regrowth, ClassLRU,
failed-read rollback and destruction deferred until backend completion.
No aggregate test suite ran. Both inference runner and server were
rebuilt. `--profile-experts` now exports `eviction_search` in nanoseconds
and `eviction_candidates` as a count. Existing source subphase counters
continue to distinguish transfer, seek and validation.

## Uniform Metal expert slots

A focused positional-read experiment replaced expert `fseek`/`fread`
with `pread`, retaining the same validation and original bytes. It
removed approximately 6.6 ms of seek time but did not improve the
profiled request: 2.4090 s before, 2.4918 s after. Both outputs were
exact. The experiment was removed; the source read implementation is
unchanged. A single comparison is not a general verdict on positional
I/O, but it provides no reason to adopt this change.

The remaining profile attributed 0.3018 s to buffer allocation/setup
and 0.1836 s to admission. Mixed expert sizes caused 7,717 physical
allocations for 26,100 misses: the LRU victim was often too small for
the next payload. The Qwen4Exp Metal executor now selects a common
power-of-two slot size covering its largest expert, starting at 64 KiB
and capped at 1 MiB. This model's maximum payload is 921,600 bytes,
so it uses 1 MiB slots. Models with a larger maximum retain generic
64 KiB rounding. Other architectures and backends retain their policy.
This is a fork-owned executor change using the store's existing
allocation-granularity parameter, without an upstream hook or new pool.

The entire physical slot remains charged, including padding. Admission,
lookahead capacity and the minimum working set use the same rounding.
The reported minimum increases by 65,536 bytes to 6,408,295,578 bytes.
Pins, retirement, original expert bytes and routed arithmetic are
unchanged. Reuse still selects the same policy's unpinned victim; it
now fits regardless of the preceding expert's quantization in the
qualified geometry. No ordinary-path allocation or dispatch is added.

The profiled candidate took 2.1350 s. Physical allocations fell from
7,717 to 2,216, allocation/setup from 0.3018 to 0.1670 s, and admission
from 0.1836 to 0.0195 s. Read time was 1.0071 s versus 1.0607 s.
Larger slots hold fewer entries: hits decreased from 2,355 to 2,091,
misses increased to 26,364, and logical reads increased from
15,235,187,200 to 15,398,156,800 bytes (1.07%). GPU submissions remained
7,617. This tradeoff is accepted because request time improved; the
read-time difference alone is not evidence of faster storage.

Two unprofiled pairs in BEFORE/AFTER/AFTER/BEFORE order measured
2.3726, 2.1282, 2.1219 and 2.3369 s/request. Paired reductions were
10.3% and 9.2%; prefill took 1.9422, 1.6940, 1.6930 and 1.9152 s.
Builds and GPU runs were serialized. These use the same 33-token
prompt, four generated tokens, context 128 and microbatch 8 described
above. The filesystem cache was uncontrolled and warm. Cold storage,
other quantizations and long-context cache tradeoffs remain unqualified.

All four timing runs and the profiled candidate matched all 3,973,120
ordinary-control logit bytes, tokens `[271, 248068, 198, 760]` and
state digest `f4b73d884f1f4807`. Peak managed use was 8,589,881,498 bytes,
below the 8,589,934,592-byte cap. The final source also passed the
8 → 7 → 8 → 7 GiB prefill/decode sequence with a rejected 1 MiB
request and identical logits/state. The final shrink released
1,006,632,960 Metal bytes; grow allocated none at its boundary.
Every observed applied-budget snapshot remained within its cap.

Two selected Metal tests passed 27 assertions: `uniform slots reuse
small and large experts within physical cap` and `Qwen4Exp vector
overlap cold hot and tight budget`. They cover reuse in both size
directions, padding clearing, pinned retirement, full physical charging,
regrowth, exact routed outputs and no optional reads at the working-set
floor. The inference runner and server were rebuilt. No aggregate suite
or unrelated ordinary-mode tests ran; the ordinary execution path has
no changes from this optimization.

## Reusing the zeroed Metal tail

A focused profile of the uniform-slot version measured 12,246,500,864
bytes of direct-read padding clears and 0.0881 s in those clears,
within 0.1667 s of allocation/setup. Every miss cleared the complete
unused tail even when most of it was already zero.

For Qwen4Exp with Metal's host-buffer capability, the store now saves
the previous payload length before rebinding an unpinned victim. A
fresh buffer still clears its entire tail. A reused buffer clears only
the part of the old payload beyond the new payload; the remainder was
already zero, and the complete source read overwrites the new payload.
Published weights are read-only. Failed or cancelled reads discard the
unpublished owner, and lease release still synchronizes before reuse.
Other architectures retain full-tail clearing; staged reads retain
full-buffer clearing. No upstream files or ordinary execution path were
changed, and no new payload allocation or unaccounted pool was added.

With profiling, cleared bytes fell to 3,414,793,216 and padding time to
0.0455 s. Allocation/setup fell to 0.1258 s. Profiled request times were
2.1221 s before and 2.0992 s after. All store counters, managed memory,
logical reads, physical allocations, reuse counts and GPU submissions
were identical. The diagnostic runner now reports `padding` in
nanoseconds and `padding_bytes` as a byte count under `expert_profile_ns`;
padding time is nested within allocation time, not additive to it.
Disabled profiling samples no additional timestamps.

Two unprofiled pairs in BEFORE/AFTER/AFTER/BEFORE order measured
2.1224, 2.0927, 2.0921 and 2.1241 s/request, reductions of 1.40% and
1.51%. Prefill times were 1.6856, 1.6664, 1.6597 and 1.6900 s.
Runs and builds were sequential. This is a modest whole-request gain
on the existing warm, short profile, not a storage throughput claim.
Both profiled requests and all four timing requests matched the saved
ordinary Metal control's complete logits, committed IDs and state
digest. Managed peak remained 8,589,881,498 bytes under 8 GiB.

The final implementation also passed 8 → 7 → 8 → 7 GiB with the
intervening rejected 1 MiB request, exact logits/state and bounded
applied-budget snapshots. The last shrink released 1,006,632,960 Metal
bytes; grow allocated none at its boundary.

Two selected Metal cases passed 67 assertions: `Qwen4Exp reused slots
preserve the complete zero tail` and `Qwen4Exp vector overlap read
failure drains leases`. The first checks every tail byte after fresh,
equal-size, larger and smaller payloads, compares original payload
bytes, checks the other-architecture fallback, and verifies failed-read
publication and accounting. The second checks failure after GPU work
has been submitted, synchronization and lease release. The runner and
server were rebuilt. No aggregate suite ran. Cold I/O and long-context
performance remain unqualified; the remaining profiled source-read
cost is about 1.00 s and GPU completion wait about 0.42 s, with nested
overlap timers that cannot be summed into request time.

## Eight-expert vector submissions

After tail-clear reuse, a focused profile still measured 7,617 expert
graph submissions, 0.1066 s submitting them and 0.4233 s waiting for
completion. The Qwen4Exp Metal executor now admits workspace for up to
eight expert groups and batches eight independent vector groups per
submission, including one-token decode. During vector prefill it may
acquire up to eight subsequent known-demand experts while GPU work is
pending. All counts remain constrained by available charged capacity.
No prediction or background worker is introduced.

Other Metal families and matrix-wave prefill retain four-group compute
limits. Early next-projection reads retain their four-entry limit. The
existing constructor admission loop halves workspace groups when the
budget cannot fit the larger workspace. Runtime group and ahead counts
still shrink with the applied budget, and optional reads remain absent
at the single-expert floor. The same original-weight vector operations,
row order and output scatter are used; only independent nodes share a
larger submission.

The qualified profile's minimum working set increases from
6,408,295,578 to 6,413,653,258 bytes (5,357,680 additional bytes, about
5.11 MiB). This covers the larger graph/tensor arena, Metal scratch and
host workspace in the shared ledger, reducing cache capacity accordingly.
Fixed callback arrays grow by roughly 4.3 KiB of stack storage in the
fork-owned executor, including for other streamed families; this is
covered by the existing runtime envelope, not a new allocation. Ordinary
llama.cpp never enters this callback. No upstream hook or algorithm was
changed; no performance claim is made for other streamed families.

The eight-group profile took 1.8801 s versus 2.1066 s for four groups.
Submissions fell to 4,230, submit time to 0.0744 s and completion wait
to 0.2795 s. The number of expert matmuls stayed 50,880. Hits changed
from 2,091 to 2,090; logical reads increased by 473,600 bytes to
15,398,630,400. Physical allocations increased from 2,216 to 2,235.
Known-demand reads inside submission windows decreased from 20,439 to
18,183 acquisitions (11,950,054,400 to 10,643,840,000 bytes), because
there are fewer, larger groups and more initial demand outside each
window. This is a submission/synchronization improvement, not evidence
of increased I/O overlap volume or faster storage.

Two sequential unprofiled pairs in BEFORE/AFTER/AFTER/BEFORE order
measured 2.0833, 1.8864, 1.8850 and 2.0994 s/request: reductions of
9.45% and 10.21%. Prefill times were 1.6538, 1.5258, 1.5254 and
1.6683 s. Builds and GPU runs were serialized; the existing short
33-token/four-output profile used an uncontrolled warm filesystem cache.
Cold I/O and long-context performance remain unqualified.

Both profiled runs and all four timing runs matched the ordinary Metal
control's complete logits, four token IDs and serialized-state digest.
Candidate managed high-water was 8,589,917,658 bytes under the
8,589,934,592-byte cap. The final source also passed the prefill/decode
8 → 7 → 8 → 7 GiB sequence with a rejected 1 MiB request, exact
outputs/state and bounded applied-budget snapshots. The final shrink
released 1,010,827,264 Metal bytes; grow allocated none at its boundary.

Three selected Metal executor cases passed 44 assertions:
`Qwen4Exp vector overlap cold hot and tight budget`, `Qwen4Exp vector
overlap read failure drains leases`, and `bounded read overlap preserves
cold hot and tight-budget outputs`. The Qwen fixture now exercises two
eight-expert groups, exact cold/hot results, the single-expert floor,
constructor fallback to smaller workspace and source failure after the
first group's submission. The other-family case checks the retained
four-group behavior. The inference runner and server were rebuilt;
no aggregate suite ran.

## Sixteen-expert vector submissions

Qwen4Exp Metal now batches up to sixteen vector groups per submission
and can acquire up to sixteen subsequent demand experts during vector
prefill. Both limits still follow the applied memory budget. The larger
workspace uses 10,715,360 additional bytes (10.22 MiB), raising this
profile's minimum to 6,424,368,618 bytes. Constructor admission can fall
back to eight or fewer groups. Other Metal families retain their previous
workspace and group limits. Qwen4Exp matrix waves use at most four groups
even when the admitted workspace is larger. The fork-owned callback uses
larger bounded stack arrays within the existing runtime envelope.

On the same 33-token prompt and four-output profile at 8 GiB, submissions
fell from 4,230 to 2,271 while the 50,880 expert matmuls were unchanged.
Profiled GPU completion wait fell from 268 to 206 ms. Logical reads rose
from 15,398,630,400 to 15,403,264,000 bytes. Two sequential unprofiled
pairs in BEFORE/AFTER/AFTER/BEFORE order measured 2.2009, 1.9874,
1.7623 and 2.1265 s per request, for paired reductions of 9.7% and
17.1%. Timings varied across runs with uncontrolled filesystem cache
state. These measurements cover this short request and do not establish
long-context or cold-storage performance.

All comparisons matched the saved ordinary Metal logits byte for byte,
the four token IDs and the serialized-state digest. Managed high-water
was 8,589,516,378 bytes. The 8 → 7 → 8 → 7 GiB prefill/decode run also
matched the control and stayed within every observed applied cap. It
rejected 1 MiB and released 1,015,021,568 Metal bytes on the final shrink.
The two Qwen4Exp overlap cases now exercise two sixteen-expert groups,
including tight-budget fallback and failed reads after submission. Those
cases and the four-group matrix overlap case passed 44 assertions on Metal.

## Long vector prefill at 8 GiB

The same checkpoint and M4 Max ran a 523-token geography prompt with
context 1024, microbatch 8, all GPU layers and four generated tokens.
The two streamers differed only in the eight- versus sixteen-expert
submission and lookahead limits. This tests many successive vector
microbatches. Metal's matrix path for microbatches of at least 32 keeps
the four-group compute limit and is outside this comparison.

Without phase timers, four sequential runs in 8/16/16/8 order timed
prefill at 37.630, 36.607, 34.474 and 35.557 s. The two paired reductions
were 2.72% and 3.05%. The process disk-read counter at the prefill
boundary ranged from 24.02 to 32.34 GB. Both sixteen-group runs saw
less process disk I/O than their paired controls, so these timings
describe the observed runs with uncontrolled filesystem cache state.
A separate profiled pair took 33.847 s with eight and 34.186 s with
sixteen, with 22.65 and 23.99 GB of process disk reads respectively.

Profiling isolates the structural change. Prefill submissions fell from
60,033 to 32,340 (46.1%) for the same 737,460 expert matmuls. Submission
time went from 1.120 to 0.945 s and Metal command time from 2.328 to
1.999 s. Completion wait rose from 1.558 to 1.760 s and the reads
inside submission windows fell from 216.22 to 173.12 GB. Those
bytes count reads issued between submission and synchronization. They
do not establish how long the reads and GPU execution actually overlapped.

Both builds read 259,348,531,200 logical expert bytes for 447,456 misses
and zero store hits during prefill. With sixteen groups, the 25.507 s
expert read timer included 25.030 s in data transfer (98.1%), 0.315 s
in source validation, 0.105 s in seek, and 0.043 s in setup. Transfer
includes file-cache copies and disk service. The process disk-read
counter is a separate measurement and includes all process reads.
The timers include nested phases and overlapping work. These results
point to reducing repeated payload reads across microbatches as the
next experiment. Seek and validation account for about 0.42 s together.

The sixteen-group build peaked at 8,589,915,610 bytes in the managed
ledger, below the 8,589,934,592-byte cap. The eight-group build peaked
at 8,589,920,090 bytes. All six streaming runs matched all 3,973,120
captured logit bytes across the four output positions, including the
first prediction after prefill, as well as committed IDs and the final
state digest. A separate ordinary Metal run using the 8 GiB upstream
cache also matched every captured logit byte, IDs `[271, 248068, 198, 760]`
and digest `993c85a8e86139bf`. The native cache reported zero CPU misses.
Its option caps the cache alone and this run served as a numerical
control. The logits SHA-256 is
`a7882d4b8ee304d3352d02fb31faa7ab4a01a1b1e3b8d38c986b341b84ede994`.

## Retaining frequent experts across microbatches

The 523-token, microbatch-eight profile above was captured as ATET3
at a fixed 8 GiB. Replay matched all 451,776 LRU acquisitions,
including each hit decision and physical buffer size. Of the 447,456
prefill acquisitions, 176,313 were for expert slices used in the
previous microbatch of the same tensor (102.21 GB of payload).

A separate holdout calculation picked up to twelve experts per tensor
by observed frequency times payload size in the first sixteen
microbatches. Those keys accounted for 26.16 GB, or 12.9%, of payload
demand in the remaining microbatches. This calculation identifies
reusable keys and assumes the selected slices are available. It does
not simulate admission, pins, or policy warmup. The runtime uses the
existing adaptive ClassLRU policy instead of a prompt-specific list.

The fixed-cap replay, including leases and 1,199,232 bytes of policy
history, predicted 23,750,041,600 fewer bytes (9.10%) across the whole
request. Runtime matched the prediction. Prefill hits rose from zero
to 41,117, and logical prefill reads fell from 259,348,531,200 to
235,655,334,400 bytes. For example, expert 466 in block 25 and expert
296 in block 26 each delivered 61 prefill hits on each of the gate,
up and down projections. These IDs are observations for this prompt.

The original ClassLRU scanned resident entries to find the oldest
unpinned entry in each of eight frequency classes. The store now uses
the existing intrusive links to keep acquisition order per class and
scores those same eight candidates. Pinned entries are skipped.
The class scores and tie order are preserved. Plain LRU uses class
zero. Queue endpoints add 112 bytes per store to the runtime envelope
on this platform. History, buffer and staging charges are unchanged.
There are no changes to the ordinary inference path.

The original scan visited 824,769,960 entries and spent 6.47 s in
prefill victim search. The queue version visited 2,563,313 and spent
0.081 s in the same timer. The latter run also captured a trace, so
its total latency includes trace overhead and is excluded from
the performance comparison. Replay matched every observed decision
and physical allocation on that ClassLRU trace. All runtime cache
counters matched the original ClassLRU run.

Two pairs without trace or timers, in LRU/ClassLRU/ClassLRU/LRU order
on the same final binary, measured prefill at 28.906, 26.706, 26.800
and 28.955 s. Paired reductions were 7.61% and 7.44%. The filesystem
cache was uncontrolled. Process disk reads were 12.12, 11.30, 11.42
and 12.85 GB at the prefill boundary. These results apply to this
prompt and checkpoint. This comparison used the explicit
`--cache-policy class-lru` control.

ClassLRU peaked at 8,589,592,298 bytes in the managed ledger. Every
Metal run matched the saved ordinary Metal logits byte for byte,
with matching token IDs and state digest. Two targeted Metal cases
passed for LRU order and ClassLRU retention/retirement. The
allocation/read rollback case uses a mock CPU buffer type that Metal
rejects during initialization. It passed on CPU. No aggregate suite
ran.

## Automatic ClassLRU for long Metal prefill

Two new prompts ran on the same checkpoint at a managed 8 GiB,
microbatch eight, context 1024 and four output tokens. A 333-token
narrative measured LRU/ClassLRU prefill pairs of 18.933/16.451 and
17.540/16.470 s, reductions of 13.11% and 6.10%. Logical prefill reads
fell from 142.78 to 131.32 GB (8.0%). A 327-token technical prompt
with code measured 20.500/20.167 and 20.525/19.279 s, reductions of
1.62% and 6.07%, with reads falling from 149.27 to 134.81 GB (9.7%).
Both sets used sequential, counterbalanced runs without phase timers
or tracing. Filesystem cache state varied across the runs.

The 33-token control used context 128. The initial LRU run incurred
2.65 GB of process disk reads. The second LRU/ClassLRU pair incurred
about 33/36 MB and measured 1.499/1.478 s of prefill. This short
comparison supports keeping LRU for short work given the limited
timing margin. Logits matched ordinary Metal byte for byte for all
three prompts. The token IDs and state digests also matched.

The automatic policy selects ClassLRU at the start of a Qwen4Exp
Metal decode batch containing at least 320 tokens, with microbatch
eight, active read overlap and a requested and applied managed budget
of exactly 8 GiB. The threshold is measured per submitted decode batch.
A caller that splits a long prompt into batches smaller than 320
tokens keeps LRU. The threshold sits below the shortest measured long
prompt (327 tokens). Explicit policy overrides, demand traces and
experimental preview/early reads retain their previous behavior.

The store admits temporary frequency history only for qualifying work.
It can reclaim unpinned slots under the existing ledger if the cache
is already full. Admission preserves the mandatory working floor and
falls back to LRU when the history cannot fit. The next short batch
(including token generation) or an accepted resize away from 8 GiB
restores LRU and returns the history charge. Merging the frequency
queues restores global acquisition order while keeping the resident
weights. Selection runs inside the existing managed-budget branch of
decode. Ordinary inference keeps its existing path.

The final automatic technical run matched the explicit ClassLRU
prefill counters, with 25,144 hits, 134,805,337,600 logical read bytes
and a managed peak of 8,589,592,298 bytes. It released the 1,199,232-byte
history charge when generation began. The final automatic short run
matched explicit LRU counters and memory exactly. Both automatic runs
matched all 3,973,120 saved logit bytes, committed IDs and state
digests of their ordinary Metal controls.

Three targeted Metal cases passed 23 assertions. They cover the length,
microbatch and memory gates, explicit override, valid and rejected
resize, resident-key ordering after a switch, re-admission with a full
cache and tight-cap fallback. The inference runner and server were
rebuilt.

## HTTP latency at three engine budgets

The M4 Max served the same raw 33-token prompt with one slot,
context 1024, batch 512, microbatch eight, eight CPU threads and
99 requested GPU layers. Prompt caching was disabled. One 8-output
request warmed each server, followed by two sequential 24-output
greedy requests with EOS ignored. TTFT is elapsed client time from
request start to the first nonempty text SSE event carrying a token.
Decode TPS counts the remaining 23 tokens and was cross-checked
against the server timing. Server startup is excluded.

- 8 GiB: TTFT 1.462 and 1.506 s. Decode 8.25 and 9.10 tokens/s.
- 16 GiB: TTFT 2.049 and 1.881 s. Decode 5.88 and 6.20 tokens/s.
- 32 GiB: TTFT 0.808 and 0.818 s. Decode 4.49 and 12.28 tokens/s.

This is a short-request snapshot with uncontrolled filesystem cache
and host activity. In particular, the two 32 GiB decode samples vary
substantially. The 33-token prompt selects LRU at every cap. This
snapshot precedes the server prefill hint below.

## Server prompts split across decode batches

The single-slot server ran the same 523-token prompt at 8 GiB,
context 1024, batch 256 and microbatch eight. Prompt progress
confirmed three slices of 256, 256 and 11 tokens. The original
per-call threshold missed all three. The managed server now supplies
the total uncached prompt length on each prompt slice through a
fork-owned decode call. The hint has call lifetime. It follows the
existing backend, microbatch and budget gates and does not change
tokenization, batch layout, sampling, or slot scheduling.

Server debug logs confirmed ClassLRU on all three slices, including
the final eleven tokens. A 33-token request immediately afterward
in the same slot selected LRU. Both requests used zero cached
prompt tokens. The server metrics now expose the managed high-water
mark, which was 8,589,915,610 bytes against an applied 8,589,934,592
after both requests. The idle minimum returned to the LRU floor of
6,449,468,906 bytes once generation began.

The production server's long and short slot files matched the
ordinary Metal control byte for byte (134,226,000 and 119,147,720
bytes respectively). In a separate temporary server build, a local
probe captured 7,946,240 bytes of pre-sampling FP32 logits across
the eight output positions. Every byte matched the ordinary Metal
control using the same HTTP requests and batch layout. All runs
returned IDs `[271, 248068, 198, 760]` for each request. The ordinary
control used the upstream host-expert cache and reported zero CPU
misses. The logit probe is outside the repository and production
build. The targeted Qwen4Exp policy test also passed ten assertions.

## Counterbalanced 8 and 16 GiB HTTP comparison

The 33-token HTTP profile above was repeated in 8/16/16/8 order.
Each fresh server received one 8-output warmup and one measured
24-output request. A temporary build outside the repository sampled
cumulative expert-store and macOS process disk-read counters immediately
before prompt processing, at the first sampled token, and at the
24th sampled token. Differences distinguish prefill and the 23 decode
steps. The four timing runs used counters with phase timers disabled.

TTFT was 1.453, 1.172, 1.179 and 1.462 s, respectively. Decode TPS
was 7.74, 6.19, 6.51 and 9.21. The larger cap improved TTFT in both
pairs and delivered slower decode. All 24 output token IDs matched
across runs. Client and server decode timings agreed.

Prefill used 233 hits out of 24,135 acquisitions at 8 GiB, and
12,928 hits at 16 GiB (1.0% and 53.6%). Logical expert reads were
13.950 and 6.544 GB respectively. Process disk-read deltas in run
order were 15.94, 37.54, 31.36 and 0 MB. Decode used 11,010 and
21,257 hits out of 33,120 acquisitions (33.2% and 64.2%), with
12.926 and 6.980 GB of logical expert reads. In run order, process
disk-read deltas were 0.999, 1.450, 1.045 and 0.014 GB. Logical
expert reads include filesystem-cache hits. The macOS counter is
for the whole process and is separate from logical expert traffic.

A diagnostic 16/8 pair instrumented the full resident accounting walk
at the start of each streamed projection. Decode took 3.643/2.503 s
at 16/8 GiB. Expert read time was 1.191/0.827 s, with 1.368/0.025 GB
of process disk reads. The resident walk took 0.349/0.073 s.
Admission took 0.256/0.059 s and buffer allocation/setup took
0.276/0.075 s. The victim search was only 0.004/0.009 s and submission
counts were the same. The slower decode combined more expensive
source reads with higher resident-cache and buffer costs. These
instrumented phases nest within projection time. Disk-cache state was
uncontrolled, so this one profiling pair does not fix the split
between filesystem effects and cache-accounting overhead for all runs.

Both caps were respected in every snapshot. The probes changed no
production source or ordinary execution path.

## Removing per-projection resident accounting walks

The managed executor previously called `store->stats().resident_bytes`
before each streamed projection. This enumerated every resident entry,
including its pin count, just to compute available group capacity. The
cost grew with cache size rather than with the experts in the current
projection. The store now maintains physical resident bytes under its
existing mutex and exposes a constant-time query. Successful publication
adds the complete buffer and tensor-arena charges; eviction, idle trim,
and the last retiring lease remove them. Unpublished reads remain in the
shared ledger, oversized reuse retains its full charge, and pinned retired
entries stay resident until synchronized release. Trace capacity uses the
same counter. Replacement order, group selection, arithmetic, and the
ordinary execution path are unchanged.

The same fresh-server HTTP workload was compared at 16 GiB in
before/after/after/before order, with phase timers disabled. TTFT was
1.188/1.101/1.115/1.197 s; decode TPS was 6.373/7.382/7.338/6.568.
Mean TPS improved from 6.470 to 7.360 (13.8%), and mean TTFT fell
from 1.193 to 1.108 s (7.1%). Prefill and decode hits, misses, logical
reads, evictions, allocations, reuses, and submission counts were
identical. Decode process disk reads were 1.420/0.778/0.856/0.764 GB,
so filesystem-cache state still contributes to the timing spread.

One instrumented candidate pair at 16/8 GiB measured only 0.095/0.088 ms
for all decode resident-size queries, versus 349/73 ms for the prior
accounting walks. Decode TPS was 7.170/9.530. Source-read time was
1.154/0.819 s, with 1.419/0.013 GB of process disk reads, despite
6.980/12.926 GB of logical expert reads. A larger explicit expert cache
does not guarantee fewer physical reads when the OS file cache differs.
These runs did not control or flush that cache.

A separate diagnostic candidate run at 16 GiB used the existing
`GGML_METAL_NO_RESIDENCY=1` switch for that process only. TPS rose
from 7.170 to 8.144 and TTFT fell from 1.113 to 1.012 s. Admission
fell from 258.5 to 36.3 ms and allocation/setup from 256.0 to 30.5 ms
during decode. Read time actually rose from 1.154 to 1.192 s; disk
reads were similar at 1.419/1.388 GB. This isolates additional residency
management overhead, rather than a cache-hit improvement. Current Metal
creates a residency set per buffer, periodically walks all sets under
one lock, and searches its collection when removing a buffer. This
experiment does not separate those mechanisms or qualify disabling
residency for production. No Metal residency default or implementation
was changed. A future optimization must isolate streaming buffer
management and preserve ordinary Metal behavior and memory safety.

All seven HTTP runs returned the same 24 token IDs and stayed within
their managed caps (peak 8,589,915,610 / 17,179,769,226 bytes).
Three selected Metal store tests passed 38 assertions, covering failed
recycled reads, oversized reuse and trace capacity, and uniform-slot
retirement/shrink/grow/trim accounting. The existing four-output 8 GiB
control also matched all 3,973,120 logit bytes of the saved ordinary
Metal baseline and its state digest `f4b73d884f1f4807`. The HTTP token
comparison is not a claim of bytewise logits coverage at all 24 positions.
No full suite or unrelated ordinary-path workload was run.

## Bounded residency groups for streamed Metal experts

A temporary device-side probe split the 16 GiB residency costs. During
23 decode steps, the heartbeat visited an average of 10,236.5 sets,
issued 3,521,357 `requestResidency` calls, and spent 1.085 s in those
walks. The allocation thread spent 225.9 ms waiting to register buffers
and 190.7 ms waiting to unregister them. Collection removal took 20.0 ms;
set initialization and teardown took only 16.4 and 5.7 ms. Heartbeat time
runs on another thread and is not additive with request latency or lock
wait time. Contention with the large heartbeat walk, rather than set
creation alone, was the main identified residency cost.

Qwen4Exp streaming now requests a private Metal buffer type through two
proc-address capabilities. The type owns bounded residency groups of up
to 64 independently allocated expert buffers. These are residency groups,
not compute-wave groups or shared physical slabs. Every buffer retains
its original physical size, tensor layout, ledger charge, zero-tail rule,
and synchronized lease lifetime. Removing a buffer commits its removal
from the set before releasing the Metal resource and its host allocation;
neighbors remain resident. Empty groups are destroyed immediately. Leases
keep the type owner alive even after destruction of the store facade.
Host/driver bookkeeping remains inside the existing 2 GiB runtime envelope.

Group mutations use the device heartbeat's lock because
[MTLResidencySet methods are not thread-safe](https://developer.apple.com/documentation/metal/mtlresidencyset?language=objc).
The fork-owned device include deliberately accesses the backend's private
buffer layout and residency collection inside that translation unit.
Allocation/free were adapted from upstream commit
`31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`; upgrades must check shared
allocation alignment and ownership, tensor lookup layout, residency
collection locking, and buffer type callbacks. Existing proc-address hooks
cannot allocate grouped sets without this backend-local implementation.
No private Metal layout is exposed to the inference layer and no public
llama/ggml ABI was extended.

Ordinary shared/private/mapped types retain their original allocators,
destructors, and successful type checks. The added type-recognition clauses
come after the existing checks; two new capability lookups come after
existing proc-address cases. Unknown/non-Metal types incur an additional
identity comparison. The grouped owner is created only by a qualified
Qwen4Exp expert store without an explicit buffer-type override. Unsupported
devices, SDK/runtime capabilities, disabled shared storage/residency, or
owner-creation failure retain the prior buffer type. There is no global
residency switch, new worker, or additional GPU submission.

The same 33-token HTTP workload was run before/after/after/before at
each cap, with one 8-output warmup and a 24-output measured request per
fresh server. At 16 GiB, TTFT was 1.079/0.994/0.991/1.127 s and TPS
was 7.276/8.659/8.621/7.270. Mean TTFT improved from 1.103 to 0.992 s
(10.1%); mean TPS improved from 7.273 to 8.640 (18.8%). Decode disk
reads were similar at 1.413/1.380/1.383/1.384 GB. At 8 GiB, TTFT was
1.429/1.431/1.432/1.446 s and TPS was 9.637/9.745/9.716/9.508.
The small 8 GiB difference does not establish a repeatable speedup.

Hits, misses, logical reads, evictions, allocation/reuse counts, submission
counts, and all 24 output IDs were identical before/after at each cap.
Both caps retained their previous peaks of 8,589,915,610 and
17,179,769,226 bytes in the HTTP probe. The remaining 8-versus-16 GiB
decode difference includes substantially more physical disk reads at
16 GiB; the OS file cache was neither controlled nor flushed.

The instrumented grouped 16 GiB run averaged 166 sets per heartbeat,
made 59,096 heartbeat residency calls during decode, and spent 22.2 ms
in the walks. Admission fell from 250.5 to 35.8 ms and allocation/setup
from 268.7 to 38.9 ms. The probe's old per-buffer registry counters do
not measure the new group-mutation critical sections separately; their
remaining costs are included in admission/allocation timings.

Three targeted Metal tests passed 82 assertions: a 130-buffer lifecycle
across three groups, Qwen4Exp zero-tail/reuse/read-failure behavior, and
the ordinary oversized-buffer control. The new lifecycle case covers
emptying a middle group, keeping pinned neighbors charged, shrink/grow,
refilling partial groups, and leases outliving the store. It was rerun
after matching upstream's autorelease-pool handling at buffer destruction.
Separate four-output 8 and 16 GiB controls each matched all 3,973,120 logit
bytes of the saved ordinary Metal baseline and state digest
`f4b73d884f1f4807`. Their peaks were 8,589,516,378 and 17,179,843,594 bytes.
The 24-output HTTP checks compare token IDs, not every logit byte. No full
suite, long-prefill performance qualification, or multi-context stress
test was run for this change.

The focused ordinary Metal HTTP control kept streaming and the managed
budget disabled, using the same upstream 8 GiB MoE cache as the earlier
native controls. With only the 8-output warmup, before/after/after/before
TPS was 5.384/5.182/4.703/5.198 and TTFT was
0.752/0.793/0.737/0.746 s. The reverse pair read 1.642/1.435 GB from
disk across the whole measured requests, so this apparent slowdown was
confounded by physical I/O. A single clarifying pair warmed all 24 output
positions first: TPS was 20.347/20.276, TTFT 0.757/0.786 s, and physical
reads 59.5/81.8 MB before/after. This warmed decode control found no
material slowdown (0.35% difference); it does not establish identical
latency for every cache state. All ordinary controls returned identical
24-token sequences. No ordinary cache policy or residency behavior was
changed to obtain these results.

## Explicitly cold GGUF cache and repeated HTTP requests

The grouped-residency server was measured sequentially at 8, 16, and
32 GiB, using the same 33-input/24-output prompt, context 1024, batch
512, microbatch eight, eight threads and one slot. Prompt reuse was
disabled. Each fresh server received three full requests: one after
explicit file-cache invalidation, then two identical repetitions.
No inference or ordinary-path implementation changed for this experiment.
Compact phase counters and memory observations are recorded in
[the file-cache evidence](atomic-turbine-filecache.json).

After server readiness, a temporary helper mapped only the two original
GGUF shards read-only and applied `MS_SYNC | MS_INVALIDATE`.
[Apple documents this invalidation operation](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/msync.2.html).
`mincore` observed zero resident pages for both files after each reset;
file size and modification time were unchanged. A small temporary fixture
first confirmed that the helper observed 64 cached pages and removed them.
No system-wide purge, artificial memory stress, or file-content writes
were used. Page counts were sampled again after each request, outside its
timing interval. The server's initialization had already acquired 2,379
expert slices before the first request, so "cold" refers to the GGUF
file cache, not an empty expert store. Startup and cache preparation are
excluded from TTFT. The two repetitions are observations of warming, not
a proof of eventual steady state or zero-I/O operation.

Results below use seconds, tokens/s, and decimal GB for I/O. Decode covers
the 23 steps after the first token. Physical reads are macOS per-process
disk-read deltas, including readahead and any other server reads; logical
reads count requested expert payload bytes. These counters need not match.

| Budget | Request | TTFT | Decode TPS | Prefill hit % | Decode hit % | Logical GB, prefill/decode | Disk GB, prefill/decode |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| 8 GiB | Cold file cache | 5.168 | 5.03 | 0.77 | 33.24 | 13.977 / 12.926 | 9.492 / 4.342 |
| 8 GiB | Repeat 1 | 2.651 | 7.00 | 1.08 | 33.24 | 13.936 / 12.926 | 2.084 / 1.173 |
| 8 GiB | Repeat 2 | 2.361 | 7.11 | 1.08 | 33.24 | 13.936 / 12.926 | 1.712 / 1.126 |
| 16 GiB | Cold file cache | 4.662 | 4.78 | 42.47 | 64.18 | 8.099 / 6.980 | 8.323 / 5.080 |
| 16 GiB | Repeat 1 | 3.299 | 5.01 | 51.46 | 64.18 | 6.878 / 6.980 | 3.279 / 3.089 |
| 16 GiB | Repeat 2 | 3.256 | 5.02 | 51.46 | 64.18 | 6.878 / 6.980 | 3.083 / 3.073 |
| 32 GiB | Cold file cache | 4.913 | 6.42 | 45.93 | 83.24 | 7.628 / 3.301 | 8.196 / 3.397 |
| 32 GiB | Repeat 1 | 0.758 | 16.41 | 100.00 | 100.00 | 0 / 0 | 0.007 / 0.003 |
| 32 GiB | Repeat 2 | 0.718 | 15.51 | 100.00 | 100.00 | 0 / 0 | 0 / 0 |

After repeat two, cached GGUF pages occupied 10.82/7.81/5.67 GiB at
8/16/32 GiB. System-wide file-backed memory was 26.58/19.25/8.58 GiB,
while wired memory was 9.53/17.39/27.88 GiB. These are system counters,
not isolated allocations attributable entirely to inference. Compression
occupied 1.37/1.36/1.81 GiB. No swap-outs occurred during the measured
requests; only the first 8 GiB request saw 16 pages of system swap-ins.

The 16 GiB repeat-two decode requested 46% fewer expert bytes than
8 GiB, yet read 2.73 times as many bytes from disk. The physical/logical
read ratios were approximately 44% versus 8.7%. Together with the smaller
observed file cache and larger wired footprint, this supports competition
between the explicitly resident expert copies and original GGUF pages.
The engine cap does not include the OS file cache: an 8 GiB engine can
benefit from many additional GiB of cached file pages. Hit rate alone
therefore does not predict either physical I/O or whole-request speed.
This is measured contention on this host/workload, not a universal rule
that 16 GiB is slower.

At 32 GiB, the complete expert working set for this repeated request fit
after the first pass. Managed peak was only 26.500 GiB: the cap is a
ceiling, not an eager allocation. Both subsequent prefill and decode had
zero expert misses. The small first-repeat disk traffic was not expert
payload traffic; the second repeat had no process disk reads at all.
Thus the larger cache wins once it eliminates this workload's misses,
even though fewer original GGUF pages remain cached.

For this short repeated workload, 8 GiB is the preferable smaller-memory
profile and 32 GiB is the fastest warmed profile. The intermediate
16 GiB profile cannot be recommended for speed solely on its larger cap.
Do not turn these three measurements into an automatic cap-selection
threshold: different prompts, context sizes, background applications and
file-cache histories change the working set and available RAM. Future
cache sizing should consider physical I/O and total memory pressure,
not merely expert hit rate. No default, I/O caching policy, or budget
semantics were changed.

All nine requests returned identical 24-token sequences and respected
their managed caps. This experiment did not recapture full logits or
run a test suite. A newly linked temporary probe aborted in argument
parsing before inference; its results were discarded. Measurements used
the existing qualified, standalone grouped-residency probe instead.
The absolute warm timings differ from earlier unconditioned runs because
file-cache history was explicitly reset here and warming was bounded to
two repetitions. Results do not establish cold process-start latency,
long-prompt performance, or an idle-machine upper bound.

## Context 4096 and a single-server request series

The current 8 GiB Metal profile was checked at context 4096, the current
experimental context ceiling, with a 3,481-token raw-completion prompt.
One server/slot processed the long prompt twice and then the established
33-token short prompt. Each request generated 16 tokens with temperature
zero, seed 42, `ignore_eos=true`, and `cache_prompt=false`. This is a
fixed-length numerical/performance probe, not a conversational quality
evaluation or a claim about natural EOS length.

The server used batch 256, microbatch eight, eight threads, one slot,
`--cache-ram 0 --ctx-checkpoints 0 --no-warmup`, and the fixed 8 GiB
engine budget. Both long requests processed all 3,481 input tokens in
thirteen 256-token slices followed by 153 tokens. No input was truncated
or skipped through prompt caching. The repeated input is available as
[the context fixture](atomic-turbine-qwen38-context.txt), with structured
measurements and hashes in [the evidence](atomic-turbine-qwen38-context.json).

The same three requests then ran sequentially in a separate ordinary
Metal server with identical context, batch, microbatch and sampling
settings. It used the previously qualified upstream host-expert control:
`--cpu-moe --no-repack --moe-cache-mib 8192 --moe-cache-cpu-share 0`,
with the managed engine budget disabled. Its cache budget is not a total
8 GiB engine cap. No ordinary execution code or policy was changed.

| Request | Prompt tokens | Streaming TTFT, s | Streaming decode, tokens/s | Ordinary TTFT, s | Ordinary decode, tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long, first | 3481 | 178.596 | 7.517 | 195.300 | 4.470 |
| Long, repeat | 3481 | 174.805 | 7.256 | 181.802 | 5.190 |
| Short, same slot | 33 | 1.916 | 7.657 | 2.838 | 5.270 |

TTFT excludes server startup and includes the first sampled token.
Decode TPS uses the 15 steps after that first output. A temporary server
probe outside the repository captured pre-sampling FP32 logits and
process memory observations; its synchronous logit output is included
in these timings. Filesystem cache state was not reset or controlled,
and the order was not counterbalanced. The ordinary measurements are a
numerical reference, not evidence of an equal-memory speedup.

Managed high-water was 8,589,819,482 bytes, below the 8,589,934,592-byte
cap throughout the series. After every request, managed use was exactly
8,588,619,882 bytes and the idle minimum was 6,550,394,346 bytes.
The cumulative released counter increased by 2,248,176 bytes after each
long request and remained unchanged after the short request. The larger
context did not leave additional managed charges after the second pass.

Streaming process footprint at the three end-of-generation observations
was 7,335,237,704 / 7,335,450,696 / 7,335,483,464 bytes. The last observed
lifetime peak was 7,335,516,232 bytes (6.832 GiB). Endpoint growth from
the first to the last request was 245,760 bytes, not proportional to
prompt length. These observations do not establish absence of leaks over
an extended soak. The ordinary process's observed lifetime footprint
peak was 9,318,870,840 bytes (8.679 GiB). RSS and OS file-cache pages
are separate measurements and must not be added to footprint or the
engine ledger as though they were independent allocations.

Each request captured 15,892,480 logit bytes (16 complete vocabulary
rows). Direct byte comparison found zero differing FP32 values across
all 47,677,440 bytes and identical generated token IDs. Both long slot
files, each 225,618,840 bytes with 3,496 saved tokens, matched their
ordinary counterparts byte for byte. The subsequent short slot file
also matched byte for byte: 119,516,984 bytes with 48 saved tokens.
The two streaming long runs additionally had identical logit and state
SHA-256 hashes. Thus the larger history and transition back to a short
request preserve the checked numerical and serialized-state behavior.

Only this three-request series per mode was run. No full suite,
multi-slot load, context shifting, slot restoration, long soak, or
context beyond 4096 was tested here. The result qualifies this concrete
text-streaming profile, not all family variants or arbitrary long-context
workloads. Long prefill remains the practical bottleneck: roughly three
minutes to process 3,481 tokens at 8 GiB, despite a much shorter follow-up
request. Production source was unchanged; only the server target was
rebuilt to align its objects with the existing common-parameter headers.

## CUDA admission gate: 4 GiB RAM / 2 GiB VRAM

This section records the initial gate before CUDA PLE row streaming.
The [follow-up implementation and allocation checks](atomic-turbine-qwen38-cuda.md)
remove PLE residency and identify the subsequent state-budget failure.

On 2026-11-15, both original local GGUF shards were inspected with the
repository's `GGUFReader`, without reading weight payloads or loading the
model on a GPU. The [metadata result](atomic-turbine-qwen38-cuda-admission.json)
records the exact sizes and the admission calculation. This is a necessary
memory lower-bound check, not a CUDA runtime or peak-memory measurement.

The 144 routed tensors can remain on disk, but the current CUDA loader still
retains `per_layer_token_embd.weight`: IQ4_NL, shape `[160, 320001536]`,
28,800,138,240 bytes (26.82 GiB). `llama_model_base::load_tensors` enables
`ml.stream_ple` only for a managed UMA context on MTL0; the executor's
`configure_rows` also rejects discrete backends. The discrete loader disables
whole-file mapping and lazy loading. Moving layers between CPU and CUDA does
not remove or split this resident PLE allocation.

The table alone exceeds the 4 GiB RAM cap, the 2 GiB VRAM cap, and their sum.
All non-routed payload is 32,916,001,280 bytes. Including the unchanged 2 GiB
host runtime envelope gives a combined lower bound of 35,063,484,928 bytes
(32.66 GiB), before state, scratch, staging, expert working set and alignment.
Thus no layer split on the current CUDA path can meet the requested caps.
No allocation or inference trial is needed to establish that rejection.

There are two additional capability gates, independent of the capacity proof:

- The streaming executor permits this checkpoint's Q2_0, IQ2_XXS, IQ2_XS,
  IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL routed formats only on Qwen4Exp Metal.
  This is a streaming qualification gate, not a claim that upstream CUDA
  cannot compute those formats.
- The common CUDA server loader selects the qualified LFM2 boundary-reduction
  mode, whose context guard rejects Qwen4Exp. Removing that guard would not
  establish Qwen arithmetic equivalence.

The initial inventory missed the Qwen3.8 shards already present under the
Ubuntu Strata installation; the token-row follow-up uses those complete files.
Its GPU reports 8188 MiB total and its host reports 66,052,665,344 RAM bytes.
No 75.84 GB model transfer was started: the metadata proof already fails the
requested engine-budget gate. No CUDA forward pass, baseline comparison or
new performance claim was made. Production source and the ordinary execution
path remain unchanged; no test suite was run.

### Required implementation order

1. Add an explicitly gated CUDA PLE row-read path using original packed bytes,
   with host admission and synchronized ownership. Qualify its gather output
   against ordinary CUDA before removing the resident table.
2. Qualify the checkpoint's routed formats and preserve Qwen4Exp routed
   arithmetic without enabling the LFM-specific boundary-reduction mode.
3. Measure actual separate-domain admission at 4/2 GiB, first on a short
   explicit context and microbatch. Keep the 2 GiB runtime envelope intact.
4. Only after admission succeeds, compare complete logit rows and serialized
   state with ordinary CUDA using the same placement and request. An ordinary
   control may use more memory; it must not weaken the streamed caps.

Even hypothetical PLE row streaming is not proof of fit. Removing only the
PLE payload leaves 4,115,863,040 resident bytes. After the runtime envelope,
just 179,104,256 bytes (170.81 MiB) remain across both domains for all state,
compute, staging, experts and alignment. Separate-domain placement further
constrains that space. The existing Metal floor is not a CUDA floor; another
bounded resident-weight path may be necessary. Do not silently increase caps,
reduce the runtime envelope or report this hypothetical arithmetic as admission.

The payload inspection can be repeated without a model transfer:

```sh
PYTHONPATH=gguf-py python3 - <<'PY'
from pathlib import Path
from gguf import GGUFReader

root = Path.home() / 'code/models/qwen38-iq3'
resident = 0
for path in sorted(root.glob('Qwen3.8*.gguf')):
    reader = GGUFReader(path)
    for tensor in reader.tensors:
        if '_exps.weight' not in tensor.name:
            resident += int(tensor.n_bytes)
        if tensor.name == 'per_layer_token_embd.weight':
            print(tensor.name, tensor.tensor_type.name, tensor.shape, tensor.n_bytes)
print('non-routed payload:', resident)
print('payload plus runtime envelope:', resident + 2 * 1024**3)
print('combined requested cap:', 6 * 1024**3)
PY
```

## Long-prefill I/O profile and bounded file-cache advice

The context-4096 fixture above was profiled at the same 8 GiB budget,
256-token server batch, eight-token microbatch, eight threads and one slot.
The 3,481-token request took 211.281 s with detailed timers. Its prefill
made 2,885,267 expert-cache misses and 490,408 hits, requested
1,663,154,291,200 original weight bytes, and incurred 116,028,379,136
process disk-read bytes between the start and first-token probes.
Logical reads include repeated copies from the OS file cache; they are
not SSD traffic. A separate unprofiled control took 184.716 s, so the
profiled wall time must not be compared directly with candidate timing.

Expert reads accounted for 158.090 s, including 154.886 s in source
transfer. Admission was 2.394 s, allocation/setup 1.899 s and graph
construction 1.318 s. GPU submission was 6.884 s, completion wait
11.632 s, and measured GPU command service 13.584 s. These spans overlap
and nest: in particular the 133.829 s compute span includes synchronous
lookahead reads, not just GPU computation. They must not be summed.
A ten-second stack sample independently placed most main-thread samples
inside the read syscall, while seven idle CPU workers frequently polled.

The retained change gives macOS bounded file-cache hints for the next
known-demand group before its synchronous reads. It uses
[F_RDADVISE](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html),
which can initiate asynchronous file-cache reads without a user destination.
Only the automatic Qwen4Exp Metal ClassLRU prefill profile at exactly
8 GiB enables it. Explicit policy overrides, trace runs, short requests,
decode, other budgets, CUDA and ordinary llama.cpp retain their I/O path.
The existing admitted lookahead capacity bounds each group to at most
16 selected experts; resident experts receive no hint. A window is
consumed by checked demand reads before the next window is issued.

No new thread, user buffer, mapping, lease or replacement-policy touch
is introduced. Source offsets, validation, cancellation and publication
still belong to the unchanged synchronous demand read. Unsupported or
failed hints fall back to that read. Kernel file-cache pages remain OS
managed, as with existing buffered reads; the engine ledger is not a
system-wide RAM or file-cache limit. The implementation stays in the
fork-owned source, store and executor without an upstream hook or public
llama ABI change. The ordinary path never constructs this executor.

A 523-token probe reduced TTFT from 31.969 to 26.708 s (16.5%) with the
same 410,915 misses and 238,396,441,600 logical read bytes. Synchronous
expert-read time fell from 23.722 to 15.646 s. Hint syscall time is outside
that read counter and included in whole-request timing; the read-time
difference is not the end-to-end speedup.

Other small experiments were rejected: microbatch 16 preserved token IDs
but changed logits and state even on the 33-token control; 512 KiB cache
rounding reduced logical bytes by only 0.05% while TTFT rose to 32.867 s;
disabling automatic filesystem read-ahead took 31.374 s and disabling
CPU polling took 31.781 s, neither establishing a convincing gain.
Those experiments remained in temporary binaries and changed no defaults.

Two full unprofiled pairs ran in BEFORE / AFTER / AFTER / BEFORE order,
with a fresh server per run and the same 33-token request immediately
after the long request in its slot. Each generated 16 tokens:

- `full-before`: long TTFT 184.716 s, decode 7.502 tokens/s; short TTFT 1.914 s, decode 7.871 tokens/s.
- `full-after`: long TTFT 174.751 s, decode 7.276 tokens/s; short TTFT 1.933 s, decode 7.761 tokens/s.
- `full-after2`: long TTFT 173.492 s, decode 7.257 tokens/s; short TTFT 1.978 s, decode 7.811 tokens/s.
- `full-before2`: long TTFT 202.107 s, decode 6.787 tokens/s; short TTFT 2.170 s, decode 7.374 tokens/s.

Paired long-request TTFT reductions were 5.4% and 14.2%. File-cache
state was not purged or controlled, so these are scoped paired results,
not cold-storage guarantees or a statistical distribution. No decode
acceleration is claimed. The hint is inactive during decode and the short
request; indirect file-cache effects can still vary their timings.
Process cumulative disk counters are recorded in the evidence, including
startup; they are not phase-separated SSD bytes or evidence of reduced
physical traffic. Builds and measurements were serialized.

All candidate outputs matched the saved ordinary Metal control by direct
byte comparison: 63,569,920 logit bytes (64 complete vocabulary rows),
the token IDs, two 225,618,840-byte long states and two 119,516,984-byte
short states. Both baselines also matched. The candidate's managed
high-water remained 8,589,819,482 bytes under the 8,589,934,592-byte cap,
with idle use 8,588,619,882 bytes and minimum 6,550,394,346 bytes.
The largest observed candidate process-footprint peak was 7,324,981,272 bytes
(6.822 GiB). Kernel file-cache residency is separate.

Structured counters, rejected experiments, hashes and comparisons are in
[the prefill evidence](atomic-turbine-qwen38-prefill.json). The long input
remains [the context fixture](atomic-turbine-qwen38-context.txt).


The focused source test passed 23 assertions covering packed bytes across
shards, invalid hints, validation after truncation and a poisoned source.
The selected Metal automatic-policy boundary test passed ten assertions.
No aggregate suite, cold-cache campaign, long soak, context shift,
concurrent slot, other model variant or native pressure test was run.
Large prefill still rereads a substantial expert working set across
microbatches. This improves scheduling of known reads; it does not remove
that traffic or qualify larger microbatches as bitwise interchangeable.

## Reducing repeated expert reads during long prefill

The retained 1 MiB Metal slots and bounded file-cache advice above remain
in place. This follow-up changes only automatic long-prefill victim
selection in the fork-owned expert store. It keeps the same eight class
queues and 128-token frequency half-life, but compares current decayed
frequency instead of dividing age by the coarse frequency class.

The previous rule can protect newly read cold entries because their age
is zero, while a frequently reused entry can share the same coarse class
and lose its place. The new rule considers the oldest and newest unpinned
entry of each class, so a recent cold entry can compete for eviction.
There are at most sixteen scored candidates; active leases are skipped
at queue ends. The score is `age / 128 - log2(crf)`, the logarithm of
inverse current frequency. This avoids exponential overflow for old
entries. Weights predating optional history have zero frequency and are
cold candidates. Ties prefer the lower class and then the larger stable
tensor/expert key. It is a bounded heuristic, not a search for the globally
least valuable resident expert.

The existing automatic gate remains Qwen4Exp/Metal, an uncached prompt
of at least 320 tokens, microbatch eight, enabled known-demand overlap,
no explicit policy/trace/preview override and an applied/requested 8 GiB
budget. Decode, short requests and a resize away from the profile restore
plain LRU and release history. Explicit `class-lru` preserves the original
victim rule for comparison. The ordinary llama.cpp path does not construct
this store. No upstream boundary, graph arithmetic, model bytes or public
ABI changed. The policy adds a mode flag inside the already covered
runtime metadata; it adds no history allocation, weight buffer or worker.

The saved 523-token ATET3 trace reproduced all 451,776 old runtime
acquisitions, including hit decisions and physical allocation sizes.
The new candidate predicted 220,547,584,000 prefill read bytes versus
235,633,574,400 for ClassLRU, a 6.40% reduction with the same history and
staging accounting. This trace uses context 1024. The runtime probes below
use context 4096 and a different available expert-cache capacity, so their
absolute counters should not be equated with replay.

The optional replay is available without changing its default behavior:

```sh
python3 tools/expert-source/replay-cache.py long-class.trace \
  --prefill-frequency --output replay.json
```

Capture the control trace with the inference runner's existing
`--expert-trace long-class.trace --cache-policy class-lru` options.
Only prefill replay counts predict the proposed policy: that trace keeps
the explicit policy during generation, whereas automatic runtime restores
LRU. Replay is not a latency or physical SSD model.

Three other directions were rejected. Weighting the original class-head
score by payload size slightly increased reads; adding a 128-token age
floor reduced reads by only 0.21%. Compact 64 KiB allocations with exact-size
reuse predicted a much larger 14.02% prefill read reduction. Runtime did
reduce reads from 238.396 to 204.789 GB, but TTFT increased from 28.184 to
38.520 s. Allocation/setup increased from 0.269 to 8.230 s and admission
from 0.309 to 2.975 s. Fewer read bytes did not offset buffer churn, so
compact allocation was not retained.

The retained frequency rule on that 523-token runtime probe reduced
prefill reads from 238.396 to 222.753 GB (6.56%), misses from 410,915 to
384,084, and TTFT from 28.184 to 26.195 s. Original logit and state hashes
matched. Uniform slots remained reusable; allocation/setup was 0.167 s.
These small probes enabled phase timers and are not the full-prompt
performance measurement.

The full 3,481-token context fixture used the same 8 GiB engine budget,
context 4096, server batch 256, microbatch eight, eight threads and one
slot. Every run generated 16 tokens with greedy sampling, seed 42,
`ignore_eos=true` and no prompt-cache reuse, then processed the same
33-token control in that slot. A fresh server ran each case in
BEFORE / AFTER / AFTER / BEFORE order. Detailed phase timers were disabled;
cache counters and process I/O were recorded at phase boundaries.

Prefill logical bytes fell from 1,663,154,291,200 to 1,638,479,744,000:
24,674,547,200 fewer bytes, or 1.48%. Hits increased from 490,408 to
532,575 and misses decreased from 2,885,267 to 2,843,100. Both repeated
runs of each policy had identical logical counters. All prefill misses
reused existing buffers, with zero new physical allocations after the
start probe. GPU submissions stayed at 239,868.

There is a measured tradeoff after prefill: decode reads increased from
8,362,598,400 to 8,554,508,800 bytes (0.192 GB, 2.29%) over the fifteen
decode steps after the first output. The policy has already returned to
LRU, but the resident working set differs. Total long-request logical
reads still decrease by 24,482,636,800 bytes. The following short request
has unchanged prefill/decode counts of 13,984,870,400 / 7,858,124,800 bytes.
No decode speedup is claimed.

- `full-before`: long TTFT 192.695 s, decode 7.141 tokens/s; short TTFT 1.978 s, decode 7.839 tokens/s. Long prefill/decode process disk reads: 128.240/1.223 GB.
- `full-after`: long TTFT 176.348 s, decode 6.954 tokens/s; short TTFT 1.992 s, decode 7.764 tokens/s. Long prefill/decode process disk reads: 121.159/1.271 GB.
- `full-after2`: long TTFT 178.314 s, decode 6.912 tokens/s; short TTFT 2.026 s, decode 7.621 tokens/s. Long prefill/decode process disk reads: 123.416/1.304 GB.
- `full-before2`: long TTFT 179.645 s, decode 7.129 tokens/s; short TTFT 1.981 s, decode 7.744 tokens/s. Long prefill/decode process disk reads: 114.118/1.198 GB.

Paired long TTFT reductions were 8.48% and 0.74%. File-cache state was
not purged or controlled. In the reverse pair the candidate actually
incurred more process disk traffic despite fewer logical reads. The
repeatable result is the reduction of expert-read work; these timings
do not establish a stable percentage speedup or lower physical SSD
traffic. Short-request timing and decode differences are reported above
rather than hidden behind unchanged arithmetic or an inactive policy.

All candidate outputs matched the saved ordinary Metal control by direct
byte comparison: 63,569,920 logit bytes (64 full vocabulary rows), token
IDs and four slot files (two 225,618,840-byte long states and two
119,516,984-byte short states). Both controls also matched. Managed
high-water stayed 8,589,819,482 bytes under the 8,589,934,592-byte cap;
idle use was 8,588,619,882 bytes and the minimum 6,550,394,346 bytes.
The maximum candidate process-footprint peak was 7,321,753,600 bytes
(6.819 GiB); OS file-cache pages remain outside the engine ledger.

Full counters, hashes, replay predictions and rejected experiments are in
[the retention evidence](atomic-turbine-qwen38-retention.json). The long
input is unchanged in [the context fixture](atomic-turbine-qwen38-context.txt).


Three selected C++ store cases passed 29 assertions for cold-tail
selection, entries predating history, buffer reuse, pinned retirement,
shrink completion, explicit ClassLRU and restoration of global LRU order.
One selected Python replay case passed for cold-tail selection, pinned
candidate exclusion and staging capacity. The complete saved trace also
validated the unchanged old-policy replay. No full suite, cold-cache
campaign, native pressure test, other context size, multi-slot load or
long soak was run. The measured text fixtures do not qualify every prompt
or family variant. Most repeated weight traffic remains; this is a modest
retention improvement, not removal of the small-microbatch I/O limit.

## Cross-microbatch expert reuse: feasibility, not runtime qualification

This section records the initial offline investigation. The isolated runtime
prototype in the following section updates its implementation status.

The current 8 GiB prefill completes all layers for each eight-token
microbatch. A later microbatch usually needs weights that have already
been displaced by other layers. This section evaluates a different
traversal: for a bounded window, finish one layer for each original
microbatch in chronological order before advancing to the next layer.
Each projection retains its original token count, routed IDs and acquire /
release order. This is not a larger matrix batch. Increasing the actual
microbatch to sixteen previously changed logits and state, so that
shortcut remains rejected.

A diagnostic-only store patch captured the automatic frequency policy on
the unchanged 3,481-token context fixture: context 4096, server batch 256,
microbatch eight, one slot, eight threads and sixteen greedy outputs.
Unlike the public trace control, the probe observes automatic retention
without replacing it with explicit ClassLRU or disabling read advice.
It starts when automatic retention becomes active and closes before the
transition back to LRU. Its tensor manifest maps source indices to layer
and gate/up/down names. The patch is an artifact for an isolated diagnostic
build; it is not applied to runtime sources.

The 436-microbatch prefill contains 61,479 routed projections and 3,375,675
expert acquisitions. The final layer's FFN is absent from microbatches
without requested outputs. The one-token prompt tail is included in
prefill, not mislabeled as generation. Replay reproduced every observed
hit/miss and physical allocation size, with 532,575 hits, 2,843,100 misses
and 1,638,479,744,000 logical read bytes. Replay starts empty and fails
closed if unrecorded initial residency changes an observed decision;
eviction totals are simulation counters, not validated runtime counters.

The candidate uses plain LRU in layer-major order. Windows stop at the
existing 256-token server batch boundaries. Capacity is the smallest
observed cache-plus-read-staging allowance (2,040,311,558 bytes), minus
an explicit hypothetical workspace reserve. Policy history is not credited
back, making this capacity conservative. With a 16 MiB reserve:

- Window one reads 1,950,017,894,400 bytes: 19.01% more than current retention. It is a plain-LRU control, not the current runtime.
- Window two reads 1,603,389,158,400 bytes: 2.14% less.
- Window four reads 1,254,804,377,600 bytes: 23.42% less.
- Window eight reads 794,282,060,800 bytes: 51.52% less.
- Window thirty-two reads 299,595,904,000 bytes: 81.72% less.

Window eight still reads exactly 794,282,060,800 bytes with a 256 MiB
workspace reserve. Its largest single-layer working set contains 780
unique projection/expert keys, requiring 818,176,320 bytes in the existing
uniform slots, including tensor arenas. Window thirty-two reaches 1,038
keys / 1,088,803,872 slot bytes. These are physical cache allocations,
not just packed payload sizes. Miss staging and recorded pinned leases
remain accounted for in replay.

The residual alone is small: the model has 2,560 embedding dimensions
and four hyperconnection streams, so eight eight-token microbatches need
2,621,440 bytes for one F32 residual per token; thirty-two need 10,485,760.
These are lower bounds, not complete workspace estimates. Snapshot inputs,
attention masks, graph metadata, backend copies and live temporaries must
also be admitted by the shared ledger before a runtime window can begin.
The 16 / 256 MiB reservations are sensitivity parameters, not measurements
of an implemented scheduler.

The trace-producing runtime retained byte-identical ordinary Metal results:
15,892,480 logit bytes (sixteen full vocabulary rows) and the 225,618,840-byte
serialized long-request slot were directly compared with the saved native
control. Managed high-water was 8,589,819,482 bytes under the 8,589,934,592-byte
cap. Observed trace-run TTFT was 178.195 s and decode was 6.857 tokens/s;
these describe the existing traversal with diagnostic I/O. They are not
candidate timings. No reordered inference graph has run, so no candidate
bitwise-equivalence, latency or physical SSD claim is supported.

### Implementation boundary and next gate

The large I/O opportunity is established on the full fixture. The next
runtime target is a bounded window of eight original microbatches; begin
with two only to isolate state/ordering correctness. Window thirty-two
is an offline opportunity, not the first implementation target.

A separate streaming-only runner must preserve the following contracts:

1. Freeze each microbatch's original inputs, output pruning, attention
   mask and cache-index metadata before advancing its memory context.
   `llama_memory_hybrid_idx_context::apply()` mutates KV/indexer layout;
   the recurrent context also selects state slots. Reusing mutable context
   pointers for later input initialization would expose future metadata.
2. Execute each layer's KV, recurrent, PLE convolution-history and QSA
   pooled-key writes in microbatch order. A layer output tensor alone is
   not a complete boundary: builders also expand explicit state-copy nodes.
3. Own every live cross-boundary activation and backend copy. The current
   `process_ubatch()` reuses one graph result and scheduler, and its allocator
   can alias regions across splits. Passing a graph view to an already
   allocated scheduler does not select that view: compute executes the
   scheduler's saved split list. The observation callback is not a resumable
   graph cursor. Neither mechanism safely implements this traversal.
4. Retain original eight-token operation shapes and check that new cuts
   do not change Metal fusion/kernel selection. Compare all logits and the
   complete serialized state with the sequential control before timing.
5. Admit additional graph/input/activation storage against the same 8 GiB
   ledger, draining leases and asynchronous work before resize or abort.
   Fall back before starting a window if admission fails. Do not partially
   rerun an already committed window against mutated recurrent state.

No ordinary or streaming execution path, public ABI, scheduler or model
builder was changed by this investigation. Runtime enablement, actual
workspace accounting, abort/resize/restore and multisequence qualification
remain open. Three selected replay tests passed for exact observations,
server boundaries, reserve-dependent eviction, lease validation and malformed
or incomplete traces. No full suite or unrelated runtime check was run.

### Reproduction and artifacts

The fork-owned offline tool rejects mismatched observations before writing
candidate results:

```sh
python3 tools/expert-source/replay-microbatches.py long.trace \
  --manifest long.trace.tensors --prefill-tokens 3481 \
  --observed-policy prefill_frequency --batch-tokens 256 \
  --windows 1 2 4 8 32 --reserve-mib 16 --output replay.json
```

The observed input must contain exactly the prefill, including its short
last microbatch, without a decode suffix. It must retain complete ATET3
lease events. For the capture, apply the
[diagnostic patch](atomic-turbine-qwen38-microbatch-probe.patch) only to an
isolated build and set `QWEN_WINDOW_TRACE` to a writable output path.
Start the server with the profile above and `--no-warmup`,
`--engine-budget-mib 8192`, `--cache-ram 0`, `--ctx-checkpoints 0` and
`-fit off`. Submit the context fixture with `cache_prompt=false`,
`temperature=0`, `seed=42`, `ignore_eos=true` and `n_predict=16`.
Do not additionally enable the public expert-trace override. Replay is
an offline tool and its working memory is not part of inference admission.

[Structured evidence](atomic-turbine-qwen38-microbatches.json) records trace
and manifest hashes, the diagnostic workspace, exact-output checks, memory,
read counts and workspace sensitivity. The large trace and model are not
added to the repository. The result justifies implementing an isolated
layer-window prototype; it does not qualify enabling that prototype.

## Isolated layer-window runtime prototype (historical)

The opt-in implementation below supersedes this diagnostic. The old patch and
evidence preserve the initial experiments; they are not the current build path.

The next gate now runs real interleaved graphs on Metal. Two original
microbatches (a 16-token prefix) and eight original microbatches (a 64-token
prefix) both match ordinary Metal byte for byte, including full serialized
state immediately after prefill and after three further decode steps.
Both tests use context 4096, microbatch eight, eight threads, one sequence
and the first tokens of the existing context fixture. The streaming engine
cap remains 8 GiB. This is a fresh-context, single-window diagnostic, not
server integration or a full-length prefill qualification.

### Ownership and execution

`tools/expert-source/build-qwen38-window-probe.py` builds separate control
and prototype executables in a caller-selected diagnostic directory. It
copies the context and Qwen4Exp model compilation unit, patches only those
copies, and replaces their objects in a copied static library. The working
sources, original libraries and production executables are not modified.
The small generated hook diff is recorded in
[the runtime probe patch](atomic-turbine-qwen38-window-probe.patch).

The initial fork-owned diagnostic prepared one original graph and
one scheduler per microbatch, calling the memory context's `apply()` and
`set_inputs()` before advancing to the next microbatch. Each graph keeps
its own inputs, backend buffers and live intermediates. A worker remains
inside its original scheduler compute call and pauses at the existing,
synchronized observation callback. Only one worker computes at a time;
all others wait. Thus the prototype neither treats callback cancellation
as resumption nor substitutes graph views into an already allocated
scheduler. It requires no ggml scheduler changes or private scheduler
layout access.

The existing allocation observer charges backend buffers to the shared
ledger. A separate conservative ticket covers graph arenas, scheduler host
arrays, allocator/split metadata, callback scratch and bounded worker
stacks. The bound tracks the checked scheduler implementation with thirty
initial split inputs, at most three backends and no pipeline copies. The
build tool rejects a changed split-input default; allocator changes still
require reviewing this bound. This is a diagnostic bound, not a generally
supported allocator contract. Failure discards the context; transactional
abort, resize and fallback after partial progress are not implemented.

### Confirmed state-ordering failure and correction

The initial two-graph serial control already matched native logits and
state. Interleaving those graphs at `l_last` names failed: every first-
microbatch layer matched the serial trace, while the second microbatch
matched layer zero and first diverged at layer one.

The model's PLE layer is layer one. `build_ple()` calls the recurrent-state
gather helper, which explicitly expands its state reads before expanding
the operations that consume the preceding layer's residual. Without a
prior graph expansion of `l_last-0`, those PLE history reads appear before
the apparent layer-zero observation boundary. The second microbatch
therefore reads stale PLE history before the first has processed PLE.
The layer output's name alone was not a safe boundary.

The diagnostic model-builder copy now explicitly expands `res_hc` after
each `l_last` callback, before constructing the following layer. This is
an ordering barrier, not a new arithmetic operation. The corrected serial
control and interleaved run both match native bytes. The first callback
matcher also needed exact tensor names: accepting every `l_last-` prefix
included derived tensors as duplicate boundaries. The final matcher accepts
only a complete layer number with no suffix and validates chronological
layers zero through forty-seven.

### Scoped results

For the initial two-window implementation, both the corrected serial and
interleaved controls match:

- 3,973,120 logit bytes: four complete vocabulary rows.
- 118,532,216 bytes of prefill state.
- 118,624,520 bytes of state after continuation.

The eight-window implementation bounds each graph arena and scheduler to
8,192 slots, rather than the initial overprovisioned two-window reserves.
It validates a 7,769-tensor upper bound before scheduler allocation. All
384 observed boundaries execute in layer-major order. Against the ordinary
64-token Metal control, both the prototype and unchanged streaming match:

- 3,973,120 logit bytes, SHA-256 `d98002da4717607a1330b76a4ffb56931f8782f41ff1cfa2dd2ee785bfccc923`.
- 120,009,080 prefill-state bytes, SHA-256 `6f995082b8bbfc07b2caf91fdb0425267f9e6b0c0fc5778fece7d6245a03d8a7`.
- 120,101,384 continuation-state bytes, SHA-256 `53d6a29f8d7e5ada47dbee0f84830d52885ea55a0e299c389fa0a8ec11906baa`.

These were direct file comparisons, not token-only or digest-only checks.
All four greedy token IDs also match. The cap includes a 32 MiB reservation
for state-file staging, whose largest state tensor range is checked before
capture.

The unchanged streaming control has zero prefill cache hits, 52,008 misses
and 30,258,278,400 logical read bytes. The real eight-window execution has
14,684 hits, 37,324 misses and 21,827,008,000 bytes: 27.86% less I/O work.
The control uses ordinary streaming LRU at this short prompt length; this
is not a comparison with long-prefill automatic frequency retention.

The prototype reserves 1,750,617,728 bytes (1.630 GiB) for host metadata
and worker overhead. After graph preparation its fixed admitted use is
8,415,957,754 bytes, leaving only 173,976,838 bytes (165.917 MiB) for cache
and miss staging. Managed high-water reaches 8,589,737,514 bytes, below
the 8,589,934,592-byte cap. This accounts for the large difference from the
offline window-eight result with a hypothetical 16 / 256 MiB workspace.
The first prototype proves ordering and reuse, but pays heavily for
independent scheduler host reservations.

Single unpaired short-prefill timings were 2.872864 s for unchanged
streaming and 2.561553 s for the prototype. File-cache state was uncontrolled;
the prototype includes graph construction, worker synchronization and
per-layer hashes. These timings do not establish a stable speedup, server
TTFT, decode TPS improvement or the full prompt's performance. Native
control timing is not an equal-memory performance reference.

### Historical evidence

[Structured runtime evidence](atomic-turbine-qwen38-window-runtime.json)
records the initial controls, rejected no-barrier result, exact comparisons,
layer order, memory and I/O. The diagnostic include files have been superseded
by the isolated implementation in `src/llama-prefill-window.cpp`.

## Opt-in full-prefill layer windows

The server now accepts `--engine-prefill-window`, disabled by default. It keeps
each arithmetic microbatch at eight tokens, prepares up to eight independent
graphs, and executes the same layer across those microbatches before advancing
to the next layer. Weights stay resident across that window instead of being
reread after traversing the other 47 layers. Windows stop at server batch
boundaries; the final partial window and microbatch are supported.

The qualified profile is the local Qwen3.8 IQ3_XXS fixture on Metal, one text
sequence, context 4096, batch 256, microbatch 8, fixed total engine budget 8 GiB.
The full 3481-token request is 55 windows, including one partial window. The
following 33-token request uses ordinary scheduling. Numerical qualification
compares every vocabulary logit for 16 generated tokens and both serialized
slots directly with the saved ordinary Metal reference. That reference uses
the upstream CPU-host expert cache and is not an equal-total-memory baseline.
The performance control is current streaming with the new option absent.

### Measured server result and tradeoff

Two unprofiled paired observations on the same fixture give:

- Pair 1: TTFT 167.474 s sequential versus 125.432 s windowed (25.10% less).
- Pair 2: TTFT 162.999 s sequential versus 122.150 s windowed (25.06% less).
- Prefill logical reads: 1,638,479,744,000 versus 794,282,060,800 bytes
  (51.5232% less); hits rise from 532,575 to 2,010,252. Submissions stay 239,868.
- Physical prefill reads: 74.50 / 46.10 GB sequential versus 53.34 / 30.37 GB
  windowed. OS cache state was uncontrolled: these are observations, not cold-I/O
  guarantees. Builds and other GPU checks did not overlap measurements.
- Window managed high-water is 8,589,922,714 bytes against 8,589,934,592 allowed.
  Peak observed process footprint is about 7.42 GB; file cache and undocumented
  driver allocations are not the engine-ledger contract.

Decode is a small measured tradeoff: 7.30 / 7.63 tokens/s windowed versus
7.93 / 7.83 sequential, a 7.99% / 2.63% decrease for the 15 decode steps after
the first token. Logical decode reads increase from 8,554,508,800 to
8,615,475,200 bytes (0.71%), and new expert buffer allocations are 645 versus
one after window resources are released. This identifies extra work; it does
not isolate all of the timing difference from file-cache noise. The prefill
savings are roughly 41–42 seconds, versus about 0.05–0.16 seconds of additional
measured decode time in this request. No decode speedup is claimed.

The subsequent short request takes 1.911 / 1.746 s TTFT with the option versus
1.750 / 1.776 s without it. Its logical prefill/decode reads, hits, misses and
submissions are unchanged. The first short result is slower; the second overlaps
the controls. This does not establish a universal no-regression guarantee for
OS cache state after windowed prefill, so the feature remains explicit opt-in.

An initially chained variant took 274.755 s: choosing LRU inadvertently removed
macOS known-demand file-cache advice, which had been gated on ClassLRU. Keeping
that advice while the window capability is active restored the benefit. The
profiled corrected diagnostic took 129.804 s and spent only 1.173 s preparing
all graphs; those instrumented timings are excluded from the paired comparison.

All eight server results (long and short in four processes) passed direct byte
comparisons of full logits and saved slots. An ordinary 64-token Metal control
with the shared scheduler hook also matches its pre-change reference. Final
preflight-only guards were added after timing and checked with eligible text,
explicit cache-policy override and synthetic embedding input; they do not alter
the qualified text scheduling or arithmetic. Detailed counters, hashes,
chronological order, targeted tests and limitations are in
[the structured server evidence](atomic-turbine-qwen38-window-server.json).

### Isolation and memory ownership

`llama-prefill-window.cpp` and its worker header own scheduling. The existing
model graph callback expands `l_last` before the next layer is built, preserving
the PLE-history ordering established by the diagnostic. The model builder and
arithmetic kernels receive no additional changes. A worker pauses only at a
synchronized observation boundary. Workers take turns; they do not concurrently
execute graphs on a shared backend. The implementation does not use abort
callbacks as resumable coroutines.

A private ggml factory shrinks only freshly constructed window schedulers.
Ordinary scheduler sizes and algorithms stay intact. The admitted window owns
512 MiB of host metadata reserve (64 MiB per slot, including graph metadata and
bounded worker stacks); a separate 224 MiB transient reserve covers scheduler
construction. Device/backend buffers remain charged individually to the same
engine ledger, including lazy allocations on workers. A guard checks expanded
allocator graph arrays and dependency-view capacity before writes; oversized
original graphs are rejected before scheduling. The private layout contract is
pinned to revision `31953bcf74ba489a8bbb0c936c5fcf24fcf9485c`: upstream updates
must review constructor capacities, input copies, optimizer dependencies,
graph-result arenas, allocator metadata and stack bounds together.

Necessary shared hooks are the guarded decode dispatch and two small scheduler
allocation hooks. Disabled windows add no workers, graph arenas, I/O, timers or
GPU work. The unavoidable structural cost is one `size_t` in each private ggml
scheduler and its allocation-time guard, plus context dispatch guards. Window
control metadata exists only in already-managed contexts. This is not a claim
that every ordinary configuration has undergone a new performance qualification.

### Fallback and resize contract

Unsupported request shapes, short prompts, custom eval/abort callbacks,
embeddings, adapters, automatic memory policy and other budgets use the existing
sequential path. The optional feature is not CUDA, multi-sequence, MTP or
multimodal qualification. Ledger admission and scheduler construction precede
any KV/recurrent state mutation. Insufficient metadata reserve falls back before
an input is applied. Graph preparation/compute failure after mutation marks the
context unusable; it must be discarded, never replayed against advanced state.

A manual resize requested during compute is applied after the current window
finishes and all its workers, schedulers, graph arenas and reservations are
released. The memory iterator already identifies the next unprocessed microbatch;
the original decode loop handles that remainder. No tokens are applied twice.
The focused 128-token test requests 8 to 7 GiB from a separate thread while a
real expert callback is outstanding. It completes one 64-token window, resumes
the remaining 64 tokens sequentially, matches all logits and full state, and
accepts growth back to 8 GiB. Rejected admission with a 1280 MiB diagnostic
reservation also matches the native 64-token reference without starting windows.

The P11 follow-up below qualifies slice-boundary cancellation and explicit
file-based rollback. Native OS pressure, broader context/model coverage,
automatic checkpoint rollback and longer multi-request stress remain open. The
ordinary/default server path retains its earlier support boundaries.

### Reproduction

Use the existing context fixture and normal qualified flags, adding:

```sh
--engine-budget-mib 8192 --engine-prefill-window -c 4096 -b 256 -ub 8 -np 1 \
  -t 8 -tb 8 -ngl 99 -fit off --cache-ram 0 --ctx-checkpoints 0 --no-warmup
```

To build only the small comparison driver against current libraries:

```sh
python3 tools/expert-source/build-qwen38-window-probe.py \
  --build build-metal --output /tmp/qwen38-window-probe
```

`window-compare MODEL native|stream PROMPT_FILE OUTPUT_STEM` writes `.f32`,
`.prefill.state`, `.state` and `.json`. `QWEN_WINDOW_TOKENS=64` or `128`
selects the checked prefix; `QWEN_WINDOW_RUNTIME=1` enables windows in streaming
and supplies a whole-prefill token hint. `QWEN_WINDOW_HOLD_MIB=1280` exercises
admission fallback; `QWEN_WINDOW_SHRINK=1` on the 128-token case exercises the
threaded request and sequential continuation. These environment controls are
confined to the diagnostic driver; the server uses the explicit CLI option.

The final eligible 64-token text case and explicit-policy fallback match native
logits and both full states. Synthetic embedding input starts zero windows.
An equal-hint sequential repeat matches the window-bypass result byte for byte,
but another option-off control differs in logits/state. Thus synthetic embedding
repeatability remains unqualified even on the pre-existing path; this is not a
new multimodal support claim. The initial harness also used unequal whole-prefill
hints; that mismatch was corrected before the repeated-control observation.
The embedding discrepancy is recorded as an open boundary, not a passing
bitwise qualification. The qualified server requests contain text tokens only.

The three focused `test-prefill-window-scheduler` cases cover exact CPU results
and scheduler reuse, oversized input rejection, and backend dependency expansion
rejection with an ordinary-scheduler control. No full suite or aggregate gate
was run.

## Known-route advice after windowed prefill

The next profiling pass separates cache refill allocation from read latency.
For the previous profiled long request, 15 decode steps spent 1056 ms reading
experts and 48 ms allocating/resetting buffers, including 18 ms of padding.
File validation and seeks accounted for only 15 ms. Therefore no new slab pool,
shared graph/cache buffer ownership, or earlier speculative refill was introduced:
moving the same allocations into prefill would not prove a whole-request gain.

Instead, after an explicit layer window has run, the fork-owned executor may
hint one group of already-selected single-token expert ranges to macOS. The
existing demand reads still validate file identity/size and publish only complete
original GGUF bytes. There are no new workers, user buffers, leases, graph nodes
or arithmetic changes. The hint limit is the smaller of the current compute
capacity and router selection count. The asynchronous demand-acquisition limit
is unchanged; file advice does not add background expert-cache publication.
Single-token projections include the final one-token microbatch of a later short
prefill, not just generated tokens.

The state is set only by entering `--engine-prefill-window`. Disabling that
option or accepting an automatic memory policy clears it. Read-overlap disable,
explicit cache-policy overrides, tracing, and requested/applied caps other than
8 GiB suppress it. Default execution and contexts that never entered a window
retain their previous behavior. The implementation changes only fork-owned
executor/context control files; it adds no upstream hook or public llama ABI.

### Targeted evidence

The full server fixture is unchanged: 3481 tokens, then 33 tokens in the same
slot, 16 outputs each, context 4096, batch 256, microbatch 8, total engine cap
8 GiB. Two comparisons use opposite orders:

- Profiled control then candidate: long decode 7.40 to 8.26 tokens/s (+11.67%);
  short-followup decode 8.01 to 8.83 (+10.27%). Long decode read time falls
  from 965.5 to 664.1 ms; allocation stays about 48 ms.
- Unprofiled candidate then control: long decode 7.38 to 8.32 tokens/s (+12.70%);
  short-followup decode 8.02 to 8.87 (+10.62%). Long TTFT is 124.01 versus
  123.18 s; the prefill algorithm/work is unchanged, so no new TTFT gain is claimed.
- Every full-vocabulary logit row and both serialized slots in all four server
  runs match the saved ordinary Metal reference byte for byte. Managed peak
  remains 8,589,922,714 bytes. Logical prefill reads remain 794,282,060,800 bytes;
  long decode still reads 8,615,475,200 bytes and allocates 645 cache buffers.

The short 64-token/31-decode crossed-order diagnostic did not show a stable
warmed gain: its last control was slightly faster than the advice runs. Hint
syscalls have overhead, and reduced time inside `fread` alone is not a speedup.
The old diagnostic also incorrectly propagated its whole-prefill token hint into
single-token decode. The driver now supplies that hint only for its initial
prefill; server measurements used the correct behavior throughout. Use the
full server pairs for the performance conclusion, not the prefix diagnostic.
OS file-cache state was uncontrolled; no purge, memory-pressure workload,
concurrent build or GPU check ran during measurements. These observations do
not establish a universal cold/warm-cache improvement.

Six focused final integration cases count entries into the advice helper in an
isolated instrumented store: active windows, option disabled, automatic policy,
overlap disabled, cap shrunk to 7 GiB, and a context that never enabled windows.
Only the active case calls the helper (44,640 entries for 31 decode steps; cached
entries can be skipped before any syscall). All other cases count zero calls.
Full logits and prefill/final states match the appropriate saved controls, and
usage stays within each applied cap. These counters are absent from production.
The [guard instrumentation patch](atomic-turbine-qwen38-decode-guard-probe.patch)
and [structured evidence](atomic-turbine-qwen38-decode-advice.json) record the
checks, controls, phase data and source hashes.

The existing comparison driver now accepts `QWEN_WINDOW_STEPS=4..128` (default
4), records synchronized per-decode milliseconds and separate decode I/O counts,
and enables optional decode phase timing with `QWEN_WINDOW_DECODE_PROFILE=1` in
streaming mode. Entry zero in `decode_ms` is prefill; remaining entries exclude
logit/state serialization and token selection. The normal build helper still
builds only this diagnostic executable against existing libraries.

The P11 follow-up below qualifies two additional text prompts and the selected
server lifecycle. Native memory pressure and synthetic embedding repeatability
remain open. No full suite or aggregate
verification ran, and no new claim is made for ordinary-path throughput.

## P11 windowed HTTP lifecycle and alternate prompts

This is the Qwen3.8 Flash Next GSQ-RCO IQ3_XXS Metal profile with one
text slot, context 4096, batch 256, microbatch 8, eight CPU threads,
and a fixed 8192 MiB engine budget. Windowed prefill is explicit opt-in.
Automatic context checkpoints and the RAM prompt cache are disabled.
The server is bound to localhost. A temporary server build captures
the full FP32 logits before the sampler and emits prompt-slice boundaries.
No production code changes were needed for this qualification.

- An SSE client disconnected while the second 256-token slice of the
  1080-token inventory prompt was running. The server recorded cancellation
  before the slice completed, released the slot at 512 processed tokens
  after eight 64-token windows, and did not generate any output. The
  client observed idle after 9.03 s. This is cancellation at a server
  slice boundary. It does not interrupt an active window or expert read.
- A fresh request in the same slot processed the complete inventory prompt
  with zero cached tokens. Its full logits and serialized slot matched
  the ordinary Metal reference.
- The 1087-token slot was saved and restored in a fresh server, then
  advanced to 1091 tokens, erased and rolled back using the saved file.
  Re-saving the restored slot reproduced the original bytes. Both
  continuations reused the same 1087 cached tokens and matched in all four
  logit rows, token IDs, and all serialized state bytes. The continuation
  also matched the ordinary Metal control. This covers explicit file-based
  rollback with server checkpointing disabled.
- On the inventory prompt, `stop=["blue"]` stopped after one sampled token
  and omitted the matched text. A positive bias on token 248046 tested
  the EOS path with one sampled token and no subsequent decode step.
  Both requests ran 17 windows and their raw first logit row matched
  the unstopped request. EOS was forced through the sampler for this test.

The second server processed the restore/continuation, stop and EOS cases,
then processed a new 1398-token maintenance-ticket prompt (22 windows)
and the 1080-token inventory prompt again (17 windows). Both distinct long
prompts and the repeated inventory were byte-exact against ordinary Metal.
Their observed TTFT/decode throughput were 49.824 s / 7.46 tokens/s and
38.232 s / 7.73 tokens/s respectively. These are series observations, with
no matched performance control at the same memory limit. The ordinary
control uses a CPU-host expert cache as a numerical reference.
After both long requests, idle engine usage was 8,588,619,882 bytes.
The series peak was 8,589,917,210 bytes against the 8,589,934,592-byte cap.
The total at the end was 73 completed windows in the second server.
The first inventory run's SSE collector initially counted progress
events as output. The raw capture and saved state were retained, the
progress placeholders were removed, and its client TTFT was discarded.

This passes the selected fixed-budget, single-slot windowed text lifecycle
and the two additional prompts. The option remains explicit opt-in.
Faster cancellation inside a slice, automatic checkpoint rollback, native
pressure, multi-slot execution, and a prolonged soak are separate checks.
See [the comparison evidence](atomic-turbine-qwen38-window-p11.json).
