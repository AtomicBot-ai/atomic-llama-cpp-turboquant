# P07 Qwen runtime preview and bounded prefetch

This report records the original runtime path. The
[cost follow-up](atomic-turbine-qwen-preview-cost.md) replaces opt-in mode 2 with
matrix-tail scheduling and a specialized four-row predictor; the original
first-window behavior is retained as diagnostic mode 3. Default remains off.

## Decision

The independent runtime path is implemented and numerically qualified for the
measured Qwen3.5 MoE checkpoint on M4 Max. It remains **off by default**: the
crossed comparison did not demonstrate acceleration over the existing
known-route read/GPU overlap. This is an opt-in experiment, not a new production
performance claim or completion of all P07 work.

The main graph builder, actual router IDs, expert arithmetic and GGUF bytes are
unchanged. This implementation does not revive the rejected shadow graph branch
from the [offline preview evaluation](atomic-turbine-preview.md).

## Execution and ownership

`--expert-preview preview` enables only the predictor; `--expert-preview prefetch`
also enables optional reads. `off` is the default. The idle-only extension API is
`llama_expert_preview_set(ctx, mode)`, with modes 0/1/2 respectively. Context-level
admission requires budgeted Qwen3.5 MoE on Metal and LRU. Demand-only ATET tracing
is explicitly incompatible with speculative accesses; the runner also excludes
simultaneous offline preview capture. Other architectures and selection-bias
routers are not qualified by this path.

At the first gate/up projection of a source layer, the CPU uses that projection's
existing normalized input and a validated host-readable alias of the next F32
router. No router weights are copied. Prediction runs after an existing expert
GPU graph is submitted and before its completion wait, on the caller thread.
There is no worker thread, second GPU graph, or inference graph consumer.

Independent eight-lane float accumulation predicts scores without changing
compiler floating-point rules or inference kernels. Ties use ascending expert
ID. Predictions are approximate and never replace actual routing. Rows are
computed lazily in rank-first, row-order traversal; stop after four unique,
nonresident future expert bundles across the whole microbatch. Each bundle has
original gate/up/down slices. The predictor is skipped if the actual current
projection working set is fully resident, or if the entire target layer is
resident. The former is conservative: it may miss a useful cold-next-layer
opportunity in exchange for avoiding speculative overhead on warm requests.

The predictor owns an explicitly charged 24,616-byte workspace on this build,
including descriptors, scores and bounded row Top-4 storage. Immutable router
aliases refer to already charged model weights. This raises the qualified
ubatch-32 minimum from 4,933,600,320 to 4,933,624,936 bytes.

Before optional reads, admission retains headroom for the current GPU group,
known-route lookahead and one largest read reservation. Remaining capacity
allows zero to four complete bundles; at most 12 optional slices are pinned.
All acquisitions use the existing store, original source validation, LRU,
physical buffer reuse and ledger. A miss cannot bypass budget admission.

Optional leases end within the source projection callback, after synchronization.
Only ordinary evictable cache residency survives into later projections. No
prediction, I/O queue or extra lease survives to a graph/resize boundary. A source
failure after submission synchronizes GPU work before leases unwind and rejects
the projection through the existing sticky error path. Actual demand remains
authoritative even when predictions are unused or evicted early.

This deliberately differs from the offline optimistic replay: leases are
released sooner, prediction may run for a next layer later pruned from the main
graph, and CPU prediction has real cost. Runtime measurements supersede replay
estimates for this implementation.

## Measured comparison

The partially cached comparison uses Ornith 1.5 35B A3B Q4_K/IQ4_XS, context 512,
microbatch 32, the checked-in 190-token prompt, 16 generated tokens, one sequence,
greedy sampling, LRU and a 5 GiB total managed budget. Existing known-route
overlap stays enabled in both arms. Model loading is outside request time.

Initial preview-only execution preserved all logits and state. Its predictor
cost was 101.55 ms over 819 generations and 1,965 evaluated rows; its different
OS read charge makes its total request time unsuitable as an isolated overhead
estimate against later runs.

Three prefetch request observations were 7.815528166, 7.767311292 and 7.908096875
seconds; three baseline observations were 7.864682791, 7.595640125 and 7.759767875.
The initial on/off pair was followed by on/off and off/on pairs. Median request
time was **7.815528166 s with prefetch vs 7.759767875 s baseline**, or 0.72% slower.
Median prefill was 6.199551958 vs 6.160046916 s. These small samples and overlapping
ranges do not prove a general regression, but they fail the acceleration gate.

Each baseline request read 38,824,181,760 logical bytes with zero cache hits.
Prefetch read 39,255,277,568 bytes, 1.11% more, despite 9,069 hits. It moved
5,582,094,336 bytes into speculative acquisition and admitted 3,276 expert bundles.
The representative predictor cost was 99.70 ms, speculative acquisition 476.84 ms;
completion wait fell from 2.503 to 2.346 s in the initial pair. These are nested
host timings, not measured GPU/CPU intersection. Matmuls (87,072), submissions
(17,367) and projections (2,505) stayed identical.

## Confirmed warm-cache overhead and correction

At 24 GiB after one warmup, two baseline requests took 6.123360583 and
6.135320833 s with zero reads. The initial candidate still predicted on a warm
working subset: 6.920281709 and 6.491291125 s, about 400/397 ms predictor time,
and 95,420,416 / 11,927,552 unnecessary read bytes. Requiring the entire target
layer to be cached was insufficient to detect an all-hot *actual working set*.

The final current-demand-residency guard eliminates that confirmed work. Both
final measured requests skip all 819 generations, with zero prediction rows,
zero speculative acquisitions and zero reads. Times were 6.580275166 and
6.220484375 s. This proves removal of predictor/read work, not timing equivalence
or universal non-regression. The default therefore remains off. The primary
crossed comparison predates this guard; no final-binary speedup is claimed.

## Accuracy, memory and focused checks

Every comparison, warm control and final live request preserves all emitted
logits bitwise against resident controls, all committed IDs and state digest
`9755c384a23111fe`. The final ubatch-128 complete-answer control additionally
matches output token 23, EOG 248046 and state `d8bd6c5d1cf32b92` bitwise.

The fixed 5 GiB candidate peaks at 5,368,596,808 of 5,368,709,120 managed bytes.
The final live case shrinks during prefill to 4,706 MiB, exercises latest-pending-
request wins, grows to 6,144 MiB, shrinks to 5,120 and 4,706 MiB, rejects 1 MiB,
and grows again. All observed used/applied values remain valid. The lifetime
peak is 6,442,156,056 of the largest 6,442,450,944-byte cap. The near-floor
shrink releases 1,318,488,928 ledger bytes in 44.25 ms; growth allocates no cache
payload immediately. There are 351 budget skips and 465 predictor generations;
optional work resumes when capacity returns. Rejected shrink preserves cap and
usage. State and logits remain exact throughout.

One new, directly selected executor case passes on CPU and Metal, 21 assertions
per backend: charge/release, preview-only behavior, global bundle bound, exact
output bytes, callback lease lifetime, warm subset bypass, floor rejection,
shrink/grow, and speculative read failure after GPU submission with drained
leases and cleared output. The affected case was rerun after adding the warm
subset guard. No unchanged suite or aggregate verification ran. Inference,
executor-test and server targets built; server request behavior was not qualified.
The real-model work comprises 12 invocations, 15 measured requests and 3 warmups.

Profile counters accumulate across requests: subtract before/after snapshots.
Store read/admission counters include speculative acquisitions; `acquire_ns`
covers demand/known-route work and `prefetch_ns` covers speculative acquisition.
Capture artifacts and kernel file cache are outside managed engine payload; the
ledger is not an RSS limiter. No cold-SSD, tail-latency, energy, native-pressure,
other checkpoint/architecture or larger-context qualification is claimed.

## Reproduction

Add `--expert-preview preview` or `--expert-preview prefetch` to the existing
budgeted inference runner command. Use `--profile-experts --state-digest --logits`
for the corresponding diagnostics. For the focused executor checks:

```sh
build-metal/bin/test-expert-executor --backend CPU 'runtime preview.*'
build-metal/bin/test-expert-executor --backend MTL0 'runtime preview.*'
```

Results and hashes are under `p07_preview_runtime` in
`atomic-turbine-evidence.json`. Raw local results use
`/tmp/atomic-turbine-qwen-overlap-rt-*`. A future activation decision needs a
measured scheduling improvement over the current cumulative baseline; cache hit
counts alone do not justify enabling this experiment.
