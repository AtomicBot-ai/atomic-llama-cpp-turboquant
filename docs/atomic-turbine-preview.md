# P07 next-layer preview and bounded prefetch

This report records the offline stage. The subsequent
[Qwen runtime evaluation](atomic-turbine-qwen-preview-runtime.md) delivers an
independent opt-in predictor/prefetch path; its acceleration gate did not pass,
so the default remains off. Statements below describe the earlier offline scope.

## Decision

Keep speculative next-layer prefetch disabled. This package qualifies read-only
route capture and an offline evaluation, not a runtime speedup. The existing
known-route read/GPU overlap remains enabled for the two qualified architectures.
The preview approximation is promising, especially for Qwen decode, but the
available evidence does not yet justify adding its execution path.

A trial that added a shadow router branch to the main GGML graph failed the LFM
numerical gate: the first committed-ID difference was at zero-based index 32,
with 42 outputs instead of the reference 52 and a different state digest. That
branch and its model flag were removed. Its exact numerical cause was not
isolated; changes to graph consumers, scheduling and evaluation callbacks were
not independently controlled. It is not accepted equivalence or performance
evidence. No shadow branch remains in the delivered graph builder.

## Local plan and delivered scope

1. Audit the next-router/current-FFN-input approximation and the global Top-4
   bound. Do not transfer results from other checkpoints or microbatch sizes.
2. Capture the existing normalized FFN input and actual selected IDs at the CPU
   expert bridge. Bind by call, microbatch, sequence, absolute token position,
   layer and projection cursor without adding graph consumers.
3. Compute preview offline, validate every captured projection against ATET3
   demand, and replay the fixed-cap LRU baseline exactly before reporting any
   candidate result.
4. Compare bounded prediction with a future-ID oracle under the same cache
   capacity, physical allocation/reuse rules and demand lease events.
5. Enable runtime prefetch only after a numerically equivalent implementation
   demonstrates latency benefit over the existing cumulative baseline. This
   last acceptance gate remains open.

Shared-first scheduling, chain speculation and native OS-pressure qualification
are outside this package. There is no new prefetch thread, model clone, enlarged
microbatch or model-weight conversion.

## Capture and predictor

`llama-expert-infer --preview-trace PATH` writes an ATP2 JSONL file and a
`PATH.bin` companion. The opt-in observer sees existing tensors synchronously;
callback pointers expire on return, callbacks must not reenter the engine, and
exceptions use the executor's sticky failure path. File writes cannot silently
turn a partial observation into a successful request. Controls must be set while
idle. Capture supports one fixed-budget request without warmup, repeats, resize
or pressure injection on Qwen3.5 MoE and LFM2 MoE.

The helper copies existing router weights and bounded activation rows to
caller-owned diagnostic files. A fixed 64 KiB stack buffer and bounded JSON
serialization use the existing runtime envelope; no engine activation queue or
persistent router copy is allocated. Diagnostic files, output logits, offline
NumPy arrays and kernel file cache are outside managed engine payload, as with
existing caller-owned trace outputs. The ledger is not an RSS cap.

The predictor multiplies the next layer's original router matrix by the current
layer's normalized FFN input. It deliberately omits the intervening residual,
attention and FFN effects. Qwen ranks raw scores, equivalent to softmax ranking;
LFM applies sigmoid and the next router's selection bias before ranking. Offline
float32 reduction and stable expert-ID ties are explicit evaluation conventions,
not claims of backend-exact preview arithmetic. Only actual route IDs feed the
inference computation.

ATP2 uses little-endian tensor bytes on the qualified machine. The evaluator
checks complete tensor ranges, widths, finite data, ID bounds/uniqueness, row
positions, contiguous layers, call/microbatch progression, projection cursors,
and all gate/up/down demand unions. It rejects unbound demand or binary tails.
The final layer follows last-row-only output pruning; it may be absent from
non-output prefill microbatches. Prediction never crosses calls or microbatches.

## Bounded replay policy

The baseline reproduces every recorded LRU hit/miss and physical allocation.
The candidate uses the same `Cache` implementation as `replay-cache.py`, including
unpinned-only eviction, physical buffer reuse and a full read reservation.

At a source layer's first projection, iterate each row's predicted Top-4 in rank
order, then row order, deduplicating globally. Skip fully resident bundles and
admit at most four unique future experts across the whole microbatch. Each expert
is a gate/up/down bundle. The bound is four total, not four times the row count.

Only one future generation holds optional leases. At target-layer entry those
leases become ordinary LRU residency before a new future generation is admitted.
Current demand still follows its original A/R lease events; early speculative
entries can be evicted before their target projection. Such reads count as
unused even with correct oracle IDs. Pending byte attribution in the offline
analyzer is not an additional runtime queue.

Admission reserves eight worst-size demand allocations plus one maximum read
reservation, using conservative physical sizes for new optional buffers. At most
12 optional slices can be pinned; all resident allocations and transient reads
remain within the recorded cache capacity. Budget-infeasible bundles are skipped.
This demonstrates a possible payload bound, not a complete runtime predictor/
thread/queue budget contract. Any future implementation must account for those
additional owners and invalidate/drain speculation safely on shrink or pressure.

The oracle substitutes actual future Top-4 IDs using the same global selection
and admission rules. It is an optimistic control, not an optimal scheduler or a
realizable predictor. Replay assumes prefetch reads finish in time and assigns
zero predictor/scheduling cost. It cannot establish critical-path stall savings,
SSD traffic, TTFT, latency tails or energy benefit.

## Observations

Both captures use M4 Max, context 512, microbatch 32, one sequence, greedy output,
original GGUF bytes, default LRU and the previously accepted known-route overlap.
LFM uses the checked-in 200-token prompt at 4 GiB and completes with 52 tokens
plus EOG. Qwen uses the 190-token prompt at 5 GiB and checks a 16-token prefix;
this Qwen observation is not a new complete-answer/EOG qualification.

LFM capture contains 2,236 layer records and 2,177 adjacent pairs. Top-4 row
precision/recall is 79.43% in prefill and 82.32% in decode; LFM selects four actual
experts per row. Qwen has 835 records and 814 pairs. Precision is 94.04% in
prefill and 90.26% in decode; recall is 47.02% and 45.13% because Qwen selects
eight actual experts while the predictor offers four.

For LFM, baseline logical reads are 99,582,345,216 bytes. Bounded prefetch adds
51,041,894,400 speculative read bytes and leaves 56,622,219,264 demand-read bytes.
Total reads increase by 8.12%; 8,081,768,448 speculative bytes are unused.
Under perfect timeliness it moves 43.14% of baseline reads ahead of demand.
The oracle moves 51.18% and still adds 0.0717% total reads through early eviction.

For Qwen, baseline logical reads are 38,824,181,760 bytes. The candidate adds
5,548,015,616 speculative bytes and leaves 33,672,069,120 demand-read bytes.
Total reads increase by 1.02%; 395,902,976 speculative bytes are unused.
The optimistic demand-read reduction is 13.27%, compared with the oracle's
14.29%. Oracle total reads increase by 0.00152% through early eviction.

The split matters: LFM prefill moves at most 10.83% of baseline read bytes, while
decode moves 80.30% and adds 17.14% extra I/O. Qwen prefill moves 5.07%, while
decode moves 44.00% and adds 4.75% extra I/O. Aggregate row accuracy is therefore
insufficient to decide whether prefetch pays off. In particular, these results
do not prove that extra I/O makes prefetch slower, or that earlier demand hits
make it faster. The critical missing evidence is correct runtime timeliness and
predictor overhead relative to the already enabled known-route overlap.

Both replays reach the four-expert/12-slice bound. Peak optional physical payload
is 31,461,696 bytes for LFM and 7,082,304 for Qwen. Candidate cache-plus-read peaks
are 732,969,936 of 732,970,288 bytes and 436,151,520 of 436,288,816 bytes,
respectively. These are residual cache capacities after all other ledger owners,
not new allocations on top of the engine cap. The actual capture runs peak at
4,294,966,576 of 4,294,967,296 managed bytes for LFM and 5,368,572,192 of
5,368,709,120 for Qwen.

## Targeted validation and reproduction

Read-only capture preserves all emitted logits bitwise against resident controls,
all committed IDs and final state: LFM `84ac8e66a7115244`, Qwen
`9755c384a23111fe`. LFM EOG also matches. The failed graph-branch trial is recorded
separately. Capture wall times include diagnostic file writes and are not used
for speedup claims.

Two new observer cases pass on CPU (10 assertions): existing strided tensor
binding, unchanged output bytes, detach, budget/lease preservation, and callback
failure before reads with a cleared output and sticky error. Four Python cases
cover rank-first deduplication, ties/LFM selection bias, exact baseline replay,
unused reads, oracle usefulness, budget rejection, stale joins, duplicate IDs,
truncated tensors, mismatched demand, missing layers and invalid release.
Only these new cases ran. The inference runner, executor test target and server
built successfully; no server-request qualification or full suite ran. Final
header dependency cleanup after real captures changes no execution behavior.

With Python 3 and NumPy available, record using the existing runner's model,
prompt, context, budget and output flags plus:

```sh
--preview-trace /tmp/preview.jsonl --expert-trace /tmp/demand.atet
```

Then evaluate and run the focused analyzer cases:

```sh
python3 tools/expert-source/evaluate-preview.py \
  --capture /tmp/preview.jsonl --demand /tmp/demand.atet \
  --output /tmp/preview-evaluation.json
python3 tests/test-expert-preview.py
build-metal/bin/test-expert-executor --backend CPU 'route observer.*'
```

Machine-readable results and artifact hashes are stored under `p07_preview` in
`atomic-turbine-evidence.json`; local raw artifacts use the
`/tmp/atomic-turbine-preview-*` prefix. No automatic activation threshold is
inferred from this two-request sample. Runtime preview/prefetch remains gated on
an independent numerical and latency qualification.
