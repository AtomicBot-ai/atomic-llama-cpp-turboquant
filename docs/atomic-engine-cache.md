# P06: demand traces and replacement policy

## L2 scope

The current bottleneck is repeated synchronous expert loading: the 5 GiB,
microbatch-32 request read 38.8 GB logically with no cache hits. Before changing
execution order or adding asynchronous I/O, measure the avoidable misses.

1. Capture actual expert acquisitions, rounded allocation cost, transient read
   staging and available cache capacity from the shared UMA ledger.
2. Reproduce the existing LRU hit/miss decisions exactly in offline replay.
3. Compare one upstream-derived ClassLRU candidate on the same requests/capacity.
4. Only integrate the candidate if the saved reads justify its cost; then compare
   real logits, state, memory and request latency with the LRU control.
5. Record the decision and retain the reproducible trace/replay tool.

Changes touch the expert store/executor, an idle-only diagnostic callback in the
context extension API, and the native runner. No router, kernel, state, precision
or upstream-base change is needed. The diagnostic writer is caller-owned and
streams records to a buffered file rather than retaining a growing trace.

The candidate borrows the upstream 128-token decayed frequency, eight classes,
LRU within a class, and age / 2^class victim ranking from `moe_cache_lru` at
`31953bcf74ba489a8bbb0c936c5fcf24fcf9485c` (upstream PR #31233).
It is an adaptation for byte-sized, independently released projection buffers,
not a replay of upstream's fixed-slot GPU cache. The bounded executor never
requires all selected experts to fit simultaneously. CPU miss computation and
upstream's large-batch bypass are not part of this candidate.

Invariants: original demand order and GGUF bytes, shared total admission including
miss staging, no eviction with live leases, physical free before charge release.
Replay assumes a synchronous single-lease executor and fixed total cap; changing
fixed graph allocations is captured in each access's capacity. Resizes and
multiple/warmup requests are deliberately rejected by this diagnostic mode.

The minimum check is exact LRU replay against captured hits, then a matched real
request only if a replacement policy is selected. Offline read reduction is not
an SSD throughput or tokens/second claim; OS page-cache state is uncontrolled.

## Implementation and reproduction

`llama_expert_cache_policy_set(ctx, "class-lru")` selects the candidate before
the first evaluation. `"lru"` remains the default. Unknown policies, an active
trace, prior acquisitions, or insufficient remaining working floor are rejected.
The frequency history and tensor clocks are reserved before allocation, counted
as runtime bytes, and included in the minimum feasible budget. They survive
cache eviction and resize; active model state is never an eviction candidate.

Each projection advances its own token clock once, and each unique acquired
expert updates frequency once. Replacement examines the oldest unpinned entry
in each frequency class and chooses the largest token age divided by 2^class.
Ties choose the lower class, matching upstream. Unlike upstream batch planning,
only actual live leases are protected: the whole routed union need not fit.
The class search reuses the store's entry scan; no unbounded access log or
additional per-access queue allocation is introduced.

The native runner accepts `--cache-policy lru|class-lru` and
`--expert-trace FILE`. The current ATET3 format is a whitespace-separated stream: a configuration
record reports candidate/active history bytes, projection records give tensor
index and token count, access records give expert ID, payload/staging-reservation bytes,
rounded resident allocation, available capacity and observed hit, and reset
records report a full quiescent cache trim. ATET2 appends the minimum required
allocation after the observed-hit field; actual allocation can be larger after
victim reuse. Replay applies each policy's own reuse decisions and verifies the
observed policy's physical sizes as well as hits. ATET3 also pins each access
until an explicit `R` release record; replay excludes held entries from either
policy's victim search and rejects incomplete lease lifetimes. Historical
ATET1/ATET2 traces retain their original ownership rules; ATET1 keeps exact-size
allocations. The writer checks output failure.
Tracing is opt-in and is not intended for latency qualification.

The [LFM follow-up](atomic-engine-lfm-performance.md) validates ATET2 against a
real trace and ClassLRU control: 6.59% fewer logical read bytes, without a
demonstrated latency advantage. LRU remains the default.

Set `MODEL` to the existing original GGUF. Reuse the Metal configuration in the
memory document; build only the affected targets:

```sh
cmake --build build-metal --target llama-expert-infer test-expert-store -j4
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 5120 \
  --cache-policy lru --prompt-file docs/atomic-engine-prefill.txt \
  --tokens 16 --ctx 512 --ubatch 32 --state-digest \
  --expert-trace /tmp/cache-lru.trace --output /tmp/cache-lru.json \
  --logits /tmp/cache-lru.f32
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 5120 \
  --cache-policy class-lru --prompt-file docs/atomic-engine-prefill.txt \
  --tokens 16 --ctx 512 --ubatch 32 --state-digest \
  --expert-trace /tmp/cache-class.trace --output /tmp/cache-class.json \
  --logits /tmp/cache-class.f32
python3 tools/expert-source/replay-cache.py /tmp/cache-lru.trace \
  /tmp/cache-class.trace --output /tmp/cache-replay.json
cmp /tmp/cache-lru.f32 /tmp/cache-class.f32
```

Replay adds back the observed policy's history charge and subtracts each
candidate's own charge, so both policies have the same total engine cap. It
requires every recorded hit/miss to match the corresponding runtime policy;
it does not silently produce a report if that check fails. The initial LRU trace
predates the configuration record and needs `--class-lru-overhead 503248`.
New traces carry that value themselves.

The measurements below precede the accepted [prefill allocation and submission
optimizations](atomic-engine-prefill-performance.md). Cache policy and demand
order are preserved; the larger graph arena is included in new budget reports.

## Recorded decision

On the initial Ornith 35B / M4 Max profile, context 512, microbatch 32 and the
same 5,368,709,120-byte engine cap:

- LRU: 68,355 misses, zero hits, 38,824,181,760 logical read bytes.
- ClassLRU: 65,926 misses, 2,429 hits, 37,444,550,656 logical read bytes.
- History: 503,248 admitted bytes; peak engine use 5,368,177,024 bytes.
- Savings: 1,379,631,104 logical bytes, 3.55% overall and 16.84% in decode.
  Prefill had the same 53,931 misses and 30,631,657,472 logical bytes.
- Every captured LRU and ClassLRU decision matched replay. All output logits
  matched bitwise and the complete native state digest remained
  `9755c384a23111fe`.

The second, complete-answer microbatch-128 trace had 30,999 misses and
17,606,770,688 logical bytes; replay predicts no saved reads from ClassLRU.
Its resident-control logits remained bitwise equal. This is why the replacement
policy remains opt-in instead of being promoted to the default.

Instrumented request times were 20.56 seconds LRU and 20.49 seconds ClassLRU;
prefill alone took 17.07 and 17.24 seconds. These are diagnostic single runs,
not a proven speedup: tracing adds work, OS cache state was uncontrolled, and
brief CPU fixture checks overlapped the ClassLRU trace capture. GPU inference
requests and builds did not overlap. No throughput claim is based on these times.

One additional ClassLRU live-control request reused the established prefill
shrink, opposing targets, two shrink/grow cycles and below-floor rejection.
Every logit and final state digest matched fixed-budget LRU. The largest shrink
returned 1,407,909,888 Metal bytes in 53.11 ms; the impossible 1 MiB target left
the session and applied cap intact. Growth allocated no buffers immediately.
Final used bytes equalled the ClassLRU fixed-budget control, 5,367,586,832,
and the raised metadata floor was reported as 4,930,844,384 bytes.

Targeted checks: three new store cases / 18 assertions (trace accounting,
frequency protection with a pinned lease and shrink, history admission floor),
three replay cases (transient staging, scan protection, runtime mismatch), exact
real-trace replay, logits/state comparisons and the single live-control request.
No full suite or aggregate verification ran. Replay does not model asynchronous
I/O, dGPU domains, tail latency or energy, and no cold-SSD benchmark was run.

## Queue after this result

- Available: fixed-budget trace capture/replay, default LRU and opt-in ClassLRU,
  with original weights, accounted history and live-control evidence.
- Next critical performance result: profile and reduce synchronous prefill
  allocation/upload/compute handshakes. Cache replacement cannot remove the
  dominant reads on these prefill traces; a general default switch is rejected.
- Independent work: P05 pressure-policy replay/API and P10A second-architecture
  qualification. Neither is claimed complete by this cache experiment.
- Conditional: asynchronous prefetch/overlap only after component measurements
  justify it and I/O completion/pinning has its own ownership checks.

The direct-read path retains the same miss-staging reservation even when no
physical staging allocation is created. ATET3 capacity, victim selection and
replay therefore retain their existing semantics; `released` counts actual
freed allocations and excludes reserved-only staging. See the
[LFM read profile](atomic-engine-lfm-reads.md).

The [bounded LFM overlap](atomic-engine-lfm-overlap.md) retains future demand
leases across the current GPU submission. ATET3 already records their lifetimes:
the new trace has at most eight pinned experts, exactly matched acquisitions and
releases, and exact runtime LRU replay. The same 6.55% ClassLRU read reduction
remains a replay observation, not an additional fixed-cap latency claim.

The [Qwen overlap qualification](atomic-engine-qwen-overlap.md) also preserves
exact ATET3 ownership replay: 30,999 acquisitions/releases and at most eight
pinned experts. ClassLRU saves no bytes on this completed-answer trace; the
overlap extension does not change the default LRU policy.
