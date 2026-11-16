# P07 Qwen preview/prefetch cost follow-up

The subsequent [optimization audit](atomic-turbine-optimization-audit.md) binds
preview to executed graph geometry and removes the excess reads recorded here.
Historical timings and workspace sizes below describe the preceding version.

## Decision and scope

Keep preview/prefetch **off by default**. Retain two measured improvements in the
opt-in path: matrix-tail scheduling and a specialized four-row CPU predictor.
They reduce internal work substantially, but the final crossed on/off comparison
does not demonstrate end-to-end acceleration over known-route overlap alone.
This does not close the P07 activation gate.

The main inference graph, actual routing, expert kernels and original GGUF bytes
remain unchanged. This package changes executor scheduling, the independent
preview dot kernel and diagnostic counters. It does not change cache policy,
source/storage validation, thread count or backend placement.

Current experimental modes are `off` (0), `preview` (1, prediction only),
`prefetch` (2, optimized matrix-tail path) and `prefetch-first` (3, diagnostic
control using the original first-window placement and single-row predictor).
All prior architecture, LRU, idle-control and trace-exclusion restrictions apply.
The [original runtime report](atomic-turbine-qwen-preview-runtime.md) records the
preceding implementation; its mode-2 behavior is now the mode-3 control.

## What profiling established

The initial instrumented 190-token/16-output Qwen request had 819 prediction
windows. Dot products cost 95.80 ms; Top-4 ranking, residency queries and the warm
source guard cost 0.98, 1.70 and 0.81 ms respectively. Optimizing sorting or cache
lookups was not justified by this profile.

In prefill, 195 of 234 preview windows also acquired subsequent known-demand
experts. Predictor plus speculative host work took 185.87 ms in those prefill
windows, against 65.86 ms of reported GPU command service. In decode, 585 windows
performed 335.71 ms of preview/read work against only 9.92 ms of GPU service.
These totals identify poor placement opportunities; they are not timestamped
CPU/GPU intersections or exact critical-path savings.

The optimized mode therefore skips non-matrix windows, including decode and the
30-row final microbatch in this request. In matrix projections it waits for a
submitted group containing the largest actual routed expert ID. There are then
no subsequent known experts to acquire, so optional reads do not compete with
known-route lookahead. A final expert can span several waves; the criterion is
absence of later expert reads, not necessarily the last GPU submit itself.
Prediction and all optional leases still finish inside the same projection
callback. The four-expert/12-slice bound and conservative mandatory headroom are
unchanged. Skip counters distinguish short windows, budget limits and warm data.

## Predictor kernel and rejected attempt

The original lazy predictor reread the router for each input row. A focused
benchmark on the captured first eight Qwen layer inputs showed that traversing
the router once for four rows preserved all scores bitwise and reduced isolated
median time from about 88 to 34 ms. Those benchmark times are not request times.

The first generic integration was rejected: runtime dot time rose to roughly
163 ms despite the isolated result. Its dynamic row-count branch remained inside
the expert loop. The final helper specializes one and four rows at compile time,
placing dispatch outside that loop and preserving each row's original eight-lane
accumulation, scalar reduction and ascending-ID tie order. The isolated helper
check also preserved every compared score. No global floating-point flags changed.

In the actual matrix-tail workload, scalar prediction cost 56.31 ms for 1,167
rows. The final helper costs 28.49, 29.12 and 29.21 ms, even though bounded
four-row blocks evaluate 1,412 rows. Selected prefetch count and bytes remain the
same. The helper resides in `src/llama-expert-preview-dot.h`; unused score lanes
are never ranked. Finite-result checks precede Top-4 selection.

Four score rows add 12,288 scratch bytes. Total admitted predictor workspace is
36,904 bytes, raising the ubatch-32 floor to 4,933,637,224 bytes. No router copy,
thread, GPU graph or model payload is added. Temporary fixed-size dot accumulators
remain covered by the existing runtime stack/control envelope.

## Runtime results

All comparison runs use the same Ornith 1.5 35B A3B Q4_K/IQ4_XS checkpoint, M4 Max,
context 512, microbatch 32, 5 GiB managed budget, one sequence, greedy sampling,
LRU and the already enabled known-route read/GPU overlap.

Optimized prefetch runs took 7.265906416, 7.418778250 and 7.527001208 seconds.
Baseline runs took 7.386953125, 7.069803916 and 7.346111542 seconds, in on/off,
off/on, on/off order. Median request time is **7.418778250 vs 7.346111542 s**
(0.99% slower); median prefill is 5.854666792 vs 5.824917667 s. Individual pairs
change direction. These observations fail the speedup gate; they do not establish
a general statistically significant regression either.

The mechanism-level improvements are repeatable in the measured counters:

- Preview windows decrease from 819 to 195; 624 short windows are skipped and
  competing known-read windows decrease from 195 to zero.
- Speculative read bytes decrease from 5,582,094,336 to 1,329,070,080. The latter
  is 780 complete future-expert bundles, not a row-scaled union.
- Total logical reads decrease from the original candidate's 39,255,277,568 to
  38,858,260,480 bytes, versus baseline 38,824,181,760. Extra I/O falls from 1.11%
  to 0.0878%; it is not eliminated.
- Final host preview/read work is about 121–122 ms per request, against about
  36 ms of reported GPU service in its selected windows. Lower work and more
  cache hits still do not by themselves prove faster requests.

The final all-hot control, after one warmup at 24 GiB, performs zero prediction
rows, zero speculative reads and zero demand reads: 195 warm matrix skips and
624 short-window skips. Its elapsed time is not an equivalence or speedup claim.

## Follow-up: useful placement ceiling

A read-only follow-up on the existing 32-row capture binds 190 matrix-source
pairs to an executed target layer. Offline Top-4 predictions hit the actual
expert union in all 760 selected cases, but only 46 lie among that target's first
four ascending expert IDs. At the measured four-wave grouping, these are the
initial acquisitions before the first GPU submit; subsequent known experts can
already use the accepted lookahead path. The corresponding upper bound is
78,381,056 of 1,294,991,360 predicted bytes, or 6.05%. It assumes the prefetched
entries survive until use and uses offline score reductions; it is not measured
critical-path savings or a general coverage guarantee.

Five additional matrix sources target a final layer pruned from their prefill
graphs. Their four-bundle payload totals 34,078,720 bytes, equal to the final
candidate's observed extra reads. This identifies avoidable absent-target work.
It does not by itself prove that another predictor or selection policy would
accelerate inference. A future runtime change must bind speculation to the
executed target graph and demonstrate benefit for otherwise-blocking demand.
No runtime behavior or default changed in this follow-up.

## Accuracy, budget and targeted validation

All emitted logits, committed IDs and final state match resident controls
bitwise across every long-prompt experiment, including rejected performance
variants. The long prompt checks 16 outputs and state `9755c384a23111fe`.
The final ubatch-128 complete-answer control also matches output token 23,
EOG 248046 and state `d8bd6c5d1cf32b92`.

The final fixed 5 GiB long-prompt path peaks at 5,368,609,096 of 5,368,709,120
managed bytes. The live control shrinks during prefill to 4,706 MiB, exercises
latest-request-wins, grows to 6,144 MiB, shrinks to 5,120/4,706 MiB, rejects 1 MiB
without changing cap/usage, and grows again. It preserves exact logits/state.
Its largest observed peak is 6,442,168,344 of the largest 6,442,450,944-byte cap;
the later near-floor shrink releases 1,283,077,408 ledger bytes in 39.80 ms.
There are 39 predictor generations, 156 budget skips and 624 short-window skips.
Decode remains intentionally ineligible after growth. Re-enabling optional work
for a matrix projection after growth is covered by the focused fixture.

Only the changed `runtime preview.*` executor case ran on CPU and Metal, with
26 assertions per backend. It covers charge/release, unchanged output, nested
profile counters, global bundle bounds, noncompeting tail windows, short/warm
skips, floor rejection, shrink/grow and failure of a separate optional GGUF shard
after GPU submission. All leases drain and the failed output is cleared. CPU
uses the first-window control for positive prefetch coverage and also checks the
optimized short-window rejection; Metal exercises the optimized matrix path.
The case was rerun after replacing the rejected generic dot loop. No full suite,
aggregate verify or unrelated test file ran.

The work used 17 model invocations, 17 measured requests and one warmup: focused
profiling, the placement experiment, rejected generic-loop controls, final crossed
pairs, and final completion/live/hot qualification. Inference, executor-test and
server targets build; server requests and native memory-pressure events were not
qualified here. Counters accumulate across requests, so before/after snapshots
must be subtracted.

Temporary disk exhaustion interrupted a redundant binary copy during exploration,
not an inference result. Original diagnostic paths and contents were preserved:
66 outputs were losslessly checked and restored, with duplicate payloads now
stored as independent APFS copy-on-write clones. About 1 GiB of headroom was
restored before final comparisons. File-cache state, background activity and
physical SSD traffic remain uncontrolled; no cold-SSD, tail-latency, energy or
broad checkpoint/shape claim follows from these timings.

## Reproduction and evidence

Use the existing budgeted inference runner with `--expert-preview prefetch`,
`--profile-experts`, `--state-digest` and a logits output. Compare with `off` on
the same binary; use `prefetch-first` only to study the old placement. The focused
checks remain:

```sh
build-metal/bin/test-expert-executor --backend CPU 'runtime preview.*'
build-metal/bin/test-expert-executor --backend MTL0 'runtime preview.*'
```

Raw local captures use `/tmp/atomic-turbine-qwen-overlap-cost-*`. Results, rejected
variants, source/binary hashes and limitations are recorded under
`p07_preview_cost` in `atomic-turbine-evidence.json`. Existing evidence snapshots
are retained. Further default activation requires a reproducible improvement in
whole-request latency, not only the reduced costs documented here.
