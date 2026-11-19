# P06: synchronous prefill profiling

## L2 scope

ClassLRU reduced decode reads but did not remove any prefill misses on either
qualified trace. Microbatch-32 prefill accounts for about 17 seconds of the
20-second request. Measure its components before changing execution.

1. Add opt-in wall-clock phase counters around admission/eviction, physical
   allocation, source reads, upload, graph setup, activation packing/copy,
   backend computation/readback and lease release.
2. Capture the existing 190-token, microbatch-32, 5 GiB LRU workload as a baseline.
3. Select the dominant measured cost and implement a bounded optimization.
4. Check affected ownership/shape cases only, then compare full native logits,
   state, ledger peak and end-to-end time against the same workload.
5. Confirm the changed mechanism on the second microbatch profile and, when
   allocation ownership changes, the established live resize scenario.

No model conversion, router changes, precision changes, or unbounded fallback.
Reservation precedes allocation, aliases have one owner, and live GPU leases
prevent reuse. GPU request measurements run sequentially without concurrent
builds or tests. Timers are disabled by default; their cumulative values are
queried only while idle. Store phases are nested in acquire, which is nested
in projection: these must not be summed together as independent times.

The minimal controls are the existing saved logits/state and a same-cap native
request. No full suite, aggregate verification, or broad performance matrix is
part of this package. Buffered reads do not establish cold-SSD performance.

## Profile and selected changes

The unchanged baseline took 16.8974 seconds for prefill and 20.4568 seconds for
the complete 16-token diagnostic prefix. Expert projections accounted for
16.6463 seconds of prefill. Their non-overlapping principal costs were:

- Backend graph submission/completion: 10.7168 seconds, 72,672 submissions.
- Acquisition: 5.4384 seconds, including 1.1600 seconds admission/eviction,
  2.3295 seconds allocation/initialization, 1.5723 seconds source reads and
  0.3618 seconds upload/synchronization.
- Gather/scatter, input copy, output readback and graph construction were small
  compared with those two costs. These are wall times, not GPU kernel timestamps.

The first experiment disabled the extra Metal command encoder on the private
executor backend. It preserved all logits/state but regressed prefill to
18.2139 seconds, with 12.0153 seconds in backend compute. It was rejected and
fully reverted; the delivered code does not change Metal encoder settings.

Two optimizations are enabled in the existing opt-in streaming path:

1. **Reuse the chosen victim's allocation.** When admission requires eviction,
   the store removes the policy-selected unpinned entry from its index. If its
   admitted backend/arena sizes match, it keeps their tickets and rebinds tensor
   metadata for the new expert. Read staging is still admitted separately.
   The full buffer is cleared, the original bytes are read and uploaded, and
   synchronization precedes publication with a new generation. No live lease can
   be reused. A failed read destroys the unpublished allocation and returns its
   charges. Incompatible sizes or insufficient staging capacity use the previous
   bounded free-and-allocate path. There is no detached pool retaining evicted
   buffers: shrink/trim still destroys physical allocations. The later
   [LFM profile](atomic-engine-lfm-performance.md) extends this to larger victims,
   preserving their full physical charges; the measurements here precede it.
2. **Batch vector micrographs for one expert.** The short prefill tail previously
   submitted one graph per selected row. It now gathers at most 32 rows of one
   expert and submits independent one-row `MUL_MAT_ID` nodes in one graph. Each
   node still selects the original vector kernel; matrix waves keep their existing
   32-row padding. Inputs and outputs have distinct aligned workspace regions,
   including backend output scratch/map bytes. One lease remains pinned until
   the whole graph completes. Host/backend workspaces have the same sizes; the
   larger, pre-admitted graph arena costs 27,280 extra bytes on Metal.

Allocation reuse alone reduced prefill to 13.5400 seconds. Allocation time fell
to 0.2081 seconds and admission/eviction to 0.2814 seconds; source read time was
1.6206 seconds. The complete implementation reduced prefill to 11.7628 seconds
and total request time to 14.2806 seconds: 30.4% and 30.2% less wall time than
the same profiled baseline in these single local runs. This is workload-specific
evidence, not a statistical speed or cold-SSD guarantee.

Final prefill had the same 53,955 misses, 30,645,288,960 logical read bytes and
72,672 matmul operations. Physical buffer allocations fell from one per miss to
743 allocations plus 53,212 reuses; GPU submissions fell to 53,955. Backend compute
time fell to 8.5680 seconds. The prefill phase boundary is taken from the runner,
not inferred from `n_tokens == 1`: a one-token internal projection can occur
inside prefill, explaining the small count difference from the older trace report.

The cap remains 5,368,709,120 bytes. Final LRU ledger peak was 5,368,291,248 bytes,
minimum 4,930,368,416 bytes, and observed process footprint peak 4,099,804,688
bytes. All logits were bitwise equal to the baseline and the native serialized
state digest stayed `9755c384a23111fe` through each accepted intermediate change.

The second existing profile, complete answer at microbatch 128, also preserved
all logits and state, returning `8` and EOG. It took 7.1259 seconds prefill and
7.2937 seconds total, versus the historical 8.6100/8.8363-second control. That
historical timing comparison is less controlled than the paired microbatch-32
profile. Its 30,999 misses used 559 physical allocations and 30,440 reuses;
31,899 GPU submissions equalled its matmul count, as expected when matrix waves
already group rows. Ledger peak was 5,368,616,752 bytes under the same cap.

Live control with ClassLRU also retained all logits and the same state digest.
The established prefill shrink, opposing targets, two shrink/grow cycles and
below-floor rejection still behaved correctly. The largest shrink returned
1,407,909,888 Metal bytes in 51.83 ms; all growth operations immediately allocated
zero buffers, and the rejected 1 MiB target released nothing. Final managed use
was 5,367,614,112 bytes: exactly the earlier ClassLRU control plus 27,280 bytes of
new graph metadata. The new ClassLRU floor is 4,930,871,664 bytes. The lifetime
peak of 6,442,353,744 bytes fits the temporarily applied 6 GiB cap. No detached
reuse pool prevented physical release.

## Reproduce the targeted profile

Use the existing Metal build configuration and set `MODEL` to the original GGUF.
Profiling is disabled by default; `--profile-experts` enables cumulative
nanosecond counters at the prefill and final snapshots. Do not add nested store
phases to `acquire`, or add `acquire` to `projection`, when computing totals.

```sh
cmake --build build-metal --target llama-expert-infer -j4
build-metal/bin/llama-expert-infer --model "$MODEL" --budget-mib 5120 \
  --prompt-file docs/atomic-engine-prefill.txt --tokens 16 --ctx 512 --ubatch 32 \
  --profile-experts --state-digest --output /tmp/prefill-profile.json \
  --logits /tmp/prefill-profile.f32
```

The extension API exposes idle-only `llama_expert_profile_set/get`. Enabling
resets the counters; disabling removes timestamp sampling. Diagnostic streaming
counters distinguish logical matmuls, graph submissions, physical allocations
and completed buffer reuses.

Targeted checks cover mixed F32/F16 layout reuse, original packed bytes, pinned
victims, generation changes, failed-read publication/charge rollback, delayed
completion, shrink/regrowth, and profiling enable/reset/disable. Routed CPU and
Metal cases cover shared/strided inputs, short vector tails, repeated IDs crossing
the 32-row batch boundary, and unchanged padded matrix waves. No full suite ran.

Executed cases: three store cases / 26 assertions, one profiling case / six
assertions, and two routing cases on CPU and Metal / 80 assertions per backend.
The real runs were sequential and had no overlapping builds or test processes.
Raw reports and hashes for baseline, rejected encoder setting, reuse-only,
combined optimization, microbatch 128 and live control are retained in the
evidence JSON. Older baselines are preserved rather than overwritten.

Remaining compute wall time is still dominant. This experiment does not separate
GPU kernel execution from command submission/wait overhead, and does not justify
asynchronous I/O or multi-expert batching yet. Any next optimization must retain
the same kernel policy and account for every additional live workspace/lease.
