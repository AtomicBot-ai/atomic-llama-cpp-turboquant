# Exact routing and shared-FFN scheduling probe

## Decision

Close this optimization-wave hypothesis for the measured Qwen profile without
changing the active inference implementation. The prototype creates a useful
CPU/GPU scheduling opportunity and preserves exact outputs, but the shared branch
is too short to establish a worthwhile speedup in these observations. The probe
has been removed from active source; the prior implementation is restored byte
for byte. No shared-schedule flag or context parameter is shipped.

The [archived patch](atomic-turbine-shared-schedule-probe.patch) and structured
`shared_schedule_probe` section of [the evidence record](atomic-turbine-evidence.json)
retain the implementation, configuration, hashes and results. This concludes the
three-direction optimization wave: prepared single-expert maps and the qualified
Qwen route index remain enabled; the earlier gate/up-read experiment remains off.
This is not completion of the overall engine plan or an exhaustive optimization
of every architecture, context size or decode schedule.

## Mechanism evaluated

The normal graph computes exact routing and routed projections before the shared
FFN. Merely moving shared FFN earlier puts it in the same GPU split as routing;
copying IDs to CPU then waits for that entire split. The prototype instead:

1. Computes the original router, top-k IDs and routing weights without changing
   their operators or numerical backend.
2. Expands a scalar CPU custom operation consuming the actual IDs. This creates a
   scheduler boundary; it neither predicts nor recomputes routing.
3. Expands the original independent shared-FFN graph before routed projections.
4. Uses the existing scheduler input-prefetch/event path to copy the routed
   activation before submitting shared FFN. The ordinary CPU expert callback can
   then read demanded weights while that independent GPU split is unfinished.

No background I/O thread, extra cache, optional expert leases or changed demand
order is introduced. All weight reads remain normal validated store acquisitions;
all expert GPU leases retain their existing completion boundaries. The original
shared/routed sum operand order is preserved. Graph buffers are allocated through
the existing common memory ledger before evaluation. The prototype is restricted
to fully offloaded, budgeted Qwen3.5 MoE, default context type, and microbatches with
at least 32 rows. Smaller inputs, including decode, retain the original graph.

The diagnostic modes are `off`, `serial` and `overlap`. The latter two build the
same graph and allocate the same buffers; `serial` explicitly waits for shared
GPU completion before ordinary expert acquisition. This separates the scheduling
opportunity from graph reordering and allocation changes. The mode is immutable
for a context lifetime. Only the archived prototype exposes these controls.

## Focused observations

Hardware is the existing Apple M4 Max / 48 GiB setup. The model is Ornith 35B,
Qwen3.5 MoE, embedding width 2048 and shared-FFN width 512. Configuration: context
512, microbatch 32, 190 prompt tokens, 16 committed tokens, 5 GiB common cap, LRU,
prepared single-expert maps and the default Qwen route index. Preview and the
previous gate/up early-read experiment are off. Builds and GPU runs are sequential.

Four directly relevant requests ran in this order:

- Initial overlap probe: request **10.4053 s**, prefill **8.6028 s**,
  source-transfer service **7.6200 s**. This identifies the mechanism and checks
  accuracy; its large I/O time makes it unsuitable as a steady latency comparison.
- Original schedule: request **6.2620 s**, prefill **4.6917 s**,
  source transfer **3.4962 s**.
- Shared-first serial control: request **6.0590 s**, prefill **4.4109 s**,
  source transfer **3.2783 s**.
- Shared-first overlap: request **6.0783 s**, prefill **4.4660 s**,
  source transfer **3.3291 s**.

Every request matches all saved resident logits byte for byte, all committed IDs,
and serialized state `9755c384a23111fe`. Every request performs exactly
**38,824,181,760 logical read bytes**, **87,072 private matmuls** and **17,367 private
expert submissions**. Those submission counts exclude the added main-scheduler
boundaries. No saved output or token subset is substituted for full comparison.

In the overlap runs, shared GPU work is incomplete before the first acquisition
in **194/195** and **195/195** windows, and after it in **130/195** and **178/195**.
In the serial control these counts are zero. Completed GPU timings are available
for all 195 windows in every shared-first run. This shows that the CPU no longer
waits for shared completion before acquiring experts; it does not measure the
intersection of actual GPU execution and host I/O intervals.

Shared GPU command service totals **28.48–30.27 ms/request**, approximately
**0.5%** of the roughly six-second request. That is a small ceiling even before
additional boundary/copy costs or UMA contention. In the matched serial/overlap
pair, the 19.3 ms request difference is smaller than the 50.8 ms source-transfer
variation. Neither a speedup nor a reliable regression is established. Comparing
the original schedule to overlap also cannot isolate benefit: changing I/O time
accounts for most of that difference. Further repetition or a larger implementation
is not justified by this probe's measured ceiling.

The prototype floor is **4,933,557,314 bytes**, versus **4,933,622,850** for the
original schedule; the 65,536-byte difference comes from graph-buffer layout.
Prototype peak charge is **5,368,529,186 <= 5,368,709,120 bytes**; the original peak
is **5,368,594,722**. No cap is raised or owner excluded for the comparison. The
incidental floor decrease is not a separately qualified feature.

## Reproduction and limits

The active source is the pre-probe implementation. To reproduce deliberately in
a separate checkout with the recorded base-source hashes, apply the archived patch
and build only `llama-expert-infer`. Set `QWEN_GGUF` to the local checkpoint and use
different output paths for each mode:

```sh
git apply docs/atomic-turbine-shared-schedule-probe.patch
cmake --build build-metal --target llama-expert-infer -j 8
build-metal/bin/llama-expert-infer --model "$QWEN_GGUF" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest --profile-experts \
  --prompt-file docs/atomic-turbine-prefill.txt --tokens 16 --budget-mib 5120 \
  --expert-shared-schedule overlap \
  --output /tmp/shared-overlap.json --logits /tmp/shared-overlap.f32
```

The probe executable and raw reports are retained locally under
`/tmp/atomic-turbine-shared-probe-infer` and
`/tmp/atomic-turbine-qwen-overlap-shared-{overlap1,off1,serial1,overlap2}.*`.
The patch passes an applicability check against the restored sources. The affected
inference target is rebuilt after restoration. No full suite or aggregate verify
ran; no unrelated tests ran. The four real-model comparisons are the targeted
runtime checks for this research decision. Live resize, injected read failures,
server requests, callback-driven execution, decode overlap, larger shared branches
and other architectures were not qualified for this discarded prototype.

Buffered reads include OS page-cache and page-fault effects; logical bytes are not
physical SSD traffic. These are individual partially cached observations, not
statistical latency or controlled cold-storage qualification. A different model
with a substantially larger shared branch could justify a new measured hypothesis;
no general rejection of shared-FFN overlap is implied.
