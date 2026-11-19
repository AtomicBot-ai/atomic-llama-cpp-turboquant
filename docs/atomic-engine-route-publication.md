# Experimental decode route publication

Status: experimental, disabled by default. No new combination is qualified for
`LLAMA_MOE_STREAMING=auto`. Ordinary streaming and the existing Flash predictor
retain their defaults. This work does not change prefill-window policy.

## Internal switches

- `ATOMIC_MOE_PUBLISH=event`: publish actual selected routes from Metal before the
  existing CPU MoE split. `shared` additionally orders the existing shared branch
  between publication and that split in the Qwen4Exp/Qwen35MoE model builders.
- `ATOMIC_MOE_PUBLISH_LAYER=0|all`: the default is the bounded layer-zero prototype.
- `ATOMIC_MOE_VERIFY_OVERLAP=1`: independently enable publication at all
  eligible layers during multi-row Qwen35MoE DFlash2 target verification.
  The existing shared branch runs after publication. Early reads use a charged
  buffer for up to four experts. CPU MoE takes completed slices and attaches
  pending slices to the demand batch. Queued slices move to demand priority.
  Use `ATOMIC_MOE_VERIFY_DEDUPE=0` with overlap enabled to compare the original
  policy of issuing separate demand reads for pending slices.
  Single-row decode and prefill use their existing execution paths.
- `ATOMIC_MOE_QWEN_PREDICTOR=1`: opt in to the existing early predictor for qualified
  Qwen35MoE layers, with four candidates and one decode row only. Flash keeps its
  existing eight-candidate policy. Selection/router bias and MTP layers are denied.
- These switches require budgeted Metal streaming with the automatic combined
  executor. `LLAMA_MOE_STREAMING=off` remains the ordinary control path.

No public llama/ggml ABI changes are introduced. The private Metal v1 capability
uses a marked contiguous I32 copy and an opaque channel. The GPU copies up to
32 routes for the single-row experiment or 64 for Qwen35MoE verification to a
16 KiB shared payload, signals its sequence, and orders subsequent
shared work after that signal. A task on the already-accounted demand pool waits
for publication. There is no new thread or GPU wait on CPU disk-read completion.
The channel reserves 64 KiB for payload/host bookkeeping plus its C++ service and
resident-group metadata. Failed admission retains the existing executor.

For the single-row experiment, actual demand uses the existing store, compact buffers, priority queue, and
leases. Resident triples are pinned before misses are admitted and execute first.
The consumer validates the published selection against the normal CPU input.
Leases remain live through private Metal completion. Cancellation drains the
service and its batches; stale sequences fail explicitly; after a latched error,
later route markers cannot start more I/O. Graph invalidation and resize retain
the existing synchronized executor lifecycle.

The main CPU MoE boundary is intentionally retained. This feature advances I/O;
it does not claim to remove all CPU/GPU fences or implement an entirely GPU-owned
MoE scheduler. Kernels, source quantization, route weights and reduction order are
unchanged. Off-path structural cost is limited to private metadata/pointer fields
and marker checks in Metal encoding/fusion; there are no new off-path GPU commands,
workers, payload allocations, or timestamps. That structural cost requires a
before/after performance control, not just correctness tests.

## Focused validation

Build `llama-server` and `test-expert-executor`; do not run an aggregate gate.
Select the publication tests explicitly, including mixed quantization, cold/hot
execution, incomplete groups, multirow fallback, source failure, later operations
after failure, cancellation, stale sequences, and minimum-budget resize.

`tools/expert-source/qualify-route-publication.py` runs exactly one requested
case. Supply an explicit protocol JSON, output directory, and binary. The protocol
contains model paths and hashes, fixed control token IDs, and the three preset
texts. It uses 246 input / 64 output tokens, 18 GiB, context 1024, microbatch 8,
ready threshold four, and a 15/8 warmup. It refuses a known concurrent inference
process. It does not flush the macOS file cache or run a model/budget matrix.

Diagnostic server builds additionally capture every full logit vector, slot
roundtrips, and two identical continuations from a restored slot. Timing builds
must omit that instrumentation. Physical process reads and logical expert reads
are separate metrics. Publication-to-read timing begins when the CPU observes
the GPU event; it is not an exact GPU timestamp. Early-I/O overlap is the union of
read intervals before entry into CPU MoE, not the sum of worker times or proof of
whole-request speedup.

Qualification requires bitwise correctness, at least 5% median decode improvement
above A/A noise on every preset text, and no reproducible all-hot regression above
2%. A single diagnostic request cannot qualify a feature. The full five-model /
four-budget acceptance matrix remains separately approval-gated.

## Recorded outcome: not promoted

The local evidence directory is `/private/tmp/moe-route-publication/`. Its
`protocol.json` freezes model/shard hashes, control token IDs, and three distinct
texts before measurements. `before/` preserves the starting server and source
copies; `source/` and `build/` contain an isolated source/build snapshot. Isolation
avoided modifying another task's concurrent prefill work or rebuilding against
partially updated shared sources. Each case under `runs/` records its binary hash,
command, inputs, output IDs, memory high water, and process disk-read counters.
These are local artifacts, not committed model files or a portable performance
guarantee. The macOS file cache was not flushed.

The existing automatic path stays selected. Both new mechanisms remain explicit
experiments: short-request hot controls reproduced regressions, and no candidate
completed the positive performance qualification required for promotion.

### Correctness and lifecycle

Qwen 35B and Ornith 35B were compared with the new flags absent, publication plus
shared overlap, predictor alone, and both together. Full 64-token logit vectors,
output IDs, saved slot bytes, and restored-slot 16-token continuations matched
bitwise within each model. The final Qwen shared case was repeated after fixing a
graph fallback hazard: shared ordering must depend on a marker emitted by the
current graph, not a descriptor left over from an earlier graph.

Flash ordinary decode matched bitwise with and without shared publication. The
DFlash pair also matched all logits, tokens, slot bytes, restored continuation,
and the same 19 recorded rollback events. Both requests accepted 41 of 110 draft
tokens. The multirow verify path emitted zero new route publications, as intended.
This establishes correctness of that fallback; it does not establish its speed.
The diagnostic DFlash runs measured 17.59 versus 11.70 tokens/s with different
physical decode reads (9.43 versus 14.20 GB), so performance is unresolved rather
than reported as unchanged.

Ornith 122B's layer-zero event prototype and predictor-alone case matched the
control's full logits, output tokens and slot. The first all-layer prototype also
matched, but its grouping was subsequently corrected to preserve resident-first
execution; its timing is not evidence for the final implementation. The final
all-layer shared case was then rerun: logits, tokens and slot remained bitwise
equal to the control, and its restored continuation matched the earlier all-layer
trace. This final case hid 294.1 ms of read intervals, but still spent 11.39 s
waiting for demand batches and measured only 3.45 tokens/s diagnostically. It
does not demonstrate a net decode gain. The four-model comparison results and
individual hot pairs are summarized in `evidence-summary.json` in the evidence
directory. Final source/build hashes are in `final-source-manifest.json`.

Focused tests passed for Metal event ordering/reuse/cancellation/failure (39
assertions), mixed-quantization published demand and lifecycle (135 assertions),
controlled delayed/stale service delivery (6 assertions), existing Flash FP32
queue/resize/failure (96 assertions), existing Qwen mixed-quantization queue (61
assertions), and ordinary execution with acceleration off (5 assertions).
The lifecycle fixture covers cold/hot and partial hits, incomplete groups,
multirow fallback, truncated-source failure, later graph execution after that
failure, lease release, and minimum-budget shrink/grow. Early failed fixture
attempts remain in the logs; corrected targeted reruns passed. These are selected
cases, not a full suite or a claim of every possible scheduling interleaving.

### Regression evidence and its limits

Three clean original/current pairs on Qwen 35B, with new flags absent, measured
33.92/38.42, 37.61/39.52, and 27.69/38.17 tokens/s. Tokens and accounted memory high
water matched. No regression was observed in this control, but baseline variance
and file-cache effects are too large to claim a default-path speedup.

The hot short-request control restores a fixed 246-token slot, processes one new
input token, then measures one decode interval (`n_predict=2`), with eight warmups
and 32 samples per process. Separate diagnostic runs confirmed zero demand reads
and 40 MoE calls in measured intervals. Three clean process pairs gave:

- Four-candidate predictor: 21.90/26.94, 23.29/26.15, 23.20/24.64 ms
  (control/experiment); pair regressions 23.0%, 12.3%, and 6.2%.
- Publication plus shared overlap: 23.28/28.80, 21.03/29.16,
  23.69/28.34 ms; pair regressions 23.7%, 38.7%, and 19.6%.

This measures a real short-request regression, including graph preparation. It
does **not** isolate steady-state decode: the existing phase-dependent graph key
changes on prompt-to-decode transitions when the new capability is enabled.
The server reused 107 graphs in the first control process versus 28 with either
experiment. Consequently these percentages must not be attributed solely to GPU
events or predictor arithmetic. The predictor diagnostic additionally read
13,737,984 unused speculative bytes per measured interval despite zero demand
reads. Its hot overhead is not purely graph setup either.

The early Ornith 122B layer-zero diagnostic hid only 5.35 ms of read time across
63 decode intervals, against 10.50 s of baseline demand waiting. The all-layer
prototype hid about 265 ms but predates the resident-order fix. Neither number
proves whole-request acceleration. Diagnostic single-request gains on other
models are likewise not qualification results.

Because the hot short-request regressions already fail the acceptance boundary,
the three-text, three-pair A/A and five-pair A/B positive qualification series was
not run. A steady-state all-hot series, complete allocation/dispatch performance
qualification, and broad configuration coverage also remain open. No tuning was
performed on the frozen texts. No new combination is enabled in `auto`, and the
full five-model / four-budget matrix was not run.

### Reproduction

Build only `llama-server` and `test-expert-executor` from the recorded source
snapshot. `build-route-publication-probe.py --source SOURCE --build BUILD
--output PROBE` creates a separate instrumented binary. Run a single case with
`qualify-route-publication.py --protocol PROTOCOL --output NEW_DIRECTORY
--binary SERVER --model qwen35 --mode shared --layers all`; use `--diag` only
with the probe binary. In this harness `--mode off` means existing `auto` with
the **new** experimental flags absent, not `LLAMA_MOE_STREAMING=off`.

Use `check-route-publication-hot.py` with the same explicit protocol/output/binary
arguments for the restored-slot control. Diagnostics and clean timing must be
separate processes. Do not run these commands concurrently with inference or
compilation. The scripts refuse known inference processes but cannot certify an
otherwise idle machine. The private marked-CONT and one-outstanding-publication
contract must be rechecked against scheduler/fusion changes on upstream updates.
