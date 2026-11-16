# Bounded LFM demand-read overlap

## Decision and scope

The direct-read baseline still spends approximately 5.7 seconds reading experts
and 3.8 seconds waiting after GPU submissions. The accepted change reads up to
four subsequent experts on the caller thread between a GPU submit and its
completion wait. These experts are selected from the actual router IDs in the
same projection. There is no prediction, next-layer preview, speculative read,
background I/O thread or second GPU graph in flight.

This is a bounded overlap step toward P07, not qualification of the full preview
or shared-first accelerator. This package originally enabled `lfm2moe` on Metal;
the subsequent [Qwen qualification](atomic-turbine-qwen-overlap.md) extends the
default to `qwen35moe`. CPU, other architectures and non-matrix waves retain sequential demand behavior.
The idle-only experimental `llama_expert_read_overlap_set` API provides an
explicit override; the diagnostic runner exposes `--expert-read-overlap on|off`.
All measurements below use LFM2.5-24B-A2B Q4_K_M, Apple M4 Max, context 512,
microbatch 32 and the existing 200-token prompt / completed 52-token answer.

## Ownership and admission

After submitting the current graph, the executor acquires the next known
experts in ascending demand order. Acquisitions return normal store leases.
When demand reaches an already acquired expert, ownership moves into the
current lease without a second cache lookup or source read. A long-running
expert can span several GPU groups while future leases remain pinned; repeated
flushes do not duplicate those acquisitions.

Before enabling overlap, each projection computes capacity after fixed owners
and one maximum-payload I/O reservation. Its additional lease limit is the
lesser of four and the remaining count of maximum-size expert allocations after
the current group's leases. At most four current and four future experts are
pinned. Every actual allocation still goes through normal store admission and
eviction. There is no extra payload pool or workspace; the small fixed lease
arrays are control storage covered by the existing runtime envelope. The
minimum budget, I/O headroom and cache policy are unchanged. Low headroom
automatically reduces or disables overlap.

Direct CPU/shared Metal writes affect only an unpublished fresh or completed,
unpinned slot. They finish before publication and need not synchronize unrelated
GPU work that owns other pinned buffers. Non-direct buffers retain staged
upload and synchronization. The source still checks identity before and after
an exact-length read and poisons itself on I/O/change failure.

The submitted graph, workspace and referenced leases remain alive throughout
the reads. Both normal completion and an exception during read-ahead wait for
the GPU before unwinding or resetting graph storage. Lease release continues
to synchronize before unpinning. No pending read survives the projection's
callback, so live resize still applies only after all relevant I/O and GPU work
has completed. No asynchronous completion can publish into a reused slot.

## Matched observations

The on/off controls use the same candidate binary; only the diagnostic switch
differs. With a 4 GiB cap and warm OS file cache:

- Complete request: **11.5549 seconds off, 9.2704 seconds on** (19.77% lower).
- Prefill: **6.6504 seconds off, 4.5696 seconds on** (31.29% lower).
- Completion wait: **3.7876 seconds off, 1.3716 seconds on**.
- Source reads: **5.7020 seconds off, 5.8611 seconds on**. Reading did not become
  faster; useful work moved into the submit-to-completion interval.
- The overlap interval acquired 22,965 subsequent experts and read
  44,517,851,136 bytes in 2.9119 seconds of host acquisition time. These are
  completed, used demand acquisitions, not unused speculative traffic.

The new `overlap_acquire` phase is nested in both `compute` and `acquire`.
It measures host work between submit and wait, not the exact intersection with
GPU execution. It must not be added to either parent phase. `wait` excludes
this host work. Summed Metal command-buffer GPU intervals changed from 2.1626
to 2.0570 seconds; they include barriers, are not pure kernel time, and do not
justify a kernel speedup claim.

All existing fixed-cap store counters match exactly: 51,165 misses, zero hits,
99,582,345,216 logical bytes, 937 allocations, 50,228 reuses, 52,905 matmuls and
13,077 submissions. No numerical kernel, matrix padding or batching changes.
Ledger high-water stays at 4,294,966,576 bytes under the 4,294,967,296-byte cap;
the floor remains 3,567,199,296 bytes.

The all-hot control uses an explicitly admitted 18 GiB cap and one warmup
request. The measured requests make zero source reads and 51,165 hits each.
Off/on request times are **5.0259 / 5.0245 seconds**, prefill **3.2893 / 3.2955
seconds**. The future-hit acquisitions cost 6.40 ms in the on case. No material
overhead is visible in this pair; this is not a statistical non-regression claim.

Every on/off and all-hot request matches the saved resident control bitwise:
full logits, 52 committed IDs, EOG 124900 and state `84ac8e66a7115244`.

## Live control and replay

The final default-enabled binary was exercised with ClassLRU and the existing
live schedule: 4096 MiB initial, 3456 during prefill, then 4608 at token 2,
3403 at token 4, rejected 1 MiB at token 6 and 4096 at token 8. The near-floor
shrink releases 1,261,568,000 Metal bytes in 10.824 ms; footprint falls by
1,272,447,720 bytes. Grow allocates no immediate Metal storage. Rejection leaves
applied budget and used bytes unchanged. Two near-floor decode steps still
perform 912 matmuls in 912 submissions. Lifetime high-water remains
4,831,813,008 bytes below the maximum applied cap of 4,831,838,208 bytes.
The complete answer, full logits and state match resident bitwise.

A separate final-binary ATET3 trace exactly reproduces runtime LRU hits and
physical allocation sizes. It contains 51,165 acquisitions and 51,165 releases,
with at most eight distinct pinned experts and none remaining at the end.
ClassLRU still predicts 6.55% fewer logical read bytes; this package does not
change the default policy or claim a new fixed-cap ClassLRU latency result.
The trace request also matches resident logits/state. Trace timings are not
used as the on/off performance comparison.

## Targeted checks and limits

Three Metal executor cases pass 31 assertions: cold/hot/floor exact output and
lease accounting, future leases across repeated groups for one expert, and
source failure after GPU submit with output rejection and complete rollback.
The same cold/hot/floor case on CPU passes 17 assertions and confirms no overlap
is used there. Only the added multi-group case was run after its addition.
The matrix fixture proves zero optional acquisitions at the memory floor; the
real live request separately exercises the near-floor decode path.

Six LFM requests cover the matched 4 GiB controls, two all-hot controls, live
resize and ownership trace; the all-hot controls each include one warmup.
Builds and GPU runs were sequential. Inference and server targets build; no
full suite, aggregate verification, server-request test, new Qwen qualification
or native OS-pressure stress run was performed.

These are individual observations with uncontrolled OS cache and GPU scheduling,
not controlled cold-SSD, latency-tail or power measurements. This package's scope is
LFM/Metal; larger contexts, additional architectures, prediction and asynchronous I/O
workers need separate qualification. Raw reports, hashes and acceptance results
are retained under `p07_lfm_overlap` in `atomic-turbine-evidence.json`.

## Reproduction

Use the original LFM GGUF as `$MODEL`:

```sh
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-turbine-lfm-prefill.txt --tokens 64 \
  --budget-mib 4096 --profile-experts --expert-read-overlap on \
  --output /tmp/lfm-overlap-on.json --logits /tmp/lfm-overlap-on.f32
```

Use `off` and distinct output paths for the control. The all-hot pair uses
`--budget-mib 18432 --warmup 1`; counters are cumulative, so subtract
`memory_before` from `memory_after` for the measured request. For ownership
replay, add `--expert-trace /tmp/lfm-overlap.atet` to a single fixed-budget
request without warmup, then run `tools/expert-source/replay-cache.py` on it.
The live schedule adds `--cache-policy class-lru --prefill-budget-mib 3456
--resize-at 2:4608 --resize-at 4:3403 --resize-at 6:1 --resize-at 8:4096`.
