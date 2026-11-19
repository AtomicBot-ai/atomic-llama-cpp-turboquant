# Qwen graph eligibility and row-copy optimization

The first proposed follow-up is now delivered and qualified in
[known-expert Metal dispatch reduction](atomic-engine-single-expert-map.md).
The second is delivered in the [bounded route index](atomic-engine-route-index.md),
with Qwen activation and a retained scan default for LFM.
The third was evaluated through [exact gate/up early reads](atomic-engine-early-reads.md)
and [early IDs with shared-FFN scheduling](atomic-engine-shared-schedule.md).
Neither qualifies for default activation on the measured profiles; the latter
prototype is archived rather than retained in active source. These three choices
are now evaluated for this optimization wave.
The audit and timings below describe the preceding implementation.

## Delivered changes

Preview is now bound to the graph about to execute, including reused graphs.
Only executor-owned custom operations are inspected, using the in-tree custom-op
layout and opaque userdata comparison. No graph nodes, edges, routes or arithmetic
are changed. All three target projections must occur uniquely after the source
and have the same row count. Missing, duplicate, zero-row and narrowed targets
fail closed; an unbound preview also fails closed. Bindings retain only scalar
order/geometry, never graph pointers, and reset before every evaluation.

The captured last-layer omission was an execution omission, not necessarily a
missing graph node: actual Qwen graphs retain zero-row or narrowed final-layer
operations. Presence alone would therefore not have fixed the unwanted reads.

The bounded graph metadata adds 20,480 admitted scratch bytes. Preview workspace
is now 57,384 bytes and the measured Qwen ubatch-32 floor is 4,933,657,704 bytes.
The default-off path allocates no preview workspace and does not scan the graph.

Contiguous expert input rows now use one memcpy instead of a per-float loop.
Strided inputs retain the original copy loop. This changes no floating-point
operation, input order, workspace size, lease lifetime or cache policy.

## Targeted observations

The graph-only Qwen request removes exactly 34,078,720 unnecessary logical read
bytes. Total reads are 38,824,181,760 bytes, equal to the saved no-preview control.
There are 190 preview windows instead of 195, 760 prefetched expert bundles,
1,294,991,360 speculative bytes, six shape skips and zero competing windows.
Five shape skips replace matrix windows; the sixth replaces a short-window skip.
Graph binding costs 0.540 ms across the entire request.

An isolated crossed-order copy benchmark uses widths 512/2048 and contiguous or
stride-8 rows. Bytes match. For width 2048, four contiguous-copy trials take
34.17–35.00 ms versus 119.92–122.80 ms for the original loop; the strided fallback
has approximately unchanged cost. This is a copy benchmark, not inference speed.

In the model, gather time decreases from 161.99 to 42.21 ms after the copy change.
Both successful requests preserve all saved resident logits, 16 committed token
IDs and serialized state `9755c384a23111fe` bitwise. The managed high-water is
5,368,629,576 bytes under the 5,368,709,120-byte cap. Read bytes, prefetch counts,
matmul count and submission count remain unchanged by the copy optimization.

Whole-request times are 7.3996 s before the copy change and 10.6850 s after it.
Source-transfer time increases from 2.4872 to 6.9187 s while GPU service stays
near 2.77/2.74 s. The local gather reduction is established; whole-request
acceleration is **not**. These sequential runs are not a controlled I/O experiment
and do not identify the cause of the changed source-transfer time. Preview stays
off by default; its activation gate remains open.

The updated preview fixture executes source and target projections in the same
graph. Its 33 assertions pass on CPU and Metal, covering missing/narrowed/duplicate
and unbound targets, graph reuse, workspace accounting, floor/grow behavior,
output preservation and speculative-read failure after submission. After the
copy change, the directly affected matrix-row test passes 32 assertions on each
backend for shared contiguous and per-slot strided inputs, including multiple
waves. No aggregate verification, full suite or unrelated test ran.

Initial abbreviated test filters selected zero cases; they were corrected. The
first full preview test exposed a fixture that warmed optional weights before
its intended read-failure probe. Warming only the source fixes that preparation.
A later attempt to write a fresh no-preview baseline failed from disk exhaustion;
it has no valid JSON/result and is excluded. Its partial logits were removed with
a hash/size failure record. Successful logits remain available at their original
paths as independently writable APFS clones of byte-identical references.

## Next optimization choices

1. **Specialize known single-expert Metal dispatch.** The executor already knows
   all local IDs are zero, but each matrix operation still invokes generic route
   mapping, invalid-row zeroing, a map barrier and the matrix kernel. A dedicated
   contract could precompute the deterministic map and skip redundant dispatches
   while retaining the exact matrix kernel. The measured request has 17,367
   submissions and roughly 2.74 s of GPU command service. Service includes barriers,
   not just arithmetic; profiling the proposed dispatch reduction is necessary.
   Do not infer that an arbitrary one-expert GGML operation has valid zero IDs.
2. **Index actual routes once.** On the existing 835-record gate capture, simply
   skipping inactive experts reduces gather ID comparisons from 16,406,528 to
   4,590,640 (72%). A bounded route index could also remove repeated known-ahead
   scans while retaining expert and row order. These are operation counts, not
   latency measurements; measure before increasing persistent workspace.
3. **Move useful reads earlier using actual routes.** Of 760 predicted bundles
   in the prior replay, only 46 hit the first four actual experts. Most predicted
   reads are already hidden by known-route overlap. Scheduling actual demand under
   independent GPU work is a more promising experiment than enlarging speculative
   depth. It requires explicit dependency, lease and budget ownership, and must
   preserve the original numerical path.

The 195 captured matrix gate projections contain 49,920 useful route rows but
475,392 padded rows (10.5% occupancy). This is a tiling limitation, not a claimed
9.5x speedup opportunity: resident Metal also tiles by expert. Replacing matrix
kernels with vector kernels previously changed numerics; reducing padding needs
its own arithmetic-equivalence investigation. Redundant lease synchronization
costs only about 3.3–3.6 ms/request here, so changing that safety boundary is not
the recommended first optimization.

## Evidence

Local runtime artifacts use `/tmp/atomic-engine-qwen-overlap-graph-bound-on.*`
and `/tmp/atomic-engine-qwen-overlap-graph-gather-final.*`. Copy-benchmark sources,
logs, operation counts and selected test logs use `/tmp/atomic-engine-gather-*`,
`/tmp/atomic-engine-route-scan-counts.json`, `/tmp/atomic-engine-matrix-density.json`
and `/tmp/atomic-engine-graph-preview-*`. Hashes and source snapshots are recorded
under `optimization_audit` in `atomic-engine-evidence.json`.
