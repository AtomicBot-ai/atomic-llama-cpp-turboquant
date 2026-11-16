# Bounded expert route index

## Decision and implementation

Budgeted Qwen3.5 MoE Metal contexts now attempt route-index admission after all
mandatory context and graph allocations. Admission failure leaves the original
scan path available, so the optional optimization cannot raise the minimum budget
required to load an existing context. Once admitted, the index is an explicit
scratch owner and raises the reported execution floor by 22,530 bytes. Disabling
it releases that charge. CPU, LFM and legacy expert-only budgets retain the scan
default. The diagnostic runner accepts `--expert-route-index on|off`; the idle-only
API is `llama_expert_route_index_set`.

Each projection reads its route IDs once and constructs stable linked lists of
flattened token/slot positions, plus an ascending list of active experts. Gathering
walks those lists; known-demand lookahead searches the active list instead of
rescanning all IDs. The optional preview warm guard also uses the unique experts.
Duplicates remain separate rows in their original order. Expert ordering, grouping,
32-row padding, prepared Metal maps, arithmetic, read scheduling and lease ownership
are unchanged. This is one build per projection, not memoization across gate/up/down
callbacks: rebuilding avoids stale routes when graph inputs or shapes change.

The fixed index supports at most 1024 experts, 256 token rows and 32 slots per row
(8192 route entries). Oversized shapes use the generic scan without growing memory.
Strided ID tensors are supported. All IDs are validated before weights are read or
GPU work is submitted. Invalid IDs retain the sticky-error/cleared-output behavior.
No tensor pointers or cache of graph generations are retained by the index.

## Correctness and isolated measurement

Two selected CPU cases pass 40 assertions; the executor case passes 31 assertions
on Metal. They check stable duplicates and strided IDs, maximum capacity, reuse
with an empty shape, oversized-shape fallback, admission/floor/grow/disable, and
invalid routes before reads/submission. Executor comparisons preserve output bytes,
access/release order, matmul counts and submission counts with shared/strided
activations, vector inputs, matrix waves, repeated waves and the generic fallback.

An isolated C++ replay uses 835 captured Qwen gate records and 2236 LFM gate records.
It compares checksums of ordered row visits and bounded known-expert lookahead,
then times four crossed-order trials of 20 passes. All orders match. Qwen takes
411–418 ms for scanning versus 8.35–9.66 ms indexed per 20 passes; LFM takes
61.3–64.4 ms versus 3.84–4.10 ms. These replay numbers isolate routing work; they
are not end-to-end inference times or GPU-overlap measurements.

## Qwen runtime comparison

The six fixed-cap runs use off/on, on/off, off/on order, context 512, microbatch 32,
190 prompt tokens, 16 committed tokens, a 5 GiB cap, LRU, preview off and the already
qualified single-expert Metal maps and known-route overlap in both modes.

The two later pairs measure 4.9361 to 4.7038 seconds and 4.9363 to 4.7994 seconds,
reductions of 4.7% and 2.8%. The first pair measures 5.2907 to 4.9675 seconds but
has a larger source-transfer difference. Source transfer in the later pairs is
2.229/2.222 seconds off versus 2.122/2.147 seconds on. Therefore the complete
observed percentage cannot be attributed solely to CPU routing; direction repeats
and the local work reduction is independently established. This is a measured
profile decision, not a confidence interval or a general throughput guarantee.

Successful ID reads drop from 60,484,104 to 192,264 across 2505 projections, with
no shape fallbacks. Index construction takes 1.30–1.39 ms/request. As a diagnostic
cross-check, projection time left after subtracting non-overlapping recorded
phases falls from about 48–49 to 21–22 ms; this residual includes timer/control
work and is not a direct route-only timer.

All six runs preserve resident logits, committed IDs and state
`9755c384a23111fe` bitwise. Logical weight reads remain 38,824,181,760 bytes;
submissions remain 17,367 and prepared Metal operations remain 44,568. The floor
changes from 4,933,600,320 to 4,933,622,850 bytes. Indexed high-water is
5,368,594,722 bytes under the unchanged 5,368,709,120-byte cap.

## LFM decision

Four runs use off/on then on/off at context 512, microbatch 32, 200 prompt tokens,
a completed 52-token answer and a 4 GiB cap. The initial 11.342 to 8.772 second
change is explained almost entirely by source-transfer warmup (8.202 to 5.618 s).
The reversed pair is 8.8988 seconds off versus 8.9267 seconds on, despite a faster
prefill. A request-level speedup is not established, so LFM stays off by default.

The index still reduces successful ID reads from 10,404,108 to 112,524 across 6708
projections; construction costs 1.03–1.12 ms. All four runs preserve complete
resident logits, tokens, EOG and state `84ac8e66a7115244` bitwise. Reads remain
99,582,345,216 bytes and submissions 13,077. The indexed floor is 3,567,221,826
bytes; high-water 4,294,966,370 remains under the 4,294,967,296-byte cap. The extra
scratch reservation reduces available cache capacity; no equal-capacity shortcut
was used to hide that cost.

## Final activation checks

The final default-on Qwen probe combines the index with opt-in preview and live
shrink/grow: 5120 to 4706 MiB during prefill, latest-request-wins, growth to 6144,
subsequent shrink, a rejected 1 MiB target, and further grow/shrink. All logits,
committed IDs and state match the saved reference. Every sampled state fits its
applied cap; lifetime high-water is 6,442,211,354 under the maximum applied cap of
6,442,450,944 bytes. Index plus preview have a combined 79,914-byte scratch charge
and a 4,933,680,234-byte floor. The index still reads exactly 192,264 IDs; preview
performs 38 generations/152 expert prefetches, demonstrating their joint path.

A separate default-on microbatch-128 completion preserves all resident logits,
token 23, EOG 248046 and state `d8bd6c5d1cf32b92`. It indexes 357 projections with
187,248 ID reads and a 5,368,330,034-byte high-water under the 5 GiB cap. This is
shape/accuracy qualification, not a paired latency result for microbatch 128.
Inference, executor-test and server targets build. Model work consists of six
Qwen and four LFM comparison requests plus these two final controls. No failed
or divergent model result is excluded from the report.

## Evidence and limits

Local model artifacts use `/tmp/atomic-turbine-qwen-overlap-route-index-*` and
`/tmp/atomic-turbine-p10a-route-index-*`. Selected test logs, the replay benchmark
source/input/output, build logs and checksums use `/tmp/atomic-turbine-route-index-*`.
The `route_index` section in `atomic-turbine-evidence.json` retains raw artifact
hashes, prior source/binary snapshots and qualification details.

No full suite, aggregate verify, native pressure-event test, server request or
unrelated workload matrix ran. Measurements cover the existing Qwen and LFM
checkpoints on M4 Max; source-transfer/cache/background conditions are not fully
controlled. The new `route_id_reads` diagnostic counts valid-path reads; failed
validation may stop before its pre-counted validation span completes.
