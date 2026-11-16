# LFM synchronous GPU submission profile

## Scope and decision

The previous fixed-4-GiB LFM optimization removed allocation churn but retained
51,165 submissions and approximately 8.8 seconds in synchronous compute. This
package separates submission, completion waiting and Metal command-buffer GPU
time before selecting a change. GPU command-buffer intervals include barriers
and are not pure kernel timing; CPU submission and GPU execution can overlap.

1. Add opt-in completion-time diagnostics on the private executor backend and
   profile the same 200-token prompt / completed 52-token answer.
2. If per-submit overhead dominates, combine a bounded number of independent
   expert waves without changing vector/matrix kernels or row ordering.
3. Admit all extra workspace, retain every referenced expert lease until GPU
   completion and preserve a single-expert path near the memory floor.
4. Keep trace/replay ownership accurate if leases span multiple acquisitions.
5. Compare full logits/state and live shrink/grow, with targeted wave/lifetime
   checks only. No aggregate suite or unrelated workload matrix.

The original router, packed weights, active KV/recurrent state and preferred
budget remain unchanged. The previously rejected single-encoder experiment is
not repeated. No asynchronous storage or speculative routing is introduced.

## Measurement and accepted change

The opt-in profiler now separates `submit` (the backend's graph-compute call)
and `wait` (its synchronous completion). A private Metal proc-address diagnostic
reads completed command buffers' `GPUStartTime`/`GPUEndTime`; unavailable samples
are not counted. The local Metal SDK documents these properties as host times
for GPU execution and completion. No capture or counter-buffer allocation is
needed. Profiling disabled means no timestamp queries. The normal synchronous
contract is unchanged: submit is always followed by synchronization.

The matched LRU baseline at 4 GiB measured 17.5370 seconds total and 10.0303
seconds prefill. Of 9.1301 seconds inside compute, submission occupied 0.5301
seconds and waiting 8.5955. All 51,165 submissions were timed, encompassing
102,330 command buffers with 2.4040 seconds of summed GPU intervals. These
intervals overlap host work and include more than matmul instructions; they are
not an additional term to add to CPU wall time, nor a precise kernel-only split.

The executor now submits up to four independent waves in one graph. Each wave
still uses the original one-expert `MUL_MAT_ID` path: matrix waves retain 32-row
padding and vector waves retain separate one-row nodes. Inputs, outputs and
Metal row maps have disjoint admitted workspace. Gather/scatter ordering is
unchanged. A current expert lease survives multiple group flushes when it owns
many rows; every graph pointer is discarded before completed leases are released.

Workspace capacity is chosen at construction from four, two or one groups
according to admission. At each projection, the actual group size is further
bounded by the applied cap after fixed owners, allowing the maximum physical
expert allocation for every held lease plus one complete read-staging buffer.
This is conservative for mixed quantization. No live resize occurs inside a
group. If a subsequent acquisition fails before submission, all pending leases
are released and the context rejects the failed evaluation.

On the measured LFM profile, larger scratch, host workspace and tensor/graph
metadata add 3,231,904 charged bytes. The LRU engine floor is now 3,567,199,296
bytes, or 3,567,322,880 with ClassLRU history. The user's cap remains 4 GiB;
available expert cache capacity pays for the extra workspace. The new minimum
is explicit rather than a promise to preserve the previous smaller floor.

The optimized LRU request took **12.0573 seconds**, 31.2% below the matched
instrumented baseline; prefill took **7.0310 seconds**, 29.9% below baseline.
Submissions fell from 51,165 to 13,077; prefill submissions from 27,453 to 7,149.
Compute fell to 3.8769 seconds (0.1817 submit, 3.6938 wait), with 2.0282 seconds
of summed GPU intervals across 26,154 command buffers. The arithmetic node
count stayed at 52,905 and logical read bytes at 99,582,345,216. Reducing
per-submit overhead is the supported explanation; no per-kernel speedup is
inferred from a single run.

All 52 committed tokens, EOG 124900, full logits and serialized state
`84ac8e66a7115244` match the saved resident control bitwise. Fixed-cap ledger
high-water is 4,294,966,576 bytes below 4,294,967,296. The remaining large cost
is source acquisition, including approximately 4.82 seconds in buffered reads.

## Ownership replay and live release

ATET3 adds one `R` record per released acquisition. Accesses pin entries until
that release; both replay policies exclude pinned victims and reject traces
ending with live leases. ATET1/ATET2 keep their historical semantics. The LFM
ATET3 trace exactly reproduces runtime hits and physical allocation sizes.
ClassLRU predicts 6.55% fewer logical read bytes on this trace; this package
does not add a matched fixed-cap ClassLRU latency claim or change the default.

A ClassLRU live request starts at 4096 MiB, shrinks during prefill to 3456 MiB,
grows at committed token 2 to 4608 MiB, shrinks at token 4 to 3403 MiB, rejects
1 MiB at token 6, then grows at token 8 to 4096 MiB. At the near-floor cap,
two decode steps perform 912 matmuls in 912 submissions, proving the single-wave
fallback; the roomier path groups those independent experts. Shrink releases
1,261,568,000 Metal bytes in 10.7502 ms and immediately reduces footprint by
1,272,447,720 bytes. Grow allocates no immediate Metal buffers; rejected admission
changes neither applied cap nor used bytes. The same complete answer, full
logits and state are preserved. Lifetime ledger high-water is 4,831,813,008
bytes below the maximum applied cap of 4,831,838,208.

## Targeted checks and limits

- Metal wave cases cover shared/per-slot strided inputs, decode, vector tails,
  matrix waves, multiple groups for one expert and the tight-cap fallback.
  The new single-token fixture expectation was corrected from four distinct
  experts to two; only that grouped case was repeated. Four selected Metal
  cases pass 126 unique assertions, including completion profiling and a source
  failure after the first lease is acquired but before group submission.
- One CPU routed-projection case passes 16 assertions for the changed common
  gather/scatter code. The lease-trace store case passes nine assertions.
  Six cases in the directly affected replay file cover old formats, physical
  allocation accounting, pinned eviction and incomplete lease rejection.
- Four LFM runs cover instrumented baseline, grouped profile, trace and live
  resize. One saved-control Qwen/IQ4_XS check exercises the shared executor at
  microbatch 128: full logits, IDs, EOG and state `d8bd6c5d1cf32b92` match.
  That Qwen run is a numerical check, not a new performance comparison.

Builds and GPU runs were sequential. No full suite, aggregate verification,
unrelated workload matrix or server request qualification ran. Timings are
single-request observations with uncontrolled OS cache/device scheduling;
they do not qualify cold SSD, latency tails or arbitrary architectures. Native
macOS pressure-event qualification remains separate. The final rebuilt inference
binary matches the one used for grouped/trace/live and Qwen controls.

## Reproduction

Use the original LFM GGUF as `$MODEL` and the existing qualified fixture:

```sh
cmake --build build-metal --target llama-expert-infer -j4
build-metal/bin/llama-expert-infer --model "$MODEL" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest \
  --prompt-file docs/atomic-turbine-lfm-prefill.txt --tokens 64 \
  --budget-mib 4096 --profile-experts \
  --output /tmp/lfm-gpu.json --logits /tmp/lfm-gpu.f32
```

For trace replay, replace `--profile-experts` with `--expert-trace /tmp/lfm-gpu.atet`
and use distinct output paths, then run `tools/expert-source/replay-cache.py`.
The live case adds `--cache-policy class-lru --prefill-budget-mib 3456
--resize-at 2:4608 --resize-at 4:3403 --resize-at 6:1 --resize-at 8:4096`.
The 3403 MiB target belongs to this exact context/profile, not every checkpoint.
Reports, before/after source provenance and hashes are under `p06_lfm_gpu` in
`atomic-turbine-evidence.json`.
