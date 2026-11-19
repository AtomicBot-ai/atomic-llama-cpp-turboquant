# Disabled-feature isolation audit

## Scope and fixes

This review checks ordinary llama.cpp execution when expert streaming, the engine
budget and automatic pressure handling are disabled. Preserving this path takes
priority over sharing implementation. It is not a claim of zero overhead on all
models, platforms or entry points.

Confirmed interference and corrections:

- Model/context entry points constructed allocation scopes even without a budget.
  Those scopes swapped thread-local allocation observers, including temporarily
  hiding a caller's observer. They are now optional stack scopes constructed only
  for an explicit contract. Reclaim callbacks are also constructed only then.
  Explicit empty scopes inside the streaming executor still intentionally suppress
  interception for buffers already admitted by their owner.
- Budget controller state, its mutex and policy were embedded in every context.
  They now live in a separately owned `llama_context_budget`, created only for a
  budgeted context. Destruction order still releases the executor before session
  ownership. This control allocation is covered by the existing runtime envelope.
- The loader copied all model/shard paths for streaming even on ordinary loads.
  Collection is now explicitly requested by the streaming load call; quantization
  and ordinary model loading retain the default false argument. Progress sizing
  uses the original loop when streaming is off, avoiding per-tensor set queries.
- The physical allocator copied its observer and entered the admission wrapper
  even with no active observer. The ordinary branch now calls the backend's
  allocator directly. A relaxed atomic count of installed observers avoids TLS
  access on this branch; enabled admission stays in a separate helper. Observer
  scopes must be balanced, as in `llama_allocation_scope`. The count is only a
  fast rejection: observer data belongs to each thread and is never shared.
  An observer active on another thread cannot intercept ordinary allocations.
- The server installed an idle callback even without automatic pressure handling,
  causing unnecessary unlocking/callback work on idle timeouts. The callback is
  now installed only in automatic mode and cleared otherwise. Existing queue
  wait timing and its upstream worker thread are unchanged.
- Ordinary server metrics created controller snapshots and called controller APIs
  before learning that the feature was disabled. These operations now sit inside
  the budget/non-yielding guard. The optional JSON payload stays null until filled,
  avoiding allocation of an empty controller object for every metrics response.

Profile restrictions, mandatory synchronization, graph binding, expert reads and
prepared Metal operations remain conditional on the corresponding executor or
budget. No new restriction is applied to ordinary contexts. LoRA, training,
multiple sequences and other unsupported streaming configurations still reach
ordinary upstream paths when the streaming/budget gates are off; those features
were inspected at their guards, not exhaustively runtime-qualified here.

## Focused checks

- Four allocator cases pass 27 assertions. Coverage includes custom backend bulk
  allocation, zero-size behavior, exception propagation, admission on/off nesting,
  cross-thread independence, reclaim and physical-release lifetime.
- The enabled idle callback passes its two assertions. The final disabled callback
  and metrics case passes three assertions, including null controller JSON and
  absence of controller metric series. The existing engine-argument/metrics case
  checks that enabled metrics still preserve integer byte values.
- One ordinary Qwen request uses no streaming, budget or expert profiling. Full
  logits, 16 committed token IDs and state `9755c384a23111fe` match the saved
  resident reference bitwise; the output has no expert/controller snapshots.
- One enabled control uses a 6144 MiB budget, replayed prefill pressure with a
  5120 MiB minimum, then manual 6144/1/5120 MiB targets. Pressure shrinks the cap,
  the 1 MiB request is rejected, and final output matches the same reference.
  All sampled usage is within the applied cap and lifetime peak is below the
  maximum applied cap. This checks that isolating the feature did not break it.
- Inference, server and the selected test targets build in the existing static
  Metal configuration. No full suite, aggregate verification or E2E matrix ran.

A subsequent [clean-upstream comparison](atomic-engine-native-upstream.md)
checks native request latency, backend memory and exact output against a separately
built pristine upstream library on one short Qwen/Metal profile.

## Measured overhead and remaining limits

A small crossed-order benchmark compares the allocator entry point with a local
copy of the upstream HEAD allocation body, compiled with the same optimization
flags. Its fake backend returns an existing buffer, isolating dispatch rather
than measuring malloc, device allocation or inference. Six rounds are recorded.

Before isolation, median fork dispatch is about 2.315 ns/call against 1.076 ns
for the upstream body. After isolation it is about 1.259 ns against 0.983 ns.
The remaining measured increment is approximately 0.277 ns/call. Results include
warmup/noise and do not establish whole-request equivalence or a universal bound.
The local upstream-body control is not a separately built pristine upstream
library. No request-speedup claim is made from these functional model runs.

Remaining structural cost is explicit: capability/null checks at integration
points, a global observer-count read on physical allocation, two owner fields and
an ownership check on backend buffers, and passive extension fields/pointers in
model/context/queue structures. When another thread installs an observer, the
ordinary allocator must inspect its own TLS before taking the generic path.
The existing fork model-parameter ABI extension also remains; consumers must
rebuild against the fork. These costs were not hidden by duplicating the entire
loader, backend or inference engine.

The ordinary native run is one Qwen/Metal profile, not a performance matrix or a
proof for every backend. Server callback/metrics checks are component tests;
HTTP request behavior and actual OS pressure delivery were not retested. Upgrade
checks in `atomic-engine-upstream.md` continue to apply.

Raw local artifacts use `/tmp/atomic-engine-disabled-*` and
`/tmp/atomic-engine-qwen-overlap-disabled-*`. Source snapshots, diagnostics hashes
and exact-output checks are recorded under `disabled_feature_isolation` in
`atomic-engine-evidence.json`.
