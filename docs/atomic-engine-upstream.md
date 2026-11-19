# Upstream integration boundaries

Keep fork implementations separate from upstream algorithms. AGENTS.md defines
the review and testing rules. This inventory describes the remaining integration
points; it is not an instruction to run a full verification gate.

Preserving existing functionality and performance takes priority over minimizing
the diff or avoiding duplication. A separate implementation is preferable when
sharing would alter or burden the ordinary path. Keep deliberate copies narrow,
record their upstream revision, and track relevant fixes when updating them.

## Review and isolation

The 2026-11-14 review covered the current upstream diff: public model parameters,
loading, graph construction and placement, context lifecycle, physical allocation,
Metal dispatch and diagnostics, common CLI plumbing, server control, and build/test
registration. Fork modules were checked at those interfaces. This was a manual
maintainability review, not a security certification or exhaustive bug audit.

- Fork C++ controls now live in `src/llama-atomic.h`. Their definitions, including
  context member functions, live in `src/llama-context-atomic.cpp`. Consumers of
  these experimental controls must include the new header; symbol names and
  signatures are unchanged. `src/llama-ext.h` matches the upstream baseline.
- Allocation-observer declarations live in `ggml/src/ggml-alloc-observer.h` with
  their existing C linkage. `ggml/include/ggml-backend.h` matches upstream.
- Prepared single-expert map validation and construction live in the backend-owned
  `ggml-metal-expert.cpp/.h`, accessed through the existing proc-address registry.
  `ggml-metal-ops.h` matches upstream. Generic dispatch keeps its original body and
  formatting, with only preparation guards around map/zero/barrier operations.
- Fork test registration is in `tests/atomic-engine.cmake`; the upstream test file
  has one include. Fork library sources are appended as one block after the
  unchanged upstream core source list. These registrations do not run tests.

Relative to the same upstream HEAD, the context source diff decreases from
297 touched lines to 89, Metal dispatch from 84 to 11, Metal backend glue from
69 to 25, and test registration from 28 to 2. Implementations were relocated,
not deleted or replaced with copies of upstream code. No precision, scheduling,
cache policy, memory floor, or default optimization setting changes in this work.

The subsequent [disabled-feature audit](atomic-engine-disabled-path.md) removes
unconditional observer scopes, controller state, source-path collection and
server controller work from ordinary execution. It records the remaining costs
and focused correctness/dispatch measurements.

## Necessary hooks and upgrade checks

1. **Model loading:** `llama_model_params` has two fork budget fields, both zero
   by default. The loader records source files, separates streamed tensor metadata
   from allocated weights, and validates the supported profile. The model keeps
   source/ledger ownership. Existing loading callbacks cannot supply on-demand
   expert slices without retaining the full tensors. This changes the public
   parameter struct layout: rebuild consumers against the fork; upstream binary
   compatibility is not promised. Review new load modes and tensor representations.
2. **Physical allocation:** `ggml-backend.cpp` intercepts buffer admission and
   releases a reservation after physical buffer destruction, including releases
   on another thread. Opaque backend buffers carry two ownership fields. With no
   observer installed, the ordinary allocation path and custom alloc_buffer_n
   dispatch remain available. Per-call estimated sizes alone cannot enforce the
   shared cap. Review new buffer allocation entry points and ownership changes.
3. **Graph integration:** the common expert matmul builder creates the CPU bridge
   only for registered streamed weights. Its surrounding MoE operations retain
   model-device placement to preserve arithmetic. Other weights use the original
   path. Review graph callback names, scheduler placement and new expert formats;
   model architecture builders are not independently forked.
4. **Context lifecycle:** short hooks claim the supported single session, account
   state/scratch allocation, bind optional work to the executed graph, check sticky
   errors and apply budgets after synchronization. The context owns executor and
   controller lifetimes. These hooks must follow future allocation/scheduler
   changes; moving them to an external wrapper would miss internal allocations.
5. **Private graph identity:** the executor reads `ggml_custom_op_params` using
   the actual in-tree definition, compares its callback and opaque userdata, and
   retains only scalar binding information. There is no public custom-op identity
   accessor. Keep this private-layout dependency explicit; do not duplicate the
   struct or infer identity from names. Recheck reused, absent and narrowed graphs
   when ggml custom-op representation changes.
6. **Metal capability contract:** prepared maps use op_params[0] and existing
   output-tail storage only for validated, exclusively owned, immutable zero-ID
   single-expert operations. Check that upstream still leaves this parameter slot
   unused and preserves map layout, extra allocation sizing and matrix selection.
   Unsupported preparation falls back to generic dispatch. The shared host-pointer
   hook rejects private buffers; timing reads completed command buffers only.
7. **CLI/server:** common code exposes explicit budget settings and rejects
   unsupported combinations. The server's inference-thread idle callback lets
   automatic pressure handling run without requests; byte metrics remain exact.
   Budget queries exclude the concurrent yielded-task path. Review queue locking,
   sleep behavior and new external state caches when updating the server.

A new control framework or duplicated model/backend implementation has maintenance
cost, but bounded duplication is acceptable when it better isolates the ordinary
path. Removing allocation interception or graph-placement constraints
would weaken the budget or exactness contract. Neither was done to reduce line
counts. Changes to these retained hooks need their own focused validation.

## Focused validation of this isolation

- Build: `test-memory-contract`, `test-expert-executor`, `llama-expert-infer` and
  `llama-server` in the existing static Metal configuration.
- One physical-buffer lifetime/admission case: six assertions pass on CPU.
- One prepared-map/generic Metal dispatch case: 14 assertions pass, including
  exact output, invalid-ID fallback and output-tail bounds.
- One Qwen ubatch-32 inference request with prefill growth to 6144 MiB, subsequent
  5120/6144/5120 MiB changes and a rejected 1 MiB request. Saved resident logits,
  committed IDs and state `9755c384a23111fe` match bitwise. Snapshot usage stays
  within the applied cap; lifetime peak stays below the maximum applied cap.
- All five fork test targets remain registered once. Three restored headers
  match upstream HEAD byte for byte. No tests run merely to enumerate targets.

No full suite, aggregate gate, server request test, shared-library build, second
platform, or actual upstream merge was performed. Builds and runtime checks were
sequential. Timing from this run is not a performance comparison. Logs and hashes
are recorded under `upstream_isolation` in `atomic-engine-evidence.json`.
