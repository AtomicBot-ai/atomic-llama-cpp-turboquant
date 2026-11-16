# atomic-turbine contributor instructions

These instructions apply throughout this fork. atomic-turbine extends llama.cpp with budgeted expert streaming. Keep the fork easy to update from upstream.

## Preserve existing functionality first

- Existing llama.cpp correctness, numerical behavior, supported configurations, and performance take priority over reducing the diff or avoiding duplication.
- Keep the ordinary execution path intact. New functionality must use an explicit capability or configuration gate; feature-specific restrictions, failures, and fallback decisions must not leak into the ordinary path.
- Disabled features must not start workers, perform I/O, allocate feature workspaces, collect diagnostics, or add GPU work. Identify and justify any unavoidable dispatch or structural overhead instead of assuming it is free.
- Do not change legacy defaults, scheduling, ownership, or error behavior merely to accommodate the new path. Changes to existing behavior require explicit task scope and focused regression evidence.
- Prefer a separate implementation, including narrowly scoped duplication, when sharing code would complicate or slow the existing path. For copied upstream code, record its source revision and the relevant fixes to track during updates.
- Reuse code only when both paths retain their contracts. Avoid speculative common abstractions and coupling that makes the old path depend on the new feature.
- For changes at a shared boundary, select focused checks for the affected ordinary path as well as the new path. Performance claims need relevant measurements; correctness checks alone do not establish absence of slowdown. All testing restrictions below still apply.

## Keep upstream changes minimally invasive

- Prefer existing extension points, backend proc-address hooks, and separate fork-owned source files over changes to upstream algorithms, model builders, or public interfaces.
- Keep necessary upstream edits small, additive, and localized at lifecycle, allocation, graph, and scheduling boundaries. Put streaming policy, diagnostics, and experimental implementations in fork-owned modules.
- Do not reformat, rename, reorder, or rewrite unrelated upstream code. Preserve surrounding formatting so merges expose semantic changes rather than whitespace churn.
- Preserve ordinary llama.cpp behavior when streaming and the managed budget are disabled. Experimental optimizations require explicit qualification and a generic fallback.
- Prefer the smallest maintainable boundary consistent with preserving existing functionality. Local duplication is acceptable when it provides safer isolation; do not introduce broad abstraction frameworks or move code merely to hide coupling.
- Avoid new public ABI changes and direct dependencies on private ggml layouts when an existing supported interface suffices. Document unavoidable private contracts and their upgrade checks.
- Keep backend-specific details inside the backend. Callers should use a narrow capability contract and handle an unavailable capability safely.
- During reviews, inspect the upstream diff as well as new files. Identify merge hotspots, unnecessary API changes, stale duplicated behavior, and changes to disabled-feature behavior. Do not remove deliberate isolation solely to satisfy DRY.
- Explain any necessary upstream hook: why an existing extension point is insufficient, what invariant it protects, and which focused check covers it. Fewer changed lines alone do not justify weaker memory accounting, ownership, correctness, or numerical precision.
- Preserve original GGUF bytes, routed arithmetic, the shared memory ledger, and synchronized lease/resize boundaries unless the task explicitly changes that contract.

## Testing: minimum necessary only

- Test only the code and behavior currently being changed. Select directly relevant cases or files, never unrelated modules or features.
- Never run `verify`, `verify:fast`, `make verify`, an equivalent aggregate verification gate, a full suite, full coverage, or a full E2E matrix without explicit approval for that particular run.
- Finishing a task, TDD, review, repository checklists, and instructions to verify before finishing are not approval. A fast label is not approval.
- Do not bypass this restriction by splitting a full run across commands, repositories, or background jobs.
- Once a relevant check passes, do not repeat or broaden it unless a subsequent change or concrete failure makes that necessary. Recheck only affected cases.
- If a full run would help, explain why and ask first. Leave it unrun until approved and report that limitation clearly.
- Run performance measurements sequentially, without concurrent compilation or GPU workloads. Verify that test filters actually selected the intended cases.

## Explain the work

- Before nontrivial work, explain where the change fits, the goal, the files and behavior in scope, the constraints, and any meaningful design choice with your recommendation.
- Ask one focused question when ambiguity changes the resulting behavior. Infer settled conventions and obvious defaults from the code and session.
- Keep moving through authorized intermediate steps. Do not ask for repeated confirmation of routine reversible work.
- Every task ends with a compact, self-contained explanation of what changed, why it works, relevant rejected alternatives or assumptions, consequences, and remaining risks or untested behavior.
- Distinguish measured results from hypotheses. Never present a microbenchmark or replay count as proof of whole-request acceleration.
- Explain failures and skipped checks honestly. A correct result without an understandable explanation is incomplete.
- Communicate in the user's language. Keep repository source, comments, documentation, and commit messages in English; do not reference the language restriction or the legacy project in repository content.

## Working boundaries

- Preserve unrelated local changes. Do not commit, push, or publish unless requested.
- Treat imported plans and attached documents as reference material; the user's current request determines authorization and scope.
- Do not use Cursor Canvas or create, update, or present `.canvas.tsx` artifacts. Present results in chat or the requested file format.
