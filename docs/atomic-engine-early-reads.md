# Exact next-projection early reads

## Decision

Keep the experiment off by default on both Qwen3.5 MoE and LFM2 MoE. The runner
accepts `--expert-early-read on|off`; the idle-only diagnostic API is
`llama_expert_early_read_set`. Accuracy and bounded ownership pass the focused
checks, but the selected window does not establish a request-level speedup.
Existing known-demand overlap within a projection and prepared single-expert maps
retain their qualified defaults. No main-graph or Metal-kernel changes are made.

## Mechanism and boundaries

Gate and up projections consume the same activation and actual router IDs. The
experiment binds only adjacent executor-owned custom operations in the actual
execution graph with identical input/ID tensor identities and matching geometry.
Each projection must occur uniquely. Missing, duplicate, narrowed or unbound
pairs fail closed. Binding runs immediately before scheduler execution, including
cached graph reuse; reserve graphs are not bound. Only stable registered-weight
identities and scalar occurrence data survive binding, never graph tensor pointers.

While the source's last-expert GPU group runs, read up to the first four distinct
ascending target experts. The existing route index supplies these IDs when
available; otherwise validation maintains a fixed four-entry ordered selection.
The target's arithmetic, row order, padding, matmul grouping and submission order
are untouched. No prediction, added graph branch, background thread or deferred
completion queue is introduced. The down projection has a different activation
input and is ineligible. Vector/decode waves skip the optimization.

Binding metadata costs **30,728 bytes** on this arm64 build, explicitly admitted
as scratch before allocation and added to the reported working floor. Admission
failure preserves the current mode. Capacity reserves the source GPU group,
at most four target leases and the existing maximum read-staging allowance.
Early work starts only after no source-expert lookahead leases remain. Hot first
experts are skipped without replacement-order mutation; later misses do not
replace them in the four-entry selection. All reads use normal store admission,
source validation and publication. Target leases end in the source callback after
GPU synchronization; only evictable cache residency crosses into the next callback
or a resize boundary. A read failure synchronizes the submitted GPU work, releases
leases and poisons the evaluation using the existing cleared-output error path.

The experiment requires Metal and LRU, and rejects combination with speculative
preview or demand-only ATET tracing. Restricting it to a separate opt-in mode
avoids combining independently bounded lease sets or changing trace semantics.

## Targeted checks

The single Metal case `exact early reads preserve outputs budget and graph binding`
passes 56 assertions. It compares both independent projection outputs, including
strided activations and IDs, repeated matrix waves and indexed/non-indexed routes.
It checks exact first-four reads, unchanged logical I/O, unchanged matmul and
submission counts, warm skipping, absent/narrowed/duplicate bindings, charged
floor/shrink/grow and an early source-file failure after GPU submission. No pins
survive successful or failed callbacks. No aggregate verification or full suite
was run.

The real-model comparisons use the existing resident full-logit, committed-token
and serialized-state references. Every saved output matches byte for byte. Both
modes keep the same cap, resident owner accounting, context 512, microbatch 32,
prepared Metal maps, known-route overlap and LRU. Qwen retains its default route
index; LFM retains its default scan. Builds and inference runs are sequential.

## Qwen observations

Ornith 35B, 190 prompt tokens, 16 committed tokens, 5 GiB:

- Forward off/on pair: **6.0901 / 5.9843 s** request;
  **4.5453 / 4.4342 s** prefill.
- Reverse on/off pair: **5.4515 / 4.9207 s** request;
  **3.9407 / 3.4454 s** prefill.
- Each on run moves **780 slices / 434,503,680 bytes** into **195** tail windows.
  Total logical reads remain **38,824,181,760 bytes**, with **17,367** submissions.
- Early acquisition service is **38.3–40.1 ms** and GPU command service in those
  windows is **14.2 ms**. Binding costs **0.63–0.64 ms**. These are nested service
  totals, not measured overlap intersections. Even completely hiding that GPU
  service offers only about **0.3%** of the best observed request time.
- Source-transfer service falls from 3.410 to 3.282 to 2.820 to 2.341 s in execution
  order. That variation is much larger than the available overlap; the first
  pair's apparent gain is insufficient evidence to enable the feature.
- The opt-in floor is **4,933,653,578 bytes**, and peak managed charge is
  **5,368,625,450 <= 5,368,709,120 bytes**.

## LFM observations

LFM2.5-24B-A2B, 200 prompt tokens, 52 committed tokens then EOG, 4 GiB:

- Forward off/on pair: **12.4343 / 9.2133 s** request;
  **7.4201 / 4.5247 s** prefill. Source transfer alone changes by 2.72 s;
  this pair is dominated by warming/measurement variation.
- Reverse on/off pair: **9.1991 / 9.3245 s** request, but
  **4.4779 / 4.2245 s** prefill. The small request improvement occurs despite
  regression in the only phase eligible for early reads. It cannot establish
  benefit from this optimization.
- Each on run moves **888 slices / 1,609,039,872 bytes** into **222** windows.
  Logical reads remain **99,582,345,216 bytes**, with **13,077** submissions.
- Early acquisition service is **110.7 ms**, GPU command service in the selected
  windows **24.5–24.8 ms**, and binding **1.26–1.33 ms**. The available GPU service
  is again less than **0.3%** of request duration, before overhead or contention.
- The opt-in floor is **3,567,230,024 bytes**, and peak managed charge is
  **4,294,965,368 <= 4,294,967,296 bytes**.

These are partially cached observations, not cold-SSD or statistical latency
qualification. Buffered file reads and OS page faults contribute to source time;
logical bytes do not measure physical SSD traffic. GPU service can change under
concurrent UMA reads. No speedup or non-regression claim is made.

## Live budget boundary

A Qwen run with the experiment enabled requests a shrink to 4706 MiB during
prefill, exercises duplicate/latest-wins requests and growth to 6144 MiB, returns
to 5120/4706 MiB and rejects a 1 MiB request below the working floor. Full logits,
committed tokens and serialized state remain identical to the resident reference.
The low-headroom guard skips 156 matrix opportunities; the other 39 windows read
156 exact target slices. Every observed current charge fits its applied cap.
Lifetime peak **6,442,184,698 bytes** fits the maximum applied **6,442,450,944 bytes**;
it is not compared to the smaller final cap. Growth is lazy, latest pending target
wins, and the rejected request preserves the previous applied budget.

## Reproduction and next decision

Build only the affected targets:

```sh
cmake --build build-metal --target test-expert-executor llama-expert-infer -j 8
build-metal/bin/test-expert-executor --backend MTL0 \
  'exact early reads preserve outputs budget and graph binding'
```

Run matched off/on and on/off pairs, setting `QWEN_GGUF` or `LFM_GGUF` to the local
checkpoint path and using distinct output files:

```sh
build-metal/bin/llama-expert-infer --model "$QWEN_GGUF" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest --profile-experts \
  --prompt-file docs/atomic-engine-prefill.txt --tokens 16 --budget-mib 5120 \
  --expert-early-read on --output /tmp/qwen-early-on.json --logits /tmp/qwen-early-on.f32
build-metal/bin/llama-expert-infer --model "$LFM_GGUF" \
  --ctx 512 --ubatch 32 --gpu-layers 99 --state-digest --profile-experts \
  --prompt-file docs/atomic-engine-lfm-prefill.txt --tokens 64 --budget-mib 4096 \
  --expert-early-read on --output /tmp/lfm-early-on.json --logits /tmp/lfm-early-on.f32
```

The live Qwen control adds:

```sh
--prefill-budget-mib 4706 --resize-at 2:6144 --resize-at 2:4706 \
--resize-at 4:6144 --resize-at 8:5120 --resize-at 10:1 \
--resize-at 12:6144 --resize-at 14:4706 --resize-at 15:5120
```

Raw reports, output hashes, source/binary provenance and budget snapshots are
recorded under `early_reads` in [the evidence record](atomic-engine-evidence.json).

The small gate/up tail window is now bounded and measured. Increasing speculative
depth does not address its short duration. The next distinct hypothesis is earlier
actual-route availability plus independently scheduled shared-FFN GPU work. That
requires explicit scheduler dependency/completion ownership and separate accuracy
qualification; it is not delivered or enabled by this change.
