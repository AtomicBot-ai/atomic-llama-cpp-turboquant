# Qwen4Exp CUDA long prefill qualification

## Final result: selected memory and bitwise gates pass

After the streaming-only routing-weight lifetime fix, all 993,280 finite
logits, IDs `[271, 248068, 271, 248069]` and the entire 225,235,640-byte
session file match the corrected ordinary CUDA control byte for byte.
The prior single-byte state difference is gone. This closes the selected
3,481-token, context-4,096, microbatch-eight, 22-GPU-layer numerical gate.

Managed peaks are 4,286,049,462 host bytes (8.50 MiB below 4 GiB)
and 2,147,483,648 device bytes (2 GiB). The runtime envelope remains
2 GiB, and the charged serializer staging remains 32 MiB. Prefill took
715.414 seconds in this one run; cache state is uncontrolled,
so this is a diagnostic timing, not proof of acceleration or regression.
The corrected full ordinary control was reused; its single-token continuation
was rechecked with the patched library and retained identical logits/state.

See the [recorded evidence](atomic-turbine-qwen38-cuda-long-prefill.json)
for binary/source fingerprints and preserved intermediate failures. The
qualification uses the existing Ubuntu CUDA snapshot plus the isolated graph
hook; it does not claim that every unrelated local change was tested there.
No aggregate suite or broad performance matrix ran. Other prompts/checkpoints,
larger contexts/microbatches remain unqualified. The subsequent
[long HTTP control](atomic-turbine-qwen38-cuda-http-prefill.md) passes the
selected 3,481-token request with 28 observed server batches and exact full
slot bytes at unchanged 4/2 GiB caps; it does not recapture full logits.
The subsequent [server lifecycle control](atomic-turbine-qwen38-cuda-server.md)
passes short raw completion, stop, SSE cancellation and slot restoration.
The follow-up [resize control](atomic-turbine-qwen38-cuda-resize.md)
passes a short context-2048 profile; context 4096 safely rejects its unchanged
64 MiB transient allowance and retains this fixed-budget qualification.

## Intermediate control: exact logits, one remaining state ULP

With the 4 GiB ordinary control cache described below, all 993,280 finite
output logits and IDs `[271, 248068, 271, 248069]` match the saved streaming
run byte for byte. The streaming implementation and its 4/2 GiB caps were
unchanged, so that expensive run was not repeated merely to fix the control.
Ordinary prefill took 208.508 seconds; this remains a single diagnostic timing
with a larger total memory allowance, not equal-budget performance evidence.

Before the allocation-lifetime fix, full serialized-state equality fails at exactly one byte, zero-based
file offset 225,233,808 of 225,235,640. It belongs to the final indexer K tensor
(layer 47), logical position 3,480, raw-key component 108. The tensor header
declares F16 with 512 bytes per row. Streaming stores `0x0d4a`
(0.0003228187561035156), ordinary CUDA `0x0d49`
(0.00032258033752441406): one FP16 ULP. This is real cache data, not metadata
or padding. Do not treat matching current logits as proof that all future
indexer decisions are identical. The follow-up below isolates and corrects this remaining difference.

## Weighted-reduction allocation lifetime

Restoring the same ordinary checkpoint immediately before position 3,480
isolates the difference to one token. Layers 0 through 45 match. Layer 46
has identical attention, shared-expert output, routed down-projection values
and router weights, but 1,267 of 2,560 weighted-reduction values differ.
Replaying those exact operands with separately rounded multiply/add matches
streaming; replaying them with FMA matches ordinary CUDA. Maximum difference
is 7.450580596923828e-9 before the residual/HC path and next-layer indexer.
Weights were captured after the down projection, while still live; reading
them after the reduction is invalid because the allocation can be reused.

The streaming final reduction reused the allocation of its 40-byte router
weights for its 10,240-byte output. The CUDA fusion overlap check therefore
correctly rejected weighted-reduction fusion. Ordinary CUDA used distinct
ranges and retained FMA. This is an allocation-lifetime issue, not inaccurate
packed expert projection or a reason to disable fusion globally.

The graph hook creates a separate small `ggml_dup` of the final routing
weights and marks that copy as an output. It applies only when a Qwen4Exp
expert executor uses the managed discrete backend and the layer resides on
that device. CPU layers, Metal, other architectures and ordinary execution
retain their existing graph. The copy remains in the normal graph allocation
and shared ledger; neither cap nor the runtime envelope changes. At ten
selected experts and eight tokens, the payload is 320 bytes per affected
layer, plus backend alignment, and one extra copy operation. No throughput
claim is made for this correctness change.

Marking the original reshape directly was rejected: `ggml_set_output` also
marks its `view_src` DIV, which prevents Top-K fusion and changes six router
weights in the first GPU layer. A separate copy keeps that fusion intact.
The existing graph construction boundary is sufficient; no backend kernel,
allocator algorithm, public API or ordinary scheduling change is needed.
The ordinary path incurs only the short-circuited executor gate during graph
construction, with no copy, worker, feature allocation or GPU operation.

The final one-token continuation passes uninstrumented bytewise logits and
complete state comparison. Repeating that same continuation in ordinary CUDA
with the patched library also matches its pre-change results. Long prefill
is checked separately because a passing single token does not establish
correctness for all preceding microbatches.

## Initial comparison: mismatched execution placement

On Ubuntu / RTX 4070 Laptop (8 GiB), both selected runs completed.
Streaming processed all 3,481 prompt tokens and four output rows under the
unchanged caps and runtime envelope. Managed peaks were 4,286,049,462 host
bytes (8.50 MiB headroom) and 2,147,483,648 device bytes (exactly the cap).
This does not qualify long CUDA prefill as numerically equivalent.

All 248,320 logits in the first output row already differ at the end of
prefill, with maximum absolute difference 1.1570472717285156. All 993,280
captured floats are finite in each run. Output IDs are:

- Streaming: `[271, 248068, 271, 248069]`.
- Ordinary CUDA: `[271, 248068, 198, 760]`.

The first two emitted tokens agree despite differing logits. At the third
output, the leading candidates are 271/198 in streaming and 198/271 in the
ordinary control. The third emitted token is then fed back, so the final
state mismatch also follows different inputs: it is not independent proof
of a state-update defect. Both session files are 225,235,640 bytes, including
their 12-byte header; their hashes differ. The follow-up below localizes the
cause to an ordinary-control cache fallback, not a confirmed streaming defect.

Prefill wall time was 724.570 s streaming and 231.439 s ordinary CUDA.
These are single diagnostic timings with uncontrolled file-cache state;
the ordinary control can use substantially more host memory and its device
expert cache alone is 2 GiB. This is not an equal-budget speed comparison.
See the [machine-readable evidence](atomic-turbine-qwen38-cuda-long-prefill.json)
for hashes, per-row differences and the exact tested binary.

## Control placement correction

The first eight tokens reproduce the mismatch without the long prefill.
Selective layer-output captures preserve both uninstrumented outputs and
show exact embeddings and layers 0 through 26. Layer 27 is the first mismatch.
Its attention, shared expert, router logits, selected expert IDs and weights
match. The first captured differing projection is the IQ2_XXS gate.
Detailed MoE captures perturb fusion slightly, so they are not used alone
as evidence of the uninstrumented numerical result.

The actual gate inputs match, while the ordinary gate output is in a host
buffer. Replaying the original 512-expert IQ2_XXS gate weights with these
inputs and IDs on CUDA gives exact native/streamed results for all 51,200
floats, but differs from that host-resident graph result.

Upstream `llama_moe_cache::get_slot_map` returns no cache when a group's
slot count is smaller than `min(tokens * selected_experts, total_experts)`.
The 2 GiB control cache assigns only 50 slots to singleton layout groups.
Eight tokens with ten experts require 80 slots, so those groups fall back
to ordinary CPU execution despite `moe_cache_cpu_share=0`. CPU share controls
misses within an active cache; it does not prohibit this whole-group fallback.
The original five-token short control needed only 50 slots and was unaffected.

The diagnostic now configures a 4 GiB ordinary cache, giving every group at
least 101 slots on the checked checkpoint. The eight-token prefix then matches
all captured logits and serialized state byte for byte without tensor tracing.
Correcting the diagnostic cache alone changed no production behavior;
there is no reason to alter streaming arithmetic to mimic an accidental CPU
fallback in the control. The streaming caps and 2 GiB runtime reserve remain
unchanged. The ordinary control has a larger total memory allowance.

## Selected control

Use the existing `atomic-turbine-qwen38-context.txt` field-archive prompt:
3,481 tokens, SHA-256
`ff564330c036acd2575b9c4174df6f1223b09d2e55fdc1f3a7d1ced3a0f7ea2e`.
The diagnostic requests context 4,096, batches of 128, microbatches of eight,
one sequence, four CPU threads and 22 GPU layers. This is 28 prefill batches
(27 full batches and a final 25 tokens), followed by up to four greedy
outputs. The final emitted token is not fed back.

Run streaming first, then ordinary CUDA on the same complete local Strata
checkpoint. Its two shard hashes were already checked in the
[token-row qualification](atomic-turbine-qwen38-cuda-token-rows.md).
The corrected native control uses the existing upstream host-expert cache with a 4 GiB
device cache and zero CPU share for GPU-layer misses. It is not subject to
the streamed host cap. Both preserve the original packed weights and use
the same CPU/GPU layer placement. The only production change is the gated
streaming allocation-lifetime hook above; ordinary defaults are unchanged.

Before using this control for another checkpoint or microbatch, inspect the
logged cache-group slot counts. The cache must hold the worst-case selected
experts for that microbatch in every GPU-layer layout group. Four GiB is a
checked diagnostic setting here, not a universal family requirement.

## Diagnostic memory

The shared streaming ledger retains 4 GiB host / 2 GiB device limits and the
unchanged 2 GiB runtime envelope. Long mode uses the public
`llama_state_save_file` API instead of allocating a complete state copy.
A separate 32 MiB host ticket is admitted before inference and held through
serialization. The checked single-sequence F32/F16 profile bounds each
serializer tensor range from recurrent, PLE and KV dimensions. The ticket
covers the temporary file-writer staging buffer and its growth. The file
contains the standard session header with an empty token list, followed by
complete context state. The resumed-token diagnostic also uses the public
file loader, with the same staging ticket; this does not qualify server slot
restoration or arbitrary state files.
The earlier short diagnostic retains its 128 MiB full-state capture buffer.
Neither diagnostic allocation changes the engine runtime reserve.

For this checkpoint, the conservative F32 tensor-range bound is 8 MiB.
The final diagnostic assertion allows three times the bound (24 MiB) for
old and new vector storage during growth, below the 32 MiB charge. The paired
run binary used a weaker two-times assertion; the actual dimensions satisfy
both. After tightening only that guard, the target was rebuilt without
repeating those initial numerical runs. The allocation-lifetime follow-up
uses the stronger guard and repeats the affected streaming run.

All final output vocabulary rows and the complete session files are compared
bytewise. Intermediate prefill output rows are not captured. Managed peaks
are engine-ledger values, not whole-process RSS or total CUDA-driver usage.
A single paired run does not qualify performance or all context-4096 inputs.

## Reproduction

Only the optional diagnostic target is built:

```sh
cmake --build build-cuda --target llama-qwen38-cuda-compare -j6
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" stream long-stream docs/atomic-turbine-qwen38-context.txt
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" native long-native docs/atomic-turbine-qwen38-context.txt
cmp long-stream.f32 long-native.f32
cmp long-stream.state long-native.state
```

Run the two modes sequentially without another GPU workload. The output
report includes prompt IDs, emitted IDs, elapsed prefill time, state size
and streamed ledger peaks. No aggregate verification or full suite is part
of this qualification.

For bounded localization, optional positional arguments select a prompt
prefix and a tensor-name regex. `--prefill-only` captures the current output
without feeding it back; `--resume-state FILE --resume-position N` restores
a diagnostic state and starts at the supplied position. The caller must use
the matching prompt/checkpoint. Optional tensor tracing reserves 4 MiB host
scratch before context creation and is disabled in final numerical runs.
Selecting fusion-interior nodes can perturb arithmetic, so trace equality
alone is not a qualification gate.
