# Qwen4Exp CUDA synchronized resize qualification

## Result and support boundary

The selected context-2048 profile passes manual shrink/grow during prefill
and decode, one cooperative cancellation of an unfinished worker read,
actual device-cache reclamation, and bitwise comparison with ordinary CUDA.
The original GGUF checkpoint, 22 GPU layers, microbatch eight, 4 GiB RAM /
2 GiB VRAM maximums and 2 GiB runtime envelope are unchanged. The existing
resize API's additional 64 MiB transient allowance is also unchanged.
Only the opt-in comparison diagnostic changed; no production engine,
ordinary graph, kernel, allocator or server path changed in this follow-up.

The final sequence shrinks to 4,064 MiB RAM / 2,019 MiB VRAM and grows back
to 4,096 / 2,048 MiB. The decode shrink completes a 1,966,080-byte (1.875 MiB)
device release. Both growth boundaries allocate no additional payload;
subsequent demand can use the larger cap. RAM cap reduction does not imply
that resident model/state RAM was released.

All 993,280 finite logits, output IDs `[16, 25, 561, 6511]` and the complete
118,870,664-byte session file match ordinary CUDA byte for byte. Managed
peaks are 4,248,156,774 host bytes and 2,147,483,648 device bytes. These are
engine-ledger counters, not whole-process RSS or CUDA-driver totals.
See [machine-readable evidence](atomic-turbine-qwen38-cuda-resize.json).

Context 4096 remains qualified for the previously checked
[fixed-budget long prefill](atomic-turbine-qwen38-cuda-long-prefill.md),
not for this resize allowance. Its non-cache device payload is already
1,985.3125 MiB at the checked decode boundary. Adding 64 MiB exceeds the
2,048 MiB cap even before the required expert working allocation. Therefore
even same-cap admission is rejected. Requests to 4,092 / 2,040 MiB and
impossible 1 MiB domain caps are rejected without changing any captured
ledger field or residency. Continuation retains exact ordinary CUDA logits
and state. Do not interpret this conservative refusal as a fixed-budget
inference failure or reduce either reserve to make the diagnostic pass.
A broader resize profile needs a lower resident footprint or separately
justified changes to its transient allowance.

## Selected sequence and evidence

The input is the first 24 tokens of the existing field-archive fixture.
Context 2048 is allocated up front; this is a short request in that context,
not a 2048-token prefill or long-context soak. One logical prefill call spans
three eight-token microbatches, followed by four output rows; the fourth
emitted token is not fed back.

1. During the first microbatch a real file-read worker pauses halfway through
   an expert. The staging remains charged. The probe cancels this first read;
   the worker is joined, unpublished owners are destroyed, and the store's
   existing one-shot retry reads the same original GGUF slice normally.
2. At the synchronized microbatch boundary the probe checks that the staged
   read has drained and applies shrink. It rearms one worker observation.
3. The second microbatch executes under the smaller caps. Its worker read
   completes; the next synchronized boundary grows both caps. The third
   microbatch executes after growth.
4. Before decode, impossible host/device targets are independently rejected
   with the entire before/after ledger snapshot unchanged. A same-cap
   admission succeeds in this profile.
5. After one decoded token the probe shrinks again and asserts a positive
   completed device-cache release. A token is evaluated under the smaller
   caps, then growth is applied before the last evaluated token.
6. All output rows and final serialized state are compared bytewise with the
   same prompt/context/placement in ordinary CUDA using its corrected 4 GiB
   expert cache. That reference has a larger total memory allowance.

Both observed midpoint snapshots have 1,078,950 charged host I/O bytes;
this returns to the persistent 656,550-byte I/O baseline at the boundary.
The cancellation happens while the worker is cooperatively paused, before
publication. This is not Linux `io_uring` cancellation of an in-flight kernel
request, arbitrary concurrent resize, or automatic OS-pressure handling.
The resize API is invoked only after synchronization, never under the store's
read lock. Existing lease, upload and retry ownership remains intact.

The earlier worker-completion control also matched ordinary CUDA. It used
4,064 / 2,024 MiB and changed the applied caps without necessarily evicting
occupied cache. A stronger 2,022 MiB probe failed its positive-release
assertion. The final 2,019 MiB case both admits the unchanged safety margin
and demonstrably reclaims cache. These distinctions are preserved in the
evidence rather than treating every accepted cap change as physical release.

## Reproduction

Only the optional diagnostic target is built. Run controls sequentially,
without compilation or another GPU workload during inference:

```sh
cmake --build build-cuda --target llama-qwen38-cuda-compare -j6
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" native resize-native \
  docs/atomic-turbine-qwen38-context.txt 24 "" --context 2048
build-cuda/bin/llama-qwen38-cuda-compare "$MODEL" stream resize-live \
  docs/atomic-turbine-qwen38-context.txt 24 "" --context 2048 \
  --resize-probe --prefill-resize-probe --resize-read-worker --resize-cancel-read
cmp resize-live.f32 resize-native.f32
cmp resize-live.state resize-native.state
```

Omit `--resize-cancel-read` to observe completion instead. The context-4096
rejection case uses prefix eight and `--resize-probe`, without the prefill
or worker options. The probe reports rejected calls explicitly; a completed
numerical run alone is not evidence that shrink was accepted.

The same qualified Ubuntu production library was reused throughout; the
final evidence records its hash and the diagnostic/source fingerprints.
No aggregate suite or throughput qualification ran. Other checkpoint/layout
profiles, a long request under reduced caps, native OS pressure, kernel I/O
cancellation on Qwen remain open. The subsequent
[fixed-budget single-slot server control](atomic-turbine-qwen38-cuda-server.md)
passes short raw completion, stop, SSE cancellation and slot restoration.
The manual resize result does not establish a server resize interface;
the selected [non-thinking chat/EOS control](atomic-turbine-qwen38-cuda-chat.md)
has since passed. A selected [long HTTP prefill](atomic-turbine-qwen38-cuda-http-prefill.md)
also passes at fixed 4/2 GiB caps; it does not extend resize qualification.
