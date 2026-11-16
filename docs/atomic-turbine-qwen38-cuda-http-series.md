# Qwen4Exp CUDA long HTTP same-slot series

The fixed-budget CUDA server ran three consecutive `/completion` requests in
one process and slot: the 3481-token field-archive prompt, an exact repeat, and
the same token list with a single replacement at zero-based index 1740
(ID 561 to 13). The follow-ups use `cache_prompt=true`. Ordinary CUDA runs the
same history in a separate process using the same binary and libraries.

The exact repeat and the edit both reported zero cached input tokens and
re-evaluated the entire 3481-token prompt in both modes. That is the safe
fallback observed with this hybrid model and server checkpoints disabled.
The separate fresh-process [exact-prefix continuation](atomic-turbine-qwen38-cuda-http-restore.md)
does reuse a saved prefix when it can append a new token.

The initial and repeated output IDs are `[271, 248068, 271, 248069]`. The edited
prompt changes them to `[271, 248068, 198, 760]`. The streamed output IDs and
complete serialized slot files match ordinary CUDA in all three cases.
Each paired comparison uses the same history because advancing and reusing
a hybrid slot can change bytes unrelated to the visible output.

The selected RTX 4070 Laptop profile uses 22 GPU layers, context 4096,
batch 128, microbatch 8, four threads, one slot, and fixed 4 GiB RAM /
2 GiB VRAM. The existing 2 GiB runtime reserve and disabled RAM prompt
cache/checkpoints remain in effect.
Managed peaks were 4,252,495,030 RAM bytes and 2,147,483,648 VRAM bytes.
After each slot save, idle usage returned to 4,174,867,606 RAM bytes
and 2,147,352,576 VRAM bytes.

Full output IDs and complete serialized slots are compared directly for
each matching request history. Raw HTTP logits are outside this capture.
Client TTFT and server decode timings are recorded for diagnosis. File-cache
state is uncontrolled and ordinary CUDA uses a different memory allowance.

Run the existing long prefill diagnostic twice with `--series`, first with
`--mode native` and then with `--mode stream`, using the same fresh output
directory. Use the exact checkpoint and flags described in the
[long HTTP prefill reproduction](atomic-turbine-qwen38-cuda-http-prefill.md#reproduction-and-remaining-boundaries).
The diagnostic starts a loopback server per mode and keeps it alive across
all three requests. The ordinary server exposes no managed-memory gauges.
No production path, model weights or memory limit changed.

[Structured evidence](atomic-turbine-qwen38-cuda-http-series.json) records
the output IDs, processed boundaries, slot hashes, timings and budget
snapshots. Long chat, automatic checkpoint rollback, multi-slot execution
and prolonged soak remain separate scopes.
