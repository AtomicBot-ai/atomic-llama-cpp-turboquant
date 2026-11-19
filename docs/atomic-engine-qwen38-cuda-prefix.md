# Qwen4Exp CUDA reuse of a saved long prefix

A focused probe reuses trusted, contiguous text-prefix files under the existing
4 GiB RAM / 2 GiB VRAM caps and 2 GiB runtime reserve. It uses the same
RTX 4070 Laptop checkpoint, context 4096, batch 128, microbatch 8, 22 GPU
layers and one slot as the [continuous HTTP control](atomic-engine-qwen38-cuda-http-series.md).
The probe uses an isolated library. Production and ordinary CUDA are unchanged.

- Exact repeat of the 3481-token prompt restored 3456 tokens and evaluated 25.
  Client TTFT was 5.10 s after restore, versus 718.52 s for the previous
  streaming repeat with full re-evaluation. The slot restore call took about
  35 ms in the probe. The measurement excludes first-time checkpoint
  creation. Output IDs and the complete 225,249,576-byte slot match ordinary
  CUDA. A separate sampling capture also matches all four FP32 logit rows
  (3,973,120 bytes) against the existing ordinary CUDA engine reference.
- A token edit at zero-based position 1740 used the earlier 1664-token file
  and evaluated 1817 input tokens. TTFT was 370.67 s versus 716.45 s for the
  earlier streaming edit. Output IDs and the complete slot match the ordinary
  CUDA edit. This case did not capture full logits.
- The two prefix files occupy 169,244,536 and 224,387,960 bytes on disk.
  Peak managed usage is 4,252,495,030 RAM bytes and 2,147,483,648 VRAM bytes.
  After each accepted continuation and slot save, idle usage returns to
  4,174,867,606 RAM bytes and 2,147,352,576 VRAM bytes.

The active recurrent state cannot rewind to an earlier prefix by trimming
token IDs alone. The fixed-budget server disables ordinary RAM checkpoints.
The disk files keep attention, recurrent and QSA state at known microbatch
boundaries, and the existing file serializer admits its staging under the RAM
ledger. This allows an append to evaluate only the missing suffix.

## Why generic restore did not pass

The stock restore path reused the tokens and produced the expected output IDs,
but its final slot differed. For the exact repeat, all saved prefix attention
keys and QSA raw/pooled keys match the uninterrupted reference up to the cut.
After resuming with generic restore, some previously saved QSA pooled keys
differ. The first different attention-key row is at position 3472. Moving
the cut to 3472 and using either live append or another restore did not fix it.

Generic restore marks the QSA index stale from position zero. The experiment
preserves the already computed pooled rows in the trusted complete prefix,
clears the previous stale marker, then sets the marker to the first appended
position. Marking the tail still lets the host rebuild its cell layout.
An earlier version only tried to move the marker forward. The setter takes
the minimum, so the pending stale marker at zero remained and the output
slot retained the same mismatch. The explicit reset is necessary.

## Reproduction and limits

Start the existing CUDA fixed-budget server with slot actions, the RAM prompt
cache and context checkpoints disabled. Tokenize the 3481-token fixture once.
Send its first 1664 token IDs with one predicted token, then extend the live
slot to 3456 input IDs with `cache_prompt=true` and one predicted token. Save
each slot at the end of its request. The sampled token is returned but not
committed to these files. Check the returned saved token count.

Use a separate diagnostic build of the same qualified snapshot with the
[isolated restore patch](atomic-engine-qwen38-cuda-prefix-probe.patch).
Restore the 3456-token file for the original input, and restore the 1664-token
file for the edited input. Send the full token list with `cache_prompt=true`,
greedy sampling, EOS ignored and four predicted tokens, then compare complete
slot files to the respective continuous CUDA controls. The diagnostic patch
is for trusted, contiguous, single-sequence prefix files in this experiment.
Generalizing it would require a separate, verified opt-in path that can
identify compatible checkpoint files and fall back to the existing restore
behavior for arbitrary slots. Automatic capture, checkpoint selection and
invalidation remain outside this probe.

The [structured evidence](atomic-engine-qwen38-cuda-prefix.json) records the
accepted runs, rejected controls, exact output checks and ledger snapshots.
Only these selected tests ran.

## Opt-in server qualification

P11 qualification is complete for the selected fixed-cap profile: Ubuntu,
RTX 4070 Laptop, Qwen3.8 Flash Next GSQ-RCO IQ3_XXS, one text slot, context
4096, batch 128, microbatch 8, 22 GPU layers and 4 GiB RAM / 2 GiB VRAM.
Append `--engine-prefix-checkpoints "$PREFIX_DIR"` to the
[qualified server command](atomic-engine-qwen38-cuda-server.md#reproduction-and-remaining-boundaries)
after creating a writable `PREFIX_DIR`. The two live states plus a staged
replacement determine the disk headroom needed. The RAM and VRAM figures
below describe the managed engine ledger.

`--engine-prefix-checkpoints DIR` enables process-local checkpoint files in a
private subdirectory of an existing directory. The server admits this option
only for the Linux Qwen4Exp profile above. The existing RAM prompt cache and
context checkpoints must remain disabled. Requests with `cache_prompt=true`
can save two full, batch-aligned prefix states and reuse the longest matching
token prefix that leaves work for inference. A valid live append keeps its
existing path. Files are replaced only after a successful write and are
removed at server teardown or explicit slot erasure. Files from public slot
restore do not enter the private checkpoint index.

The integrated Linux CUDA check ran ordinary and streaming sequences on an
RTX 4070 Laptop. Both used the same prompt history: initial 3481-token prompt,
exact repeat, and replacement of token 1740. On streaming, the first request
automatically saved checkpoints at 1664 and 3456 tokens. The exact repeat
restored 3456 tokens and evaluated 25, with 4.77 s TTFT. The edit restored
1664 and evaluated 1817, with 359.55 s TTFT. These measurements include
automatic restore and exclude checkpoint creation on the initial request.
File-cache state was uncontrolled.

Every one of the 12 full-vocabulary FP32 logit rows (11,919,360 bytes)
and all three complete slot files match the ordinary CUDA reference byte
for byte. The ordinary slots also match the previously qualified build.
Peak managed memory was 4,252,495,030 RAM bytes and 2,147,483,648 VRAM bytes.
After every request and slot save, managed usage was 4,174,867,606 RAM bytes
and 2,147,352,576 VRAM bytes. The private directory was empty after exit.

The first integrated attempt checked the hybrid cache's minimum position
as though it were the start of the attention cache. The recurrent part
makes the hybrid minimum report the oldest recoverable state, which is
later than the first token. The corrected gate checks the attention and
indexer minima and the hybrid end. Only streaming was rerun. The ordinary
reference was retained and compared directly with the corrected build.

For reproduction, use an isolated server build with raw sampling capture
and the [HTTP prefill driver](../tools/expert-source/qwen38-cuda-http-prefill.py)
with `--series --capture-logits --disk-prefix-dir DIR`. Run native first
and stream second into the same evidence directory. Native ignores the
disk option. The capture build writes the raw row immediately before
sampling to `PREFIX_PROBE_LOGITS`. The driver requires the exact
3456/1664 cache counts and compares complete logits and slot files.
The isolated comparison is documented in the
[server evidence](atomic-engine-qwen38-cuda-prefix-server.json).

## Same-slot fallback after a checkpoint file fault

The isolated integrated CUDA server was reused without a source change or
rebuild. Each mode ran three completions with the same first 513 token IDs of
the context fixture and four greedy outputs. In streaming mode, all requests
used `cache_prompt=true` and the private checkpoint directory. The first
request created 256- and 512-token checkpoints. Before the second request,
the diagnostic removed the longer file and left the shorter one intact.
Before the third, it truncated the recreated longer file by 64 bytes,
leaving the header and token list intact. The only modified files belonged
to this loopback test server.

The server logged a failed open for the missing file and a failed state
restore for the truncated file. Both requests fell back to full prefill in
slot zero with `cache_n=0` and 513 evaluated input tokens. The 512-token
checkpoint was recreated after each failure. All 12 full-vocabulary FP32
logit rows and all three saved slots matched the ordinary CUDA files
byte for byte. Peak managed usage was 4,249,544,806 RAM bytes and
2,147,483,648 VRAM bytes, within the 4 GiB / 2 GiB caps. The private
directory was removed after server exit.

The driver initially expected a platform-specific EOF message and exited
at a final log assertion after both inference and byte comparisons.
The actual restore error and cleanup were verified from the saved logs,
and an independent `cmp` rechecked every logit and slot file without
rerunning inference. This qualifies removal and truncation on the original
implementation. See [the fault evidence](atomic-engine-qwen38-cuda-prefix-faults.json).

## Checkpoint integrity

The opt-in server records the exact file size and CRC32C of the complete
checkpoint before publishing it. It verifies both against the process-local
record before loading any tensors. The bounded reader uses 16 KiB and can
use SSE4.2 on Linux x86-64. This leaves the public slot format and ordinary
restore path unchanged. The checksum detects accidental corruption of the
server's private files; it is not authentication for external files.

The CUDA fault diagnostic changed one byte from `0x04` to `0x05` at offset
133,795,112 of the 512-token checkpoint, leaving the size at 133,795,176
bytes. The server rejected the file on the integrity check and reprocessed
all 513 input tokens in slot zero, with `cache_n=0`. The four full FP32
logit rows and complete slot file matched the existing ordinary CUDA
control byte for byte. The failed tail was recreated during prefill.

The clean 3481-token repeat restored 3456 tokens and evaluated 25. The
four full logit rows and complete slot matched ordinary CUDA, and TTFT
including checksum and restore was 4.794 s. The previous observation without
the checksum was 4.766 s. These are single observations with uncontrolled
file-cache state, so the difference is not an isolated checksum cost.
Across the two cases peak managed usage was 4,252,495,030 RAM bytes and
2,147,483,648 VRAM bytes, within the existing caps. Both comparisons used
an isolated server capture library with the unchanged engine build.

See [the integrity evidence](atomic-engine-qwen38-cuda-prefix-integrity.json)
for exact offsets, hashes, timings and budget snapshots.
