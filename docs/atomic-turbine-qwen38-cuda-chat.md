# Qwen4Exp CUDA chat template and natural EOS

## Selected result

The qualified Ubuntu / RTX 4070 Laptop snapshot passes one non-thinking
chat turn on the existing Qwen3.8 GSQ-RCO IQ3_XXS checkpoint. The request
`Say only OK.` uses `chat_template_kwargs={"enable_thinking":false}`, greedy
sampling and a maximum of 16 output tokens. Both ordinary CUDA and streaming
return `OK`, with output IDs `[3793, 248046]`. The second token is natural EOS;
generation ends after two tokens, before the limit, with no stop strings.

The server's `/apply-template` output and `/tokenize` result match across
modes: 16 input tokens, including the closed thinking prefix. Both the raw
`/completion` control on that rendered prompt and `/v1/chat/completions`
report internal `stop_type=eos`. The chat API also reports
`finish_reason=stop`; that field alone would not distinguish EOS from a stop
string. The slot is idle after completion.

Complete saved slots match ordinary CUDA byte for byte after the corresponding
raw and chat requests. Each file is 118,563,052 bytes with SHA-256
`5da197913308f502dabfeadb26b6e1286e0eaffa3b8d27ff37d41993152ca477`.
The comparisons preserve request history and apply no numerical tolerance or
state normalization. This checks output IDs and serialized state, not a new
capture of full-vocabulary logits.

## Budget and isolation

The profile keeps context 4096, batch 128, microbatch eight, 22 GPU layers,
four threads, one slot, disabled external RAM prompt cache and disabled
context checkpoints. Streaming caps remain 4 GiB RAM / 2 GiB VRAM, including
the unchanged 2 GiB runtime envelope. Observed managed high-water counters
after raw/chat state saves are 4,247,577,990 RAM bytes (4050.806 MiB) and
2,147,483,648 VRAM bytes. Both fit their caps. These counters describe the
shared engine ledger, not process RSS, OS file cache or every driver allocation.

No production source, backend, defaults or ordinary execution path changed
for this gate. The existing qualified server binary and libraries were reused
without a rebuild. A separate opt-in diagnostic starts only its own loopback
server and terminates it on exit. Ordinary CUDA exposes no managed-engine
metrics in this run. Its reference uses `--cpu-moe --moe-cache-mib 4096
--moe-cache-cpu-share 0` to retain GPU execution of selected experts; its total
memory allowance is larger, so this is not an equal-budget performance study.

[Recorded evidence](atomic-turbine-qwen38-cuda-chat.json) contains the exact
request, rendered template, input/output IDs, state hashes, metric snapshots,
commands and binary fingerprints. The template SHA-256 is
`12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce`.
The selected server SHA-256 is
`4e392b688ec18e955bf1455493f66f9e727f29141e6a7c23d90a8765d2a7b6a5`.

## Reproduction and remaining boundaries

With the qualified CUDA runtime on the library search path, select these two
controls sequentially and use a fresh evidence directory. `MODEL` names the
first GGUF shard; both shards must be available. Each mode runs only the raw
template control and its matching chat request, not an aggregate server suite.

```sh
python3 tools/expert-source/qwen38-cuda-chat.py --server build-cuda/bin/llama-server \
  --model "$MODEL" --output "$EVIDENCE" --mode native
python3 tools/expert-source/qwen38-cuda-chat.py --server build-cuda/bin/llama-server \
  --model "$MODEL" --output "$EVIDENCE" --mode stream
```

This extends the [raw server lifecycle qualification](atomic-turbine-qwen38-cuda-server.md)
with one selected non-thinking chat template and natural EOS case.
The subsequent [long HTTP prefill control](atomic-turbine-qwen38-cuda-http-prefill.md)
passes for the selected raw 3,481-token request, including observed server
batch boundaries and exact full slot bytes.

## Two chat turns in the same slot

The same single-slot CUDA profile also passes a selected non-thinking dialogue
with `cache_prompt=true`. After `Say only OK.` returns `OK`, the client sends
that assistant message in the history and asks `Repeat your previous answer
exactly.` The first turn evaluates 16 input tokens. The second prompt renders
to 37 tokens, of which 17 are reused from the live slot and 20 evaluated in
both ordinary CUDA and streaming. Both turns return `[3793, 248046]` and stop
at natural EOS. Disk prefix checkpoints are disabled for this check.

All four full FP32 sampling rows (3,973,120 bytes) and both complete saved
slot files match ordinary CUDA byte for byte. The combined logits SHA-256 is
`f6190d0d1044c3f1702ca925cede9f9de5ecace4232b485a6d3bba87f52b8ca2`.
The slot SHA-256 values are
`5da197913308f502dabfeadb26b6e1286e0eaffa3b8d27ff37d41993152ca477` and
`fa34bbfd369fa205ce2c6dddaafa33ada087dc7ec6c0cbefa9e5ab8f9076377f`.
The slot returns to idle after each request. Managed peaks are 4,247,558,854
RAM bytes and 2,147,483,648 VRAM bytes. Ordinary CUDA exposes no managed
engine metrics. No production sources changed for this check.

Run the two commands above sequentially with `--follow-up` and a fresh
evidence directory. Put the isolated FP32 sampling capture library on the
library search path ahead of the usual server libraries. The check used
capture library SHA-256
`247a984d45e8acfabb15527dbd87b7fb8d5309dfca4efb16d371175433cb392f`.
The driver compares the complete captured files and complete saved slots.
This short check does not measure sustained throughput.

## Long two-turn chat over SSE

The selected long-dialogue fixture uses the
[field archive](atomic-turbine-qwen38-context.txt) with its final question
replaced by `Reply with only the word OK.` The second request carries the
whole first user message and the assistant's response, then asks `Repeat your
previous answer exactly.` Both requests use the model chat template with
thinking disabled, greedy sampling, `cache_prompt=true` and slot zero. The
4 GiB RAM / 2 GiB VRAM streaming setup keeps context 4096, batch 128,
microbatch 8, 22 GPU layers and four threads. Disk prefix checkpoints, RAM
prompt caching and context checkpoints are disabled. The follow-up can reuse
only the prefix in the live slot.

The first request processes 3472 tokens in 28 server batches. On the second
request both modes reuse 3473 of 3493 tokens from slot zero and evaluate only
the 20-token suffix. Both turns return `OK`, end at natural EOS and leave the
slot idle. Observed TTFT is 685.648 s on the first turn and 3.731 s on the
second under streaming, versus 201.191 s and 1.202 s on ordinary CUDA.

All four full-vocabulary FP32 sampling rows (3,973,120 bytes) and both
complete slot files match ordinary CUDA byte for byte. The combined logits
SHA-256 is `793e40572967dd7ef227160892dcc295752623c84c2bd277f73c49af166cdd9b`.
The first slot is 224,911,084 bytes with SHA-256
`9dfb33891660d1c835bebbc3acff4f902b19583e2c7ad43ea970bb09f8c5d38b`.
The second is 225,557,296 bytes with SHA-256
`0f45594a65241c61194da854680de17780c78271446c2a7f9559ba6accb09fc7`.
After saving each slot, the streaming ledger reports peaks of 4,253,478,070
RAM bytes and 2,147,483,648 VRAM bytes. Ordinary CUDA exposes no managed
engine metrics. The isolated comparison reuses the existing server and
capture binaries with no production changes.

TTFT measures elapsed wall time from the loopback HTTP POST to the first
nonempty assistant content delta in SSE. Role-only and prefill progress
events do not count. This includes HTTP and sampling capture overhead and
excludes server startup, template probes and slot saves. Each measurement is
one observation with uncontrolled OS file-cache state. The ordinary CUDA
reference has a larger memory allowance and uses the same two-turn history.

Run the two reproduction commands above sequentially with both `--follow-up`
and `--long-prompt docs/atomic-turbine-qwen38-context.txt`. Use a fresh
evidence directory and the same isolated FP32 capture library used for the
short two-turn check. The driver checks that the follow-up reuses the long
prefix, compares full sampling captures and both complete slot files, and
records SSE timings and managed memory separately by turn.

## Two turns with thinking and preserved history

The selected thinking check uses the same RTX 4070 Laptop server geometry and
fixed 4 GiB RAM / 2 GiB VRAM caps. The first user message asks for `7 + 5`.
The second asks to add three to the previous answer and includes both the
assistant content and `reasoning_content` from the first response. Requests
set `enable_thinking=true`, `reasoning_effort=low`, `preserve_thinking=true`,
`reasoning_format=deepseek` and `cache_prompt=true`. Both modes use greedy
sampling, slot zero and no disk prefix checkpoints.

The first turn evaluates all 57 input tokens. The second renders to 122
tokens. Both modes reuse 93 tokens from the live slot and evaluate 29 new
tokens. Both return `12` then `15`, with nonempty reasoning on both turns,
natural EOS and an idle slot after each response.

- Streaming TTFT to the first nonempty reasoning delta is 10.281 s and
  5.453 s. TTFT to the first answer delta is 20.132 s and 17.080 s.
- Ordinary CUDA TTFT to reasoning is 3.256 s and 1.660 s. TTFT to the
  answer is 6.965 s and 5.855 s.

Timings start immediately before each loopback HTTP POST. Each mode has one
run with uncontrolled OS file-cache state. Ordinary CUDA uses a larger
memory allowance. The streaming ledger reports peaks of 4,247,577,622 RAM
bytes and 2,147,483,648 VRAM bytes after both slot saves.

The 37 and 43 full-vocabulary FP32 sampling rows match ordinary CUDA byte
for byte, as do both complete slot files. The combined 79,462,400-byte logit
capture has SHA-256
`5c0a8a9cdf00a376c271a5b16e5b4d6a2b4371b0c607f80c75cc92f85470bb14`.
The slot SHA-256 values are
`4a87e339029292a2139b43ad4d569e5f30a512949a9083075fe00544559447bc` and
`6e94c29f341bef6ce76b4e5c59b57f875791c4a5149ea5f2731611e196088df7`.

Run the two reproduction commands above with `--follow-up --thinking`, a
fresh evidence directory and the same isolated FP32 capture library. This
qualifies the selected two-turn low-effort fixture with preserved reasoning.
Other thinking settings, thinking combined with tools, long thinking conversations, multimodal
requests, other checkpoints, concurrent slots, automatic pressure handling
and HTTP resize remain unqualified.

## Single tool call and result in the same slot

The selected tool check uses the same CUDA geometry and 4 GiB RAM / 2 GiB
VRAM caps with thinking disabled, `cache_prompt=true` and no disk prefix
checkpoints. The first request requires one function call. The model emits
`lookup_status` with `{"key":"alpha"}` and finishes with `tool_calls`. The
driver returns `{"status":"ZX-418"}` in a tool message correlated with the
server-generated call ID. The second request keeps the tool schema and
switches to `tool_choice=auto`. The model answers `ZX-418` and stops at EOS
without another call. Both requests use slot zero and leave it idle.

The first request evaluates all 297 input tokens. For the second request,
both modes reuse 322 of 349 tokens from the live slot and evaluate 27 new
tokens. TTFT from HTTP POST to the first structured SSE tool delta is
55.755 s in streaming and 17.089 s in ordinary CUDA. From the second POST
to the first answer delta, TTFT is 4.882 s and 1.471 s, respectively. These
are single observations with uncontrolled OS file-cache state. The ordinary
CUDA reference has a larger memory allowance.

The 26 and 6 complete FP32 sampling rows match ordinary CUDA byte for byte,
as do the complete slot files saved after each request. The 31,784,960-byte
logit capture has SHA-256
`c99a866145d5fb3896a4da7298fbb049fef5c001bee9fe650c7f0e87ea7eb24c`.
The slot SHA-256 values are
`8c942ff209bf73ddb76027ee0c13bec9c882fd431d8a264746e66ebd6611058c` and
`31ac30a45b559633df8e5b60d4033e0ea1c710bda0f19b1743dbdebdad2748ec`.
The tool call IDs are transport metadata generated per run. Each result
uses its own call ID, and both rendered prompts and their token IDs are
compared directly. Managed peaks are 4,251,511,254 RAM bytes and
2,147,483,648 VRAM bytes.

Run the two reproduction commands above with `--follow-up --tool-call`, a
fresh evidence directory and the same isolated FP32 capture library. The
driver supplies a fixed result locally. This covers one function, one tool
result and one text response. Parallel calls and thinking with tools are
outside this check.
