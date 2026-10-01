# Decision mode (`llama-server --decision`)

`llama-server --decision -m model.gguf` (or `-m <laya checkpoint dir>`, see "Loading a Hugging
Face checkpoint directly") serves a *decision model*: a small model that
returns calibrated probabilities over a fixed set of options in one forward pass, with
no generation. It is a separate process role of this fork. It has no chat routes, no
slots, no KV cache and no web UI.

Engines (layouts):

| layout | models | runtime | status |
|---|---|---|---|
| `laya` | Laya family: `laya-multilingual` (mmBERT-base encoder), `laya` and `laya-typed-decisions` (English, ModernBERT-large encoder), each with the marker head | `tools/laya`, own ggml graph, CPU only | done |
| `semif-letters` | Arbiter-4B, JevK5 (Qwen3.5 + letter readout) | libllama | not supported yet: load fails with a clear error |

Code map:

| path | role |
|---|---|
| `tools/server/server-decision.{h,cpp}` | HTTP routes, FIFO worker, error envelope, info routes |
| `tools/decision/` | static library: py-json, `decision.spec`, request checks, calibration, executor cards, engines, laya input building (`laya-decide`), the GGUF cache for `-m <checkpoint dir>` (`decision-checkpoint.h`) |
| `tools/laya/` | laya runtime (port of upstream PR #29363; links only ggml), `llama-laya-cli`, and `llama-laya-convert` (HF checkpoint -> GGUF without Python, `laya-convert.h`) |
| `gguf-py/gguf/scripts/gguf_decision_spec.py` | get / set / verify `decision.spec` in a GGUF |
| `tests/test-decision-*.cpp`, `tests/decision/` | C++ unit tests, golden files, tiny model generator |
| `tools/server/tests/unit/test_decision.py` | server contract tests (offline) |
| `tools/decision/decision-bench.cpp`, `tests/decision/bench/`, `scripts/bench-decision*` | `llama-decision-bench`, its request suite, the KPI script and report (see Benchmark) |

## Quick start

```bash
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build -j --target llama-server llama-laya-cli
./build/bin/llama-server --decision -m laya-multilingual-Q8_0.gguf --device none -t 4 --port 8090

curl -s localhost:8090/props | jq .decision
curl -s localhost:8090/v1/systemone -H 'content-type: application/json' \
  -d '{"state":"Billed twice, please refund","questions":{"refund":{"type":"noul","instructions":"Asks for money back?"}}}'
```

Start-up order: the HTTP server starts first (so `/health` answers 503 while the model
loads), then a checkpoint directory is converted (or found in the cache), then the spec and
the engine load, then the server is ready. A load error stops
the process with exit code 1.

## Flags

Environment variables use the `LLAMA_ARG_DECISION_*` names.

| flag | default | meaning |
|---|---|---|
| `--decision` | off | decision mode. Requires `-m FILE` or `-m DIR` (a laya Hugging Face checkpoint directory, converted once into a GGUF cache); `-hf`, `-dr` and `-mu` are rejected (without `-m` the server would start in router mode). Explicit only: a stamped GGUF never switches mode on its own. |
| `--decision-convert-cache DIR` | `$LLAMA_CACHE/laya/gguf-cache`, else the user cache + `llama.cpp/laya/gguf-cache` | `-m DIR` only: where the converted GGUF is kept (never in the checkpoint directory). See "Loading a Hugging Face checkpoint directly" |
| `--decision-convert-type T` | `f16` | `-m DIR` only: `f16` or `f32`. The converter's `q8_0` is not the precision-protected recipe (see "Converting without Python"); for Q8_0 quantize the f16 GGUF with `tests/laya/quantize.sh` |
| `--decision-spec FILE` | from the GGUF | JSON that replaces the embedded `decision.spec` as a whole. A bare Arbiter `calibration.json` (`{"temperature": x}` or `{"temperature": {...}}`) is accepted too: it is a required calibration, keeps the layout clamp (laya [0.5, 5]) and replaces the `laya.temperature` calibration; its `calibration.version` is its own `version` field (else `cal-<hash>`), never `gguf:laya.temperature`. |
| `--decision-plan NAME` | from the spec | compute plan, laya: `sequential`, `packed` (systemone only; the router always runs `sequential`) |
| `--decision-kernels NAME` | from the spec, then `auto` | matmul kernels, laya: `auto` (`blas` when the BLAS backend is Accelerate, i.e. on macOS; else `default`), `default` (ggml CPU kernels, the Phase 1 numerics), `repack` (CPU repack buffers for quantized weights), `blas` (BLAS backend in the scheduler), `repack+blas`. Kernels change the logits slightly, so the default logits on macOS differ from those on Linux and Windows; `default` gives the same numerics everywhere. An explicit `blas` / `repack+blas` fails at load on a build without a BLAS backend. See "CPU kernels and memory" |
| `--decision-queue N` | 4 | requests that may wait while one runs; more get 429 |
| `--decision-max-items N` | 16 | questions per `/v1/systemone` request, candidates per `/v1/router/score` request (router is also capped at 16) |
| `--decision-allow-uncalibrated` | off | serve `/v1/router/score` without a router calibration |
| `--decision-debug` | off | raw logits and token ids in answers, `POST /v1/decision/render`, request logging, test delays (below) |

Reused: `-m`, `-a/--alias` (default: the model file name without its directories and a trailing `.gguf`, for `-m DIR` the directory name; the same name as the default spec `model_id`), `--host`, `--port`, `--api-key` / `LLAMA_API_KEY`, `-t`
(engine threads, fixed at load; default: the performance cores, see below), `--load-mode`
(`mmap`, the default, keeps `token_embd` in a read-only file mapping; `none` / `dio` load it;
`mlock` / `mmap+mlock` also lock the model in RAM, and a failed lock only warns),
`--mlock` / `--no-mmap` (deprecated spellings of `--load-mode`), `--no-warmup` (skip the
warm-up forward pass at load), `--metrics`, `--timeout`, SSL flags. `--device` is
ignored by the laya engine (CPU only).

Ignored with a warning: `--parallel`, `-ctk/-ctv`, `--ctx-checkpoints`, `--spec-*` / `-md`,
`--embedding` / `--pooling`, `--mmproj`, `--lora`, `--sleep-idle-seconds`,
`--mcp-servers-*`, `--tools`, `--ui-mcp-proxy`, `--path`. The UI is always off.

A laya GGUF or checkpoint directory started without `--decision` fails fast with a hint. `--decision` inside a
router-mode child instance is rejected: run a separate process.

## API

Auth is the usual middleware: `Authorization: Bearer <key>` when `--api-key` is set.
`/health`, `/v1/health`, `/models` and `/v1/models` stay public. Probabilities are
doubles and are never rounded. Answers and scores keep the request order.

### `POST /v1/systemone`

A superset of TypeSafe `systemone`:

```json
{"model": "optional, ignored",
 "state": "<string | object | array | number | bool>",
 "questions": {
   "refund": {"type": "noul",   "instructions": "Asks for money back?"},
   "topic":  {"type": "choice", "instructions": "Topic?", "criteria": {"billing": "payments", "other": null}},
   "tone":   {"type": "choice", "instructions": "Tone?",  "criteria": ["calm", "angry"]},
   "sev":    {"type": "score",  "instructions": "Severity?", "criteria": ["low", "mid", "high"]},
   "urgent": {"type": "noul",   "instructions": "Urgent?", "criteria": {"True": "today"}, "labels": {"false": "later", "true": "now"}}},
 "truncation": "allow"}
```

Questions of the `laya` layout are read exactly as the Laya reference reads them (laya
0.3.21 `Agent._check_question`, `Agent._to_internal`, `common.render_options`;
`laya_question_parse` in `tools/decision/laya-decide.cpp`, shared with `llama-laya-cli`):

- `state` is required: absent or `null` is 400 `INVALID_REQUEST` (`param` `state`), as in
  the reference server (the text `null` would otherwise be scored). A string is used as is,
  anything else as `json.dumps(state, ensure_ascii=False)`. A list state is cut from the
  left (the newest turns stay), anything else from the right.
- `questions` is an object; an empty one gets `"answers": {}` and zero usage.
- `instructions` must be present (`EMPTY_INSTRUCTIONS` when it is not) and may be any JSON
  value: blank text is kept, a non-string is `json.dumps(value, ensure_ascii=False)`.
- `choice` criteria: an object `{label: description}` or a list of labels, with at least
  one option. The option text is `str(label)`, plus `": " + description` unless the
  description is `null` or `""`. A list label must be a string, number or bool; `null`, a
  nested value or a label equal to an earlier one by Python `==` (`1`, `1.0` and `true` are
  one label) is 400 `UNSUPPORTED_CRITERIA_VALUE`. Answer keys are the labels as `json.dumps`
  writes dict keys (`3` -> `"3"`, `true` -> `"true"`, `2.5` -> `"2.5"`), and `choice` is the
  label as given (a number stays a number).
- `score` criteria: a non-empty list of levels, none `null`; option text `level i: <level>`.
- `noul` criteria: absent, `null` or an object keyed `true` / `false` in any case (`True`,
  `FALSE`; a later duplicate wins); any other key is 400 `INVALID_NOUL_CRITERIA`. `labels`
  (noul only) is `null` or exactly `{"false": str, "true": str}`, stripped like Python
  `str.strip` (so U+2028 and U+0085 count as space), non-empty and distinct; it replaces the
  words `false` / `true` in the option texts. Bad labels are 400 `INVALID_NOUL_CRITERIA`
  (`param` `questions.<id>.labels`), labels on a choice or score question 400
  `UNSUPPORTED_CRITERIA_VALUE`.
- A structured description or level is compact JSON (`json.dumps`, `", "` / `": "`).
- `truncation`: `"allow"` (default) or `"error"`.

Where this differs from the TypeSafe / arbiter rules (`arbiter/prompt.py`
`normalized_question`), the laya layout follows the reference: blank or non-string
instructions are accepted, one choice option or score level is a valid question, list
labels do not collapse (`["a", "a"]` is an error instead of one option; `null` is an error
instead of `"None"`; `["1", 1]` are two labels), unknown noul criteria keys are an error
instead of being ignored, `labels` is accepted, a missing state is an error instead of the
text `null`, and empty `questions` is not an error. Two limits the reference does not have
stay: at most 20 options per question (`TOO_MANY_OPTIONS`, the marker slots of the laya
graph) and two labels with the same answer key (`["1", 1]`, `["true", true]`) are 400
`UNSUPPORTED_CRITERIA_VALUE`, since a JSON object cannot hold both keys (the reference
writes the key twice). Other layouts keep the TypeSafe rules: list entries become keys
through Python `str()` and collapse (`dict.fromkeys`), at least two options, non-blank
string instructions.

Response:

```json
{"model": "tiny-laya",
 "answers": {
   "refund": {"type": "noul", "noul": 0.93, "confidence": 0.93},
   "topic":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.97, "other": 0.03}, "confidence": 0.81},
   "sev":    {"type": "score", "score": 1.4, "probabilities": {"0": 0.2, "1": 0.2, "2": 0.6}, "legend": {"0": "low", "1": "mid", "2": "high"}, "confidence": 0.1}},
 "usage": {"input_tokens": 138, "output_tokens": 0, "evaluated_tokens": 138},
 "latency_ms": 42.1,
 "timings": {"queue_ms": 0.1, "render_ms": 0.4, "compute_ms": 41.5},
 "warnings": [],
 "runtime": {"layout": "laya", "format": "laya-v1", "plan": "sequential", "spec_sha256": "...", "calibration": "..."}}
```

- `noul` = P(true). `choice` = the first argmax in criteria order (laya: the label as
  given). `score` = sum(i * p_i); `legend` holds the request levels as given.
- `confidence` follows `spec.confidence`: `laya` (noul: max(p, 1 - p); choice/score:
  1 - H / ln K), `typesafe` (1 - H / ln K for all), `max_p`.
- Probabilities are `softmax(logits / T)` in double. T comes from the spec calibration
  (see below); an uncalibrated model uses T = 1, and so does a calibration that is not
  `required` for a question type or option count it has no temperature for.
- `input_tokens` sums the full prompt lengths; `evaluated_tokens` counts what was computed.
- `warnings`: `state_truncated` (laya cuts the state as the reference does: a list from the
  left, `state_ids[len - room:]`, anything else from the right), `options_truncated`, `instructions_truncated`.
  With `"truncation": "error"` any of these cuts is a 422 instead (`STATE_TRUNCATED`,
  `OPTIONS_TRUNCATED`, `PROMPT_TOO_LONG` with `param` `questions.<id>.instructions`).
- With `--decision-debug` every answer has `debug: {logits, temperature, tokens, n_state, n_state_cut, options_cut, head_cut}`.

### `POST /v1/router/score`

Scores each candidate executor independently: `p_success` values do not sum to 1, and
there is no threshold or argmax. Routing policy lives in the caller.

```json
{"task": "Extract invoice number, date and total",
 "criterion": "All three fields exact; nothing invented.",
 "candidates": [
   {"id": "local/qwen3.5-4b-q4_k_m",
    "card": {"schema": "atomic.executor-card/1", "name": "Qwen3.5-4B local", "kind": "local",
             "description": "4B general model on this laptop (Q4_K_M)",
             "checks": [
               {"skill": "field extraction", "status": "measured", "passed": 188, "total": 200,
                "criterion": "exact match", "source": "atomic-evals/extract", "version": "2026-09"},
               {"skill": "long-document QA", "status": "missing", "source": "atomic-evals/longqa", "version": "2026-09"}]}}],
 "truncation": "allow"}
```

```json
{"object": "router.scores", "model": "...",
 "scores": [{"id": "local/qwen3.5-4b-q4_k_m", "p_success": 0.9412, "logit": 2.78, "calibrated": true,
             "input_tokens": 431, "truncated_tokens": 0}],
 "usage": {"input_tokens": 431, "output_tokens": 0, "passes": 1},
 "latency_ms": 212.4, "timings": {"queue_ms": 0.1, "render_ms": 0.6, "compute_ms": 211.5},
 "runtime": {"layout": "laya", "format": "laya-v1", "plan": "sequential", "spec_sha256": "...", "calibration": "platt"},
 "warnings": []}
```

Card rules (`atomic.executor-card/1`): `name` and `kind` are required strings,
`description` is optional; at most 32 `checks`; every card string is at most 4096 bytes. A `measured` check needs integer `passed`
and `total` with 0 <= passed <= total and total >= 1. A `missing` check has no numbers.
Floats are rejected, so Python and C++ render the same text. Unknown fields are ignored
and reported in `warnings`. Candidate ids match `[A-Za-z0-9._:/@+-]{1,128}` and are unique.
`tools/decision/schema/executor-card-v1.schema.json` has the card as JSON Schema, and
`tools/decision/reference.py` has the Python rules and rendering (see "Training and calibration tools").

laya mapping: one `noul` sequence per candidate (cost: N encoder passes). The question comes
from `spec.router.question`. Without one (no `router` block, or no `question` in it) it is
the built-in question `{"type": "noul", "instructions": "Will the executor meet the success
criterion on this task?", "criteria": {"true": "meets the criterion", "false": "does not meet
the criterion"}}`. A spec question without `criteria` gets these built-in criteria. The
criteria are part of the prompt, so a router calibration is only valid for the question it
was fitted with. The state is the `card-v1` text, then the criterion, then the task, so a cut
only hits the end of the task (`truncated_tokens`, warning `task_truncated`):

```
executor: <name>
kind: <kind>
description: <description>
checks:
- <skill>: passed <passed> of <total>; criterion: <criterion>; source: <source> <version>
- <skill>: not measured; source: <source> <version>

success criterion: <criterion>

task: <task>
```

Card strings are made single-line (control characters become spaces, space runs collapse)
and absent optional parts are left out; a card without checks renders `checks: none`.

A card longer than `router.max_card_tokens` tokens is a 422 `CARD_TOO_LONG`. Without that
key the limit is 3/8 of the model sequence length (384 tokens for `laya-multilingual`, 1024);
`0` turns the limit off. `/props.decision.limits.max_card_tokens` shows the value in use.

Special-token text (`spec.special_tokens`) in the state: the mask literal (`<mask>`) always
becomes a space, as in the Laya reference (`mask-to-space`, the default). With
`escape-control` the card, criterion and task also have every CONTROL-token string of the
vocabulary (`tokenizer.ggml.token_type` 3) and every `<unusedN>` added token replaced by one
space, so user text cannot inject structure. For `laya-multilingual` that set is `<pad>`,
`<eos>`, `<bos>`, `<unk>`, `<mask>`, `<2mass>`, `[@BOS@]`, `<start_of_turn>`,
`<end_of_turn>` and `<unused0>` ... `<unused99>`; other added tokens (newline and tab runs,
HTML tags) stay tokens. For the English checkpoints it is `[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`,
`[MASK]`, `<|padding|>`, `<|endoftext|>` and `[unused0]` ... `[unused82]`; the space runs and
`|||IP_ADDRESS|||`, `|||EMAIL_ADDRESS|||`, `|||PHONE_NUMBER|||` stay tokens. The strings cannot overlap and a space cannot form a new one, so
replacing them one after another in any order gives the same text (`reference.py` can use
`str.replace`). Token counts for `CARD_TOO_LONG` and `CRITERION_TOO_LONG` use the same
replacement. Systemone questions and states always use `mask-to-space`.

`logit` is the raw z = s_true - s_false. With a router calibration,
p = sigmoid(a * z + b) (`spec.router.calibration`, Platt). Without one the route answers
501 `ROUTER_NOT_CALIBRATED` to every request, before the body is read. With `--decision-allow-uncalibrated` it answers
`calibrated: false` and p = sigmoid(z), the zero-shot baseline. The router always runs
plan `sequential` (`runtime.plan`, `/props.decision.plan.router`), even with
`--decision-plan packed`: adding or removing candidates never changes the other
candidates' scores (bitwise).

### Info routes

- `GET /health`, `/v1/health`: `{"status": "ok", "ok": true, "model": ..., "layout": ...}`.
- `GET /v1/models`, `/models`: one entry with `capabilities` (`decision`, `systemone`, plus
  `router_score` only when a router calibration exists) and
  `decision: {api_version, layout, model_id, model_version}`.
- `GET /props`: `model_alias`, `model_path`, `build_info`, and a `decision` block with
  `api_version` (1), `endpoints`, `layout`, `format`, `model_id`, `model_version`,
  `spec_version`, `spec_sha256`, `spec_source` (`gguf`, `file` or `default`),
  `question_types`, `limits` (`max_questions`, `max_candidates`, `max_options`,
  `max_checks`, `max_tokens`, `max_card_tokens`, `max_card_field_bytes`, `max_body_bytes`), `confidence`,
  `calibration {method, calibrated, version}`, `router {available, calibrated, method, card_schema, card_renderer}`,
  `plan {name, router, plans, n_threads, kernels, n_threads_blas, state_split, recipe, fa, n_batch, n_ubatch}`
  (`router`: the plan of `/v1/router/score`; `kernels`: `cpu`, `cpu+repack`, `cpu+blas` or
  `cpu+repack+blas`; `n_threads_blas`: threads of the BLAS backend, 0 without it; `state_split`:
  router state pieces are tokenized separately, see Scheduling),
  `memory {weights_loaded_bytes, weights_mapped_bytes, weights_repacked_bytes}` (where the
  weights went, see "CPU kernels and memory"), `device`, `kv_type`,
  `queue_capacity`, `input_contract`, `special_tokens`, `debug`,
  `source` (`gguf` for `-m FILE`, `checkpoint-dir` for `-m DIR`), `cache_path` (the cached GGUF
  that was loaded; `null` for `gguf`) and `checkpoint` (`null` for `gguf`, else
  `{dir, cache_dir, key, outtype, cache_hit, convert_ms, converter}`). `model_path` is `-m` as given.
- `GET /metrics` (only with `--metrics`): Prometheus counters `llamacpp:decision_requests_total`,
  `_errors_total`, `_rejected_total`, `_items_total`, `_tokens_total`,
  `_compute_seconds_total` and the gauge `llamacpp:decision_queue_waiting`.
- `POST /v1/decision/render` (only with `--decision-debug`): takes a systemone or router
  body (router if it has `candidates`) and returns per item the question, keys, state, token
  ids and cut info, without compute.

A client that checks for decision support: `/health` 200 -> `/v1/models` has `decision` ->
`/props.decision.api_version == 1`.

### Errors

Decision handlers answer `{"error": {"code", "type", "reason", "message", "param"?}}`;
`param` is the JSON path of the bad field.

| HTTP | reason |
|---|---|
| 400 | `MALFORMED_JSON`, `BODY_NOT_OBJECT`, `INVALID_REQUEST` (missing `questions` / `task` / `criterion` / `candidates`, laya: missing or `null` `state`, bad `truncation`), `UNKNOWN_QUESTION_TYPE`, `EMPTY_INSTRUCTIONS` (laya: no `instructions` key), `TOO_FEW_OPTIONS`, `TOO_MANY_OPTIONS`, `INVALID_NOUL_CRITERIA`, `UNSUPPORTED_NUMBER` (NaN/Infinity, integers outside int64/uint64, float overflow), `UNSUPPORTED_CRITERIA_VALUE`, `TOO_MANY_QUESTIONS`, `INVALID_CARD` (also a card string over 4096 bytes), `DUPLICATE_CANDIDATE_ID`, `INVALID_CANDIDATE_ID`, `TOO_MANY_CANDIDATES` |
| 413 | `BODY_TOO_LARGE` (body over 1 MiB, checked in the handler) |
| 422 | `PROMPT_TOO_LONG` (instructions cut with `"truncation": "error"`; letters engine: prompt too long), `CARD_TOO_LONG` (card over `router.max_card_tokens`), `CRITERION_TOO_LONG` (card + criterion do not fit), `OPTIONS_TRUNCATED` (option markers cut, or any option cut with `"truncation": "error"`), `STATE_TRUNCATED` (with `"truncation": "error"`) |
| 429 | `OVERLOADED`, with `Retry-After: 1` |
| 500 | `INTERNAL` |
| 501 | `ROUTER_NOT_CALIBRATED` |
| 503 | `UNAVAILABLE`: an accepted request was not run or not finished because the server is stopping or the client left |

Not from decision handlers, so without `reason`: 401 (API key middleware), 503 while
loading (state middleware), 404 for unknown routes (httplib rewrites any 404 body, so
decision handlers never use 404), and 413 for bodies over httplib's 100 MB limit. httplib
adds `Connection: close` to responses with status >= 400.

## Scheduling and determinism

One engine, one request at a time, in arrival order. Parsing and validation run on the
HTTP thread; rendering and compute run on a single worker thread. With `--decision-queue N`,
up to N requests wait while one runs (an idle worker always takes a request); the next one
gets 429. When the client disconnects, the request is cancelled: it is skipped if it has
not started, or stops between items (while rendering or between forward passes), and a
cancelled request no longer takes a place in the queue; a running request that is cancelled
does not count as busy either, so with `--decision-queue 0` the next request is taken
instead of getting 429. On shutdown (SIGINT/SIGTERM) the worker stops accepting requests
(new ones get 503 `UNAVAILABLE`), and every waiting and running request is cancelled the
same way, all under the worker lock, so no request slips in after the cancel and the process
does not wait for the queue. The worker never touches the HTTP request object.

Tokenization matches the HF tokenizer of the checkpoint (`tests/laya/verify_tokenizer.py`)
and costs O(n log n) in the text length. `tools/laya` ports both tokenizers of the family; the
GGUF key `decision.laya.tokenizer` picks one (a GGUF without it is `metaspace-bpe`):

| `decision.laya.tokenizer` | checkpoints | steps (HF `tokenizer(text, add_special_tokens=False)`) |
|---|---|---|
| `metaspace-bpe` | `laya-multilingual` | added tokens on the raw text, `" "` -> U+2581, Metaspace words, BPE with byte fallback |
| `bytelevel-bpe` | `laya`, `laya-typed-decisions` | special added tokens on the raw text; NFC (`decision.laya.normalizer`); normalized added tokens (space runs, `[unusedN]`); GPT-2 regex; byte-to-unicode; BPE |

The converter also stores the HF flags of the added tokens (`decision.laya.added_tokens`: id,
flags pairs; lstrip, normalized) and refuses a tokenizer that needs anything else (rstrip,
single_word, another normalizer or pre-tokenizer). NFC and the regex classes use the Unicode
tables of HF tokenizers itself (`tools/laya/laya-unicode-data.inc`, probed from `tokenizers`
by `tools/laya/gen-unicode-data.py`, which checks them against HF NFC on every codepoint):
Python's `unicodedata` is newer and would differ on marks added after Unicode 10.
libllama's vocab (vocab-only load) was not reused: on the English GGUF `llama-tokenize` skips NFC
(`cafe` + U+0301 ` is here` gives 6 tokens where HF gives 5) and ignores the lstrip of `[MASK]`
(`a [MASK] b` gives 66 209 50284 270 where HF gives 66 50284 270), and `tools/laya` would have to
link libllama instead of ggml alone.
For both, long words without spaces (CJK, base64, URLs) stay cheap (a 1 MB input takes well
under a second). Each request tokenizes its state once:
all questions of a systemone request share the state tokens, and router candidates share
the tokens of the criterion and the task. For that the router state splits after the card
and after `task:`, and each piece is tokenized on its own. That gives the tokens of the
whole text only where both points are token boundaries of the vocabulary (for
`laya-multilingual`: `\n\n` is an added token and a space starts a Metaspace word; for the
English checkpoints the GPT-2 regex already cuts there), so at
load the engine checks tokenize(A + B) == tokenize(A) + tokenize(B) on card endings and
task starts of every kind (letters, digits, CJK, punctuation, spaces, newlines, special-token
text, with and without `escape-control`). If any probe differs, router states are tokenized
whole: the same tokens, without the sharing. `/props.decision.plan.state_split` shows the
result.

Plans (laya):

- `sequential` (default): one graph per question or candidate, no state between calls.
  Results are bitwise stable for a fixed `-t`, and items are independent.
- `packed`: the items of a systemone request in one graph with a block-diagonal mask (from
  PR #29363). Opt-in; per the PR README it is not faster on CPU. Measured against
  `sequential`: max |dlogit| 2e-6 on F32, 1e-2 on F16, 7e-2 on Q8_0. The graph holds an
  n_tok x n_tok attention matrix per head, so items are packed in order into groups of at
  most 2048 tokens (about 230 MB for mmBERT-base); a larger request runs several graphs.
  Never used for `/v1/router/score`, where a candidate's score must not depend on the others.

There is no batching across requests and no inexact prefix reuse. Settings that change
numerics (plan, threads, kernels, recipe) are shown in `/props.decision.plan`.

### CPU kernels and memory

- Threads: without `-t` the engine runs on the performance cores (macOS
  `hw.perflevel0.physicalcpu`, which is also the common default there; Windows the cores of
  the highest `EfficiencyClass`, while the common default counts the efficiency cores too;
  other systems keep the common default), for every kernels choice (measured below). The CPU
  backend gets one persistent ggml
  threadpool, created at load (polling level 0 between graphs: idle workers sleep; within a
  graph they spin on the barrier), so the CPU backend creates no threads per graph. The BLAS
  backend is different: without OpenMP, ggml-blas converts the weights of every matmul it
  takes to F32 on `n - 1` new `std::async` threads (about 90 conversions per forward pass of
  mmBERT-base). The engine therefore gives the BLAS backend min(`-t`, 8) threads
  (`/props.decision.plan.n_threads_blas`; measured below); Accelerate runs the sgemm itself
  on its own threads. On macOS the worker thread that runs the graphs is raised to QoS
  `USER_INITIATED`. OpenMP builds get `KMP_BLOCKTIME=0` and `OMP_WAIT_POLICY=passive` unless
  set, at process start before the HTTP threads exist (LLVM libomp reads them at its first
  parallel region; GNU libgomp only at program start, so set them in the environment there).
- Memory: `token_embd` (393 MB F16 / 209 MB Q8_0 for mmBERT-base, read only by `get_rows`)
  stays in a read-only mapping of the GGUF, so only the rows of the tokens seen become
  resident; every other weight is loaded. The GGUF header, vocabulary strings and merges are
  freed after the load. `--load-mode none` loads everything; results are bitwise the same.
  `/props.decision.memory` shows the weight bytes loaded, mapped and repacked.
- Warm-up: one 64-token forward pass at load (starts the threads, faults in the weights
  that every pass reads, and sizes the compute buffer for 64 tokens); `--no-warmup` skips it.
  It changes no result. It does not bound the compute memory: the scheduler grows the buffer
  again on the first longer request (measured below).
- Kernels (`--decision-kernels`, spec `plan.kernels`, `/props.decision.plan.kernels` shows
  what runs: `cpu`, `cpu+repack`, `cpu+blas`, `cpu+repack+blas`):
  - `default`: the ggml CPU kernels. F16 matmuls round the activations to F16, Q8_0 matmuls
    quantize them to Q8_0. `llama-laya-cli`, the golden files and
    `tests/laya/verify_precision.py` use it.
  - `blas`: the BLAS backend takes the matmuls it supports (weights converted to F32, F32
    sgemm); on macOS that is Accelerate (AMX). Closer to the PyTorch reference and faster
    than `default` on both F16 and Q8_0 (table below). `auto` picks it when the BLAS backend
    is Accelerate; other BLAS libraries are not measured and stay opt-in. ggml-blas takes a
    matmul only when all its dimensions are at least 32, so sequences shorter than 32 tokens
    and the head matmuls over the marker rows still run on the ggml CPU kernels, while
    `/props` says `cpu+blas`. An explicit `blas` on a build without a BLAS backend fails at
    load (it does not quietly compute with other kernels).
  - `repack`: CPU repack buffers (`ggml_backend_dev_get_extra_bufts`) for quantized matmul
    weights that have a repacked kernel (Q8_0 / Q4_0 / Q4_K on NEON dotprod / i8mm, Q4_0 /
    Q4_K on AVX2, ...). F16 weights are never repacked. Same accuracy class as `default`,
    different rounding.
  - `repack+blas`: repacked weights are not host memory, so BLAS never takes them. With Q8_0
    (where every encoder matmul is repacked) it computes exactly like `repack`; BLAS only
    takes the weights that have no repacked kernel (F16, or types without one).
  - Platforms: the default `auto` means `cpu+blas` on macOS and `cpu` on Linux and Windows, so
    the same GGUF gives slightly different logits there. Through the server, `laya-multilingual`
    F16 is within max |dlogit| 0.014 of the reference with `auto` on macOS and 1.09 with
    `default` (the kernels Linux and Windows run; measured on this Mac, the x86 kernels round
    differently and are not measured); parity tables below. `--decision-kernels default` (or spec
    `plan.kernels: "default"`) restores the Phase 1 numerics bit for bit on every platform.
  - Results are bitwise stable for a fixed model, `-t` and kernels on one machine.

## Model metadata: `decision.spec`

`decision.spec` is a GGUF string holding UTF-8 JSON. It is the source of truth. The mirror
keys `decision.spec_version` (u32), `decision.layout`, `decision.model_id` and
`decision.model_version` are for readers that skip the JSON; the server checks that they
agree with the spec. `laya.*` keys hold architecture hyperparameters only.

Where the spec comes from, in order:

1. `--decision-spec FILE` replaces it as a whole (`spec_source: file`).
2. The GGUF `decision.spec` (`spec_source: gguf`).
3. Defaults for an unstamped laya GGUF (`spec_source: default`): calibration from
   `laya.temperature` and `laya.temperature_by_options.{buckets,values}` unless all
   temperatures are 1.0 and there are no buckets (then `calibrated: false`, which is the case
   for `laya-multilingual`), no router calibration. A non-laya GGUF without a spec does not
   load. The checkpoint's `temperature_by_options` (`"choice:3-5": 1.76`, ...) becomes the
   bucket of that type and the per-type temperature the `"*"` bucket, which is the lookup of
   the Laya reference (`laya.agent._decode_answers`); its bucket `2` means K <= 2 and is
   stored as `"1-2"`. The English `laya` ships `choice:11+` = 0.1006, which the [0.5, 5]
   clamp turns into 0.5, as the reference does.

A laya spec (file or GGUF) without a `calibration` key keeps the calibration from
`laya.temperature` of case 3 (`calibration.version` `gguf:laya.temperature`). To turn it
off, give `"calibration": {"method": "none"}` or `"calibration": null`.

`spec_sha256` is the SHA-256 of the exact spec bytes (for defaults: of their py-json
text). It is in `/props` and in every response.

Example (router on laya):

```json
{"spec_version": 1, "model_id": "atomic/router-laya", "model_version": "1.0.0", "layout": "laya",
 "input_contract": "laya-router-v1", "special_tokens": "escape-control",
 "limits": {"max_options": 20, "max_candidates": 16},
 "calibration": {"method": "temperature", "required": true, "clamp": [0.5, 5.0],
   "temperature": {"noul": {"2": 1.37}, "choice": {"2": 1.2, "3-5": 1.45, "6-10": 1.6, "11+": 1.8}, "score": {"*": 1.3}}},
 "confidence": "laya", "plan": {"name": "sequential", "recipe": "Q8_0-mixed", "n_threads": 4},
 "router": {"card_schema": "atomic.executor-card/1", "card_renderer": "card-v1", "max_card_tokens": 384,
   "question": {"type": "noul", "instructions": "Will the executor meet the success criterion on this task?",
                "criteria": {"true": "meets the criterion", "false": "does not meet the criterion"}},
   "calibration": {"method": "platt", "a": 1.0, "b": 0.0}}}
```

Calibration rules:

- Temperature lookup: the question type entry, else `all`. Inside an entry, by option
  count K: exact `"N"`, then range `"A-B"`, then `"N+"`, then `"*"`. A temperature found
  this way is clamped to `clamp`.
- With no entry or no matching bucket the temperature is exactly 1 (not clamped). Only a
  `required: true` calibration rules this out: it needs a temperature for every question
  type and option count the model accepts (noul: 2; choice and score: 2 to
  `limits.max_options`, capped by the layout: laya 20, semif-letters 16), through the type
  entry or `all` and a matching bucket, and a gap is a load error that names it (for example
  "choice with 6 options"). A calibration without `required` can fall back to T = 1 for the
  cases it does not cover. A bare `calibration.json` sidecar is always `required`.
- Temperatures and Platt parameters are fitted on the raw logits of this engine, for this
  GGUF, recipe, plan and kernels. `plan.kernels` (`auto`, `default`, `repack`, `blas`,
  `repack+blas`) pins the kernels the calibration was fitted with; `--decision-kernels`
  overrides it. Any other value (also a non-string) is a load error, and `gguf_decision_spec.py`
  refuses it too.

A router question (`router.question`) follows the laya question rules above: criteria keyed
only `true` / `false` in any case, and `labels`, when given, valid noul labels; the loader and
`gguf_decision_spec.py` both reject anything else.

Contracts: the loader rejects values the engine does not implement. laya: `input_contract`
`laya-v1` (default) or `laya-router-v1` (the router state above, needs `special_tokens`
`escape-control`), `special_tokens` `mask-to-space` (default) or `escape-control`.
`semif-letters`: `semif-v1` and `escape`.

Stamping without re-converting:

```bash
python gguf-py/gguf/scripts/gguf_decision_spec.py set    model.gguf spec.json   # in place, or -o out.gguf
python gguf-py/gguf/scripts/gguf_decision_spec.py get    model.gguf
python gguf-py/gguf/scripts/gguf_decision_spec.py verify model.gguf [spec.json]
```

`set` and `verify` check the spec with the rules of the server loader, so a spec they accept
also loads. Both implementations run the cases in `tests/decision/spec_cases.json`
(`test-decision-calib` and `test_decision.py`): a new rule goes into both and gets a case.
`set` takes a full spec only, not a bare `calibration.json`.

## Training and calibration tools

Python tools for the model team, so that training data, calibration and baselines use the text
and the logits of this engine. They are offline and write no GGUF.

**`tools/decision/reference.py`** (stdlib only) is the Python reference of the router input.
The training pipeline imports it instead of re-implementing the prompt:

- `py_dumps` / `loads_strict` / `parse_body`: `json.dumps(v, ensure_ascii=False)` and the
  engine's strict JSON parse (NaN and Infinity, integers outside int64/uint64 and float
  overflow are `UNSUPPORTED_NUMBER`; lone surrogates and more than 128 levels are
  `MALFORMED_JSON`; a duplicate key keeps its first position and its last value; one leading
  UTF-8 BOM is skipped).
- `card_validate` / `card_render`: the `atomic.executor-card/1` rules and the `card-v1` text,
  with the engine's reason, param and warnings.
- `parse_router` / `router_state` / `state_splits`: the `/v1/router/score` checks and the
  candidate state (card, then criterion, then task).
- `router_question` / `noul_options`: the spec router question (built-in default, default
  criteria) and its laya option texts.
- `Escape` / `control_strings_from_gguf`: `mask-to-space` and `escape-control`. The escape set
  comes from the GGUF (a stdlib GGUF metadata reader) exactly as `laya_escape_init` builds it.
- `render_router(body, spec, esc)`: all of the above for one request. `model_text` is the
  state the tokenizer reads. `laya_router_examples` gives one Laya noul example per candidate.

```bash
python3 tools/decision/reference.py control-tokens laya-multilingual-q8_0.gguf > control.json
python3 tools/decision/reference.py router request.json --spec spec.json --control control.json
python3 tools/decision/reference.py router request.json --spec spec.json --gguf model.gguf --candidate ID --text model
python3 tools/decision/reference.py card card.json        # validate + card-v1 text
```

Token counts (`CARD_TOO_LONG`, `CRITERION_TOO_LONG`, truncation) need the tokenizer and are
not in the reference; the render route shows them (`POST /v1/decision/render` with
`--decision-debug`).

How the reference is tied to the engine:

- `tests/decision/gen_router_golden.py` writes `tests/decision/golden/router_cases.jsonl`:
  540 exact request bodies (fixed parser cases such as a leading BOM, `-NaN`, duplicate keys,
  128 nesting levels, and two failures in one body, where the first in document order decides
  the reason as in the engine's SAX parser; 3 cases per card and request rule; random requests
  with unicode, C0/C1 controls, control-token text, 4096-byte fields, missing checks and unknown
  fields; 421 accepted, 1132 candidate states), each with the 400 reason and param from
  `reference.py`, or a SHA-256 over its warnings, the question's type and instructions, options,
  states, split offsets and escaped model texts. The escape sets in the file are those of
  `laya-multilingual` and of the English checkpoints, written into it: ctest does not read
  GGUFs, and the check against the real vocabularies (last point) runs only on a machine with
  the models.
- `test-decision-reference` (`ctest -R decision`) runs every body through
  `decision_parse_body`, `decision_parse_router`, `decision_router_items` and
  `laya_escape_text` and requires the same result for every case (`--dump NAME` prints the
  engine's side of one case). `test_reference.py` requires the file to be what the generator
  writes now, so a change on either side fails a test.
- `test_decision.py` sends every golden body to the server's render route on the tiny model and
  compares the states byte for byte and the 400 reason and param. For the escaping it renders the
  reference's `model_text` as a plain systemone state with the same question: the tokens must be
  those of the router item. The English tiny model repeats this on every 5th case.
- The escape sets of `laya-f16`, `laya-ml-f16`, `laya-en-f16` and `laya-td-f16` from
  `control_strings_from_gguf` equal the engine's (`test-decision-reference --control MODEL`;
  `test_reference.py` checks this when `TEST_DECISION_REFERENCE_BIN` is set and the models are
  there).

**`tools/decision/schema/executor-card-v1.schema.json`** is the card as JSON Schema (draft
2020-12), for tools that want a standard validator. Three rules cannot be written in JSON Schema:
`passed <= total`, the 4096-byte limit (`maxLength` counts code points) and integral floats
(`188.0` is an integer for JSON Schema; the engine refuses it). So the schema accepts a little
more than the engine. `test_reference.py` runs the schema and `reference.card_validate` on 1427
cards (hand-written cases and every card of the golden requests). They agree on every card
outside those three gaps (13 cards inside: 9 over 4096 bytes, 2 integral floats, 2 with passed >
total), and the reference refuses every card inside them. The same cards also go through the
`jsonschema` package (`Draft202012Validator`) when it is installed, as in CI. Counts carry
`exclusiveMaximum` 2^63, not `maximum` 2^63 - 1: that number is not a double, and a validator
that reads numbers as doubles would round it up and accept 2^63.

**Data format** (`tools/decision/router_eval.py`, shared by the two scripts): JSONL, one record
per task, `{"id", "task", "criterion", "candidates": [{"id", "card", "outcome": 0|1,
"logit"?}], "split"?: "fit"|"eval"}`. Every record must be a valid router request. Without
`split` a record goes to eval when `sha256("router:<id>")` falls below `--heldout-frac`
(default 0.3), so both scripts use the same split. Metrics:

- ECE: 10 equal-width bins of `p_success`, positive class. JevBench's top-label ECE bins
  `max(p, 1-p)` instead.
- Brier: mean `(p - y)^2`, the one-class form (half the 2-class convention).
- NLL, AUC, and accuracy at 0.5.

Intervals are 95 % percentile intervals of a cluster bootstrap over records: candidates of one
task share the task. Deltas are paired. An interval that contains zero means no convincing
difference. The bootstrap ECE is biased upward, so its interval can sit at or above the point
value.

**`scripts/fit-router-calibration.py`** fits `spec.router.calibration` (Platt, p = sigmoid(a z + b))
on this engine's raw logits:

```bash
# the server as it will run (GGUF, plan, kernels), uncalibrated for now
llama-server --decision -m router.gguf --decision-spec spec.json --decision-allow-uncalibrated --port 8090
python3 scripts/fit-router-calibration.py collect data.jsonl logits.jsonl --url http://127.0.0.1:8090 --spec spec.json
python3 scripts/fit-router-calibration.py fit logits.jsonl --spec spec.json --spec-out spec-router.json -o fit.json
python3 gguf-py/gguf/scripts/gguf_decision_spec.py set router.gguf spec-router.json -o router-cal.gguf
```

- `collect` posts each record to `/v1/router/score`. It stores the raw `logit` of every
  candidate and an `engine` block from `/props`: model, `spec_sha256`, plan, kernels, recipe.
  With `--spec` it checks that the server runs that exact spec: the `--decision-spec` file, or
  the output of `gguf_decision_spec.py get router.gguf > spec.json` (the GGUF text plus the
  newline `get` prints). It reports and skips requests the server refuses (for example
  `CARD_TOO_LONG`).
- `fit` refuses logits from more than one engine identity; a record without an `engine` block
  (logits from elsewhere) counts as its own. It refuses fewer than `--min-examples` (1000) fit
  examples or `--min-per-class` (50) of either outcome, and fewer than `--min-eval` (200) eval
  examples. It fits with Platt's prior-corrected targets (Newton, backtracking) and reports
  ECE, Brier, NLL and accuracy before and after, on the eval split, with the paired delta.
  "Before" is what ships now: `--before A,B`, else the `router.calibration` of `--spec`, else
  sigmoid(z). The in-sample fit-split numbers are marked as such.
- `fit` prints the `router.calibration` block. With `--spec` / `--spec-out` it writes the spec
  with that block and with `plan.kernels` set to the kernels the logits came from
  (`/props` `cpu`, `cpu+blas`, `cpu+repack`, `cpu+repack+blas` -> `default`, `blas`,
  `repack`, `repack+blas`), checks it with the `gguf_decision_spec.py` loader rules and prints
  the `set` command. `auto` is `blas` on macOS and `default` elsewhere, and the two give
  different z, so a calibration fitted on a Mac pins `blas`: such a GGUF fails to load on a
  build without a BLAS backend unless `--decision-kernels` overrides it (and then the
  calibration does not fit its logits). Collect on the platform the model will run on, or with
  the server started with `--decision-kernels default`. Without an `engine` block in the logits
  `fit` warns and leaves `plan.kernels` as it is. The router question's criteria are part of
  the prompt, so the block is valid only for the question in the spec the logits were
  collected with.

**`scripts/router-baselines.py`** scores simple baselines on the same data and split, with the
same metrics:

- `base-rate`: the fit-split positive rate.
- `pass-rate`: the pooled card pass rate (passed + 1) / (total + 2), with no fit.
- `pass-rate-platt`: `pass-rate` with a Platt fit.
- `logreg`: L2 logistic regression on card metrics (pooled and per-check pass rates, totals,
  measured and missing counts, kind).
- `engine-raw` and `engine-platt`: when the file has logits, all from one engine identity.

Deltas are taken against `--reference` (default `engine-platt`, else `pass-rate-platt`).

```bash
python3 scripts/router-baselines.py logits.jsonl -o baselines.json
python3 -m pytest -q tests/decision/test_reference.py        # offline tests of all of the above
```

## Tests

```bash
ctest --test-dir build -R decision        # py-json (golden + splitmix64 hashes), request, calib + spec cases, router, reference goldens
ctest --test-dir build -R laya-convert -LE python   # C++ HF -> GGUF converter vs the Python converter's SHA-256 (tiny fixtures)
# (-R decision also runs test-decision-checkpoint-cli: llama-laya-cli -m <tiny checkpoint dir>, cache, identity)
python3 -m pytest -q tests/decision/test_reference.py   # reference.py, card schema, calibration and baseline scripts
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest -q unit/test_decision.py
# tokenizer parity with the checkpoint's HF tokenizer (needs transformers and the real GGUF)
python tests/laya/verify_tokenizer.py build/bin/llama-laya-cli laya-multilingual-f16.gguf <hf-snapshot>/tokenizer
python tests/laya/verify_tokenizer.py build/bin/llama-laya-cli laya-en-f16.gguf <hf-snapshot>/tokenizer
# logits against the PyTorch reference (needs the laya package; see tests/laya/README.md)
$LAYA_PY tests/laya/verify_reference.py ref <hf-snapshot> items.jsonl ref.jsonl --english
python3 tests/laya/verify_reference.py cli build/bin/llama-laya-cli laya-en-f32.gguf ref.jsonl cli.jsonl
# the same items through llama-server --decision over HTTP (plan sequential, the server's threads and kernels)
python3 tests/laya/verify_reference.py server build/bin/llama-server laya-en-f16.gguf ref.jsonl server.jsonl
python3 tests/laya/verify_reference.py compare ref.jsonl cli.jsonl server.jsonl
```

The server tests are offline. The first `tiny_laya_decision_server()` call generates a random
tiny laya GGUF (`tests/decision/make_tiny_laya.py`, numpy + gguf-py) into `tools/server/tests/tmp`;
it is not a `ServerPreset`, so other test modules never build it. `tiny_laya_decision_server(english=True)`
uses a second one (`--english`: the `bytelevel-bpe` tokenizer and the temperature buckets of the
English checkpoints) and `tiny_laya_decision_server(q8=True)` a third (`--q8`: Q8_0 encoder
weights, which the repack buffers convert). The cache folder name carries a hash of the
generator, so a changed generator never reuses a stale file. `tests/decision/gen_golden.py`
writes the py-json golden files: `repr` of every power of two, 150 objects, and the SHA-256 of
30000 doubles and 1000 nested objects drawn from a splitmix64 stream that
`test-decision-json` draws the same way. CI runs both suites in the `decision-tests` job of
`.github/workflows/dev-build.yml` (Linux and Windows; warnings are errors on Linux), with
`ctest -R "decision|laya-convert" -LE python` (the converter tests included; the Python
cross-check needs torch and runs locally). They cover
response shapes, error codes 400/401/404/413/422/429/501/503, determinism, candidate
independence, start-up checks, the info routes and `-m <checkpoint dir>` (converted, then a cache
hit, the Python converter's bytes, the same answers as the Python GGUF, broken checkpoints, the
default cache locations; the tiny checkpoints of `tests/laya/convert`). With `--decision-debug`, two
environment variables make timing tests deterministic:
`LLAMA_DECISION_DEBUG_LOAD_DELAY_MS` (delay before the engine loads, for the 503 test) and
`LLAMA_DECISION_DEBUG_JOB_DELAY_MS` (delay before each job, for the 429 and shutdown tests).

## Benchmark

`llama-decision-bench` runs a suite of request bodies through the server code path in
process (request checks, engine, calibration, response text; no HTTP, no queue) and reports,
per request, the phases parse / tokenize / render / compute / post / total, p50 / p95 / p99,
CPU seconds (process user + system over the request), and the SHA-256 of the raw logits.
It also reports load time, RSS and memory footprint (macOS `phys_footprint`, Linux
`RssAnon`, Windows private bytes) after load and at the peak.

```bash
cmake --build build -j --target llama-decision-bench llama-server
python3 tests/decision/bench/gen_suite.py          # writes tests/decision/bench/suite.jsonl
./build/bin/llama-decision-bench -m laya-q8_0.gguf -f tests/decision/bench/suite.jsonl -t 4 --repeat 10 -o q8_0-t4.json

# every KPI on one machine: bench per model and thread count, then llama-server --decision
BIN_DIR=./build/bin MODELS="laya-f16.gguf laya-q8_0.gguf" bash scripts/bench-decision.sh
python3 scripts/bench-decision-report.py build/bench-decision
```

Flags: `-m`, `-f` (suite), `-t` (default: the performance cores), `--repeat` (measured runs
per request, default 10), `--warmup` (default 1), `--plan`, `--spec`, `--kernels`,
`--max-items`, `--idle-ms N` (after the runs, sleep N ms and run the suite once more: first
request after idle), `--filter`, `--poll N` (threadpool polling level), `--no-mmap`,
`--mlock`, `--no-warmup` (no warm-up pass inside the load; `load` includes it otherwise),
`--warmup-tokens N` (size of that pass, default 64), `--blas-threads N` (BLAS backend threads,
default 0 = min(`-t`, 8)), `-o` (JSON). `scripts/bench-decision.sh` takes `KERNELS` for both tools.
Requests run interleaved (every repeat runs every request once), so slow drift such as heat
hits all requests alike. `tokenize` is a separate pass after each request that tokenizes its
distinct state pieces once; it is a part of `render`. Router requests run with or without a
router calibration. A request whose logits hash differs between runs is marked `det: NO`;
the report also compares the suite hash across thread counts.

Suite lines: `{"id", "endpoint": "systemone" | "router", "group", "body"}`. The suite in
`tests/decision/bench` has the groups `systemone 1q` and `systemone 5q` (a ~170-token support
ticket), `router N=1/4/8` (executor cards of ~100 tokens, ~250-token tasks) and
`router N=1 t128` / `t512` (task length).

`scripts/bench-decision.sh` writes into `OUT_DIR` (default `build/bench-decision`) one
`__machine.txt` (CPU, cores, OS, power source from `pmset` / `upower` / `powercfg`, low
power mode, load average and top CPU users before every run), one bench JSON per model and
thread count, and one `__server.json` per server run: time from spawn to `/health` 200 (K5),
RSS and footprint of the server, warm latency, and the first request after `IDLE_S` seconds
idle over HTTP (K6). `scripts/bench-decision-report.py` prints the tables (K1, K4, K5, K6,
K7, K9). Close other applications first: a busy machine makes p95 meaningless.

Baseline before the Phase 4 work (2026-09-30, Apple M4 Max 12P + 4E, 48 GB, macOS 26.6.2, AC
power, low power mode off; one background process at 100% of one core; static Release build
with Metal, `laya-multilingual`, plan `sequential`, repeat 10, warmup 1). Latency in ms,
p50 / p95 over the requests of a group and their repeats:

| group | f16 -t 4 | f16 -t 8 | q8_0 -t 4 | q8_0 -t 8 |
|---|---|---|---|---|
| systemone 1q (~220 tokens) | 214 / 231 | 112 / 122 | 143 / 156 | 78 / 84 |
| systemone 5q | 1178 / 1289 | 624 / 687 | 900 / 1030 | 486 / 563 |
| router N=1 (~480 tokens) | 540 / 559 | 286 / 299 | 459 / 473 | 247 / 259 |
| router N=4 | 2025 / 2103 | 1073 / 1123 | 1609 / 1706 | 870 / 931 |
| router N=8 | 4111 / 4429 | 2182 / 2367 | 3205 / 3543 | 1744 / 1937 |
| RSS / footprint after load, MiB | 748 / 875 | 748 / 875 | 637 / 764 | 637 / 764 |
| server spawn -> ready, ms | 389 | 370 | 342 | 340 |
| first so-1q after 60 s idle (HTTP) vs warm p50 | 1.07x | 1.11x | 1.11x | 1.19x |

Compute is over 99.8% of every request (parse, render and post together stay under 0.5 ms;
tokenizing a request's state takes under 0.25 ms). CPU seconds per request are n_threads x wall time (worker threads spin), about 0.57 s
for q8_0 systemone 1q and 6.4 s for router N=4 at `-t 4`. Logits are bitwise identical across
all runs, across `-t 4` / `-t 8`, and between this build and a `GGML_BACKEND_DL` +
`GGML_CPU_ALL_VARIANTS` build without Metal (`apple_m4` variant, same latency); the DL build
uses 53 MiB less RSS and about 190 MiB less footprint after load.

### Phase 4 CPU levers (same machine, 2026-09-30)

Each lever was A/B-measured with `llama-decision-bench` against the baseline binary (suite
above, repeat 5, warmup 1) and kept only when it helped without breaking a gate. Numbers are
p50 in ms at `-t 8` unless noted.

| lever | effect | logits | kept |
|---|---|---|---|
| persistent threadpool (registry proc-addresses, created once, poll 0) | no measurable change: latency, CPU seconds and the first request after idle within noise (poll 50: 1% slower) | bitwise same | yes: neutral; the CPU backend creates no threads per graph (the BLAS backend still does, two rows down) |
| QoS `USER_INITIATED` for the thread that runs the graphs (macOS) | not measurable on an idle machine (meant for a busy one, K8) | bitwise same | yes |
| default threads = performance cores (12 on M4 Max) | `-t 12` vs `-t 8`: 1.35x faster with the ggml kernels | bitwise same (thread-invariant) | yes |
| ... also with BLAS kernels (after the BLAS thread fix below) | `-t 12` vs `-t 8`, F16 / Q8_0, final bench (idle machine, repeat 10): router N=1 112 / 106 vs 124 / 122 ms, N=4 429 / 398 vs 465 / 465, systemone 1q 55 / 49 vs 53 / 50, systemone 5q 298 / 269 vs 290 / 282; in the thread grid (repeat 5 x 2, load average about 9) router between 2% slower and 8% faster, systemone 5-14% slower. CPU seconds +20-30% at `-t 12` | bitwise same | yes: faster where it matters (router); a client next to a chat model can pass a smaller `-t` (atomic-chat-core passes at most 8) |
| BLAS backend threads = min(`-t`, 8) instead of `-t` | ggml-blas (no OpenMP) converts each F16 / Q8_0 weight to F32 on `n - 1` new `std::async` threads, about 90 times per pass. `laya-multilingual` F16 at `-t 12`: 4 threads vs 12 router N=4 446 vs 494 ms, systemone 5q 312 vs 330, CPU s per N=4 request 2.86 vs 3.33; 8 threads = 4 threads within noise (router N=4 410-444 vs 412-449 over 3 rounds); at `-t 8`, 4 = 8; 1 thread is 1.2-1.5x slower (the conversion then runs on one core). The English checkpoint (4x larger weights) wants more: F16 at `-t 12`, 8 or 12 threads vs 4 router N=4 927-943 vs 963-975 ms, systemone 5q 629-648 vs 670-696. Q8_0: 1 to 12 threads within noise (`build-p4/bench/m1/`) | bitwise same (thread-invariant) | yes: 8 is the best measured for both encoders |
| `token_embd` in a read-only mapping + free the GGUF metadata after load | `default` kernels: Q8_0 RSS after load 637 -> 262 MiB, footprint 764 -> 389 MiB; F16 748 -> 373 / 875 -> 500 (`auto` adds the BLAS backend, about 18 MiB) | bitwise same (911 items, F16 and Q8_0) | yes, default |
| warm-up pass at load (64 tokens) | no measurable effect on the first request: it was already within 2% of warm (server so-1q first / warm 71.8 / 70.3 ms before, 71.2 / 70.6 after, `-t 8`), and a 752-token first request (`router N=1 t512`, `-t 12`, `auto`) takes 202 / 205 ms (F16) and 208 / 199 ms (Q8_0) with / without it (p50 193 / 188); load +25-30 ms, footprint after load +19 MiB. A 512-token warm-up costs another +26 MiB and +90 ms of load for the same first request (201 / 197 ms): not taken | none | yes, default (`--no-warmup`): starts the threads and faults in the weights; the compute buffer is sized for 64 tokens and grows on the first longer request |
| `repack` (CPU extra buffer types), Q8_0 | 1.6-1.7x faster: router N=1 243 -> 143, systemone 1q 76 -> 47 | changed: 1395 of 2604 questions bitwise same, max \|dlogit\| 5.4 vs `default` | opt-in |
| `repack`, F16 | no F16 repack kernel: unchanged | bitwise same | (no-op) |
| `blas` (Accelerate in the scheduler), F16 | 2.3x faster: router N=1 283 -> 123, systemone 1q 111 -> 52; CPU seconds 3.6-4x lower | closer to the reference (below) | yes, default via `auto` |
| `blas`, Q8_0 | 2.0x faster: router N=1 243 -> 119, systemone 1q 76 -> 49; CPU seconds 3x lower | closer to the reference (below) | yes, default via `auto` |
| `malloc_zone_pressure_relief` after load | no change in footprint | - | no (reverted) |

Parity against the PyTorch reference (`laya-multilingual`, the 911 laya-eval items / 2604
questions, every one with identical input ids; one expected refusal: the null state of
`scalar_state_0890`), on two code paths:

- CLI: `llama-laya-cli -t 1` (the `build-parity/laya-eval` harness), which packs all
  questions of an item into one graph (block-diagonal mask).
- server: `llama-server --decision` over HTTP (`tests/laya/verify_reference.py server`), plan
  `sequential` (one graph per question), the default threads (the 12 performance cores; the
  `auto` rows ran with 8, and a 12-thread rerun of F16 `auto` is bitwise identical on all 911
  items) and 4 BLAS threads (before the default became 8; the logits do not depend on the
  BLAS threads either, K7). This is what a client of the server gets. Packing changes the
  rounding (see "Plans"), so the two paths differ.

| path, kernels | argmax agree | mean \|dlogit\| | max \|dlogit\| | mean TVD | max TVD | noul p>=0.5 agree |
|---|---|---|---|---|---|---|
| CLI, F16 `default` (Phase 1) | 2602 / 2604 | 0.0048 | 1.17 | 0.0012 | 0.21 | 715 / 715 |
| CLI, F16 `blas` | **2604 / 2604** | **0.0006** | **0.014** | 0.0001 | 0.001 | 715 / 715 |
| CLI, Q8_0 `default` (Phase 1) | 2529 / 2604 | 0.078 | 8.37 | 0.020 | 0.73 | 706 / 715 |
| CLI, Q8_0 `blas` | **2554 / 2604** | **0.047** | 8.09 | 0.011 | 0.51 | 708 / 715 |
| CLI, Q8_0 `repack` | 2538 / 2604 | 0.079 | 9.21 | 0.020 | 0.82 | 709 / 715 |
| server, F16 `default` (Linux / Windows default) | 2602 / 2604 | 0.0048 | 1.09 | 0.0012 | 0.21 | 715 / 715 |
| server, F16 `auto` = `blas` (macOS default) | **2604 / 2604** | **0.00055** | **0.0137** | 0.00014 | 0.0009 | 715 / 715 |
| server, Q8_0 `default` | 2530 / 2604 | 0.077 | 8.04 | 0.019 | 0.46 | 709 / 715 |
| server, Q8_0 `auto` = `blas` | **2554 / 2604** | **0.047** | 8.09 | 0.011 | 0.51 | 708 / 715 |

The server F16 `auto` run of the DL build (CPU variants, no Metal) is bitwise identical to the
static one on all 911 items. Outputs: `build-p4/parity/final/` (server) and
`build-p4/parity/cli-*` (CLI).

The ggml F16 matmul rounds the activations to F16 and the Q8_0 matmul quantizes them to
Q8_0; the BLAS path converts the weights to F32 and runs F32 sgemm, which is what the
reference does. That removes the F16 gap to the reference almost entirely (the two F16 flips
of Phase 1 are gone) and a third of the Q8_0 gap. `default` stays bitwise equal to Phase 1:
on the final tree the CLI raw logits of all 911 items (F16 and Q8_0) and the 7 CLI golden
cases (F16 and Q8_0 at `-t 1` / `-t 8`, 28 outputs, static and DL builds) are byte for byte the
Phase 1 ones.

KPIs after Phase 4 (static Release build with Metal and Accelerate, `--decision-kernels
auto` = `cpu+blas`, repeat 10, warmup 1; `default` = the ggml kernels with every other lever;
p50 / p95 ms). The `auto` columns are from the run after the BLAS thread fix
(`build-p4/bench/final2-auto-report.md`), made with 4 BLAS threads before the default became
min(`-t`, 8) (4 and 8 measured equal for this encoder, table above). A rerun with the final
binary and 8 BLAS threads (`final3-auto`) hit intermittent background load (load average up
to 16): same logits hashes, p50 within 6% at `-t 8`, but its `-t 12` and server numbers are
disturbed (p95 up to 2.9x), so it is not used. `default` and `repack` are from the run before
the fix (`final-default` / `final-repack`), whose code paths it does not touch. Memory
is in MiB (2^20 bytes): RSS counts the resident pages of the mapped `token_embd`; footprint is
macOS `phys_footprint`, which does not count clean file-backed pages.

| group | baseline f16 t8 | baseline q8_0 t8 | default q8_0 t12 | auto f16 t8 | auto f16 t12 | auto q8_0 t8 | auto q8_0 t12 | repack q8_0 t12 |
|---|---|---|---|---|---|---|---|---|
| systemone 1q | 112 / 122 | 78 / 84 | 58 / 62 | 53 / 58 | 55 / 59 | 50 / 54 | 49 / 55 | 39 / 51 |
| systemone 5q | 624 / 687 | 486 / 563 | 367 / 418 | 290 / 310 | 298 / 308 | 282 / 305 | 269 / 281 | 204 / 224 |
| router N=1 | 286 / 299 | 247 / 259 | 179 / 194 | 124 / 128 | 112 / 127 | 122 / 125 | 106 / 110 | 106 / 112 |
| router N=4 | 1073 / 1123 | 870 / 931 | 627 / 688 | 465 / 484 | 429 / 480 | 465 / 485 | 398 / 415 | 407 / 427 |
| router N=8 | 2182 / 2367 | 1744 / 1937 | 1264 / 1447 | 976 / 1018 | 895 / 918 | 917 / 1006 | 809 / 878 | 803 / 889 |
| router N=1 t128 / t512 | 196 / 509 | 139 / 388 | 102 / 283 | 86 / 229 | 80 / 193 | 82 / 228 | 75 / 188 | 67 / 208 |
| K4 bench after load, RSS / footprint | 748 / 875 | 637 / 764 | 264 / 390 | 393 / 519 | 393 / 519 | 281 / 408 | 282 / 408 | 264 / 390 |
| K4 bench after the first suite run, RSS / footprint | 809 / 775 | 698 / 664 | 335 / 292 | 457 / 414 | 459 / 416 | 345 / 302 | 347 / 304 | 332 / 289 |
| K4 bench peak, RSS / footprint | 810 / 890 | 699 / 788 | 347 / 437 | 459 / 550 | 462 / 566 | 347 / 438 | 350 / 438 | 333 / 436 |
| K4 server ready, RSS / footprint | 754 / 878 | 643 / 766 | 268 / 391 | 397 / 520 | 397 / 520 | 286 / 409 | 286 / 409 | - |
| K4 server after the requests and 2 x 60 s idle (one sample), RSS / footprint | 784 / 747 | 672 / 635 | 300 / 258 | 432 / 390 | 430 / 389 | 321 / 279 | 321 / 280 | - |
| K5 server spawn -> ready, ms | 370 | 340 | 290 | 324 | 322 | 295 | 306 | - |
| K5 server first so-1q / warm p50, ms | 104 / 103 | 72 / 70 | 50 / 49 | 53 / 50 | 52 / 51 | 49 / 46 | 47 / 47 | - |
| K6 so-1q after 60 s idle (HTTP) | 1.11x | 1.19x | 1.29x | 1.26x | 1.24x | 1.25x | 1.32x | - |
| K6 router N=4 after 60 s idle (HTTP) | 1.01x | 1.02x | 1.01x | 1.03x | 1.04x | 1.02x | 1.02x | - |
| K9 CPU s, systemone 1q | 0.90 | 0.61 | 0.68 | 0.22 | 0.30 | 0.20 | 0.27 | - |
| K9 CPU s, router N=4 | 8.58 | 6.98 | 7.66 | 2.29 | 2.77 | 2.25 | 2.53 | - |

- K1 / D3: with `auto` at `-t 8` or `-t 12`, router N=1 p95 is 110-128 ms, N=2 about 250 ms
  (two N=1 passes) and N=4 p95 415-485 ms, for F16 and Q8_0: inside the D3 goal (p95 <= 300
  ms for N=1-2, <= 1 s for N=4). At `-t 4` N=1 p95 is 173-176 ms and N=4 648-665 ms
  (`final-auto`), so N=2 (about 340 ms) misses 300 ms there.
- K4: the target is <= 400 MB = 381 MiB for Q8_0. Measured as footprint it holds in the DL
  build without Metal at every sample (below: 221 after load, 276 after the first run, 281
  peak). In the static build with Metal it does not hold after the load (`default` 390 MiB =
  409 MB, `auto` 408 MiB = 428 MB; the BLAS backend adds 18) nor at the peak (437-438 MiB),
  and holds after the first requests (289-304 MiB in the bench). In every build with Metal
  (the baseline too) the footprint falls by about 100 MiB during the first requests while RSS
  rises: the Metal device the backend registry initializes costs 187 MiB of footprint right
  after the load and about 26 MiB after the first requests (static 302-304 vs DL 276 MiB,
  Q8_0 `auto`). The server's last sample (279-280 MiB) came after two idle minutes, one sample
  each: not a steady state. RSS additionally counts the `token_embd` rows seen so far: +64 MiB
  after one suite run here, and up to the whole 209 MiB (Q8_0; 393 MiB F16) as more of the
  vocabulary appears (a 16 KB page holds about 20 Q8_0 rows). Those pages are clean and the
  OS can drop them under pressure; footprint never counts them.
- K6: the first request after an idle minute pays a fixed 7-16 ms over HTTP (1.24-1.32x on
  the 50 ms systemone 1q, 1.02-1.04x on router N=4), independent of the threadpool, the polling
  level and the warm-up: the cores and caches waking up.
- K7: logits are bitwise identical across warmup, repeats, the run after idle, `-t 4` / `-t
  8` / `-t 12`, BLAS threads 1 / 2 / 4 / 8 / 12, and the static and DL builds, for every
  kernel choice.
- K9: with BLAS the CPU time per request drops 3-4x (Accelerate runs the matmuls on the AMX
  units instead of 8-12 spinning cores).
- Not measured: K2 (HTTP overhead), K3 (GFLOPS), K8 (p95 next to a generating chat model),
  battery power, PyTorch / ONNX on the same machine. `llama-server` spawn -> ready was
  1.8 s once per freshly linked binary (macOS scans a new executable at its first launch);
  those first runs are left out above.

### D7: macOS acceleration (recommendation)

Measured options, Q8_0 / F16 at `-t 12` (DL = `GGML_BACKEND_DL=ON GGML_CPU_ALL_VARIANTS=ON
BUILD_SHARED_LIBS=ON GGML_METAL=OFF GGML_NATIVE=OFF`, `apple_m4` variant picked at runtime):

| build / kernels | router N=1 p50 / p95 | router N=4 p50 / p95 | footprint after load | footprint after the first suite run | footprint peak | logits |
|---|---|---|---|---|---|---|
| release-like static + Metal, `default` | 179 / 194 | 627 / 688 | 390 / 501 MiB | 292 / 398 MiB | 437 / 530 MiB | Phase 1 |
| static + Metal, `auto` (Accelerate) | 106 / 110 | 398 / 415 | 408 / 519 MiB | 304 / 416 MiB | 438 / 566 MiB | closer to reference |
| DL CPU variants, `auto` (Accelerate) | 107 / 113 | 406 / 417 | **221 / 333 MiB** | **276 / 389 MiB** | **281 / 393 MiB** | bitwise = static `auto` |
| static + Metal, `repack` (Q8_0) | 106 / 112 | 407 / 427 | 390 MiB | 289 MiB | 436 MiB | Q8_0-class |

Latency is Q8_0; footprint is macOS `phys_footprint` in MiB, Q8_0 / F16. The `auto` rows are
the `final2` runs (4 BLAS threads, see the KPI note); DL F16 is from the run before the BLAS
thread fix, which does not change memory.

Recommendation:

1. **Accelerate first (done, no new binary).** The BLAS backend is part of every macOS
   build (`GGML_BLAS` defaults to on for Apple), so `--decision-kernels auto` gives the
   2-2.3x speedup, 3-4x fewer CPU seconds and the better parity with the current release
   binaries. It is the default of the decision engine; `llama-laya-cli` and the golden files
   keep `default`.
2. **DL CPU variants for the decision process next**, when the app ships a dedicated
   decision binary: same latency and bitwise the same logits as the static build, 187 MiB
   less footprint right after the load and 157 MiB less at the peak (no Metal device; after
   the first requests the gap shrinks to about 26 MiB), the Q8_0 <= 400 MB (381 MiB) footprint target met after the
   load and at the peak, and an i8mm variant for M2+ without the SIGILL risk of a native build (risk 7). This
   needs the extra `libggml-cpu-*.so` / `libggml-blas.so` in the bundle and signed, so it is
   not wired into release CI here. Build note (opt-in, not in CI):

   ```bash
   # decision-only macOS build: CPU variants loaded at runtime, Accelerate, no Metal
   cmake -B build-decision-cpu -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND_DL=ON \
     -DGGML_CPU_ALL_VARIANTS=ON -DBUILD_SHARED_LIBS=ON -DGGML_METAL=OFF -DGGML_NATIVE=OFF
   cmake --build build-decision-cpu -j --target llama-server
   # bin/: llama-server, libggml*.dylib, libggml-cpu-apple_m1/m2_m3/m4.so, libggml-blas.so
   ```
3. **A separate static CPU build is not needed**: it would save the same Metal memory as
   the DL build but loses the per-CPU variant choice.

`repack` stays opt-in: on Q8_0 it is about as fast as Accelerate at `-t 12` (faster on short
systemone requests, 39 vs 49 ms) but keeps the Q8_0 activation rounding (Q8_0-class parity)
and spins all threads (about 12x wall time in CPU seconds). It is the lever for quantized
models where no Accelerate exists (Q4_0 / Q4_K on AVX2 and NEON); those platforms are not
measured here.

## English checkpoints (`laya`, `laya-typed-decisions`)

| checkpoint | encoder | ctx / head budget | tokenizer | calibration in the GGUF |
|---|---|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large, 28 layers, d 1024, 421M | 512 / 192 | `bytelevel-bpe` (NFC) | `laya.temperature` + 6 `temperature_by_options` buckets |
| `convaiinnovations/laya-typed-decisions` | same | 1024 / 256 | same | same buckets, other base temperatures |
| `convaiinnovations/laya-multilingual` (default, D1) | mmBERT-base, 22 layers, d 768, 322M | 1024 / 256 | `metaspace-bpe` | none (`calibrated: false`) |

```bash
python convert_hf_to_gguf.py <snapshot of convaiinnovations/laya> --outfile laya-en-f16.gguf --outtype f16
# or, same bytes, without Python (see "Converting without Python")
./build/bin/llama-laya-convert <snapshot of convaiinnovations/laya> -o laya-en-f16.gguf --outtype f16
./build/bin/llama-server --decision -m laya-en-f16.gguf --device none --port 8090
```

The converter takes the special ids from the tokenizer (`[CLS]` 50281, `[SEP]` 50282,
`[MASK]` 50284, which is also `laya.marker_token_id`). A GGUF loads only if the mask id equals
`laya.marker_token_id` and every special id is inside the vocabulary; a `bytelevel-bpe` GGUF must
name `[CLS]`, `[SEP]` and `[MASK]` (the fallback ids 2 / 1 / 4 are the mmBERT ones).

With 512 tokens, `laya` leaves about 316 tokens for a router state, and the default
`max_card_tokens` is 192 (3/8 of 512); the router probe finds the state split exact for this
vocabulary (`state_split: true`).

Parity (Apple M4 Max, 2026-09-30, final tree). Tokenizer: 0 mismatches in 13647 strings, 10763
of them English-heavy (`verify_tokenizer.py`), for both checkpoints, and 0 of 2000 router
head/tail splits (`laya-multilingual`: 0 in 13675). Logits: the 503 English items (1491
questions; the letters of state and questions all ASCII) of the 911-item laya-eval set, PyTorch
reference fp32 (laya 0.3.21) against two paths (`verify_reference.py`): CLI = `llama-laya-cli
-t 1`, all questions of an item packed into one graph; server = `llama-server --decision` over
HTTP, plan `sequential` (the server runs used `-t 4`; logits are thread-invariant). Every
question had identical input ids and marker positions; the one refused item (null state) is
refused by both. `calibrated |dP|` compares the answers (temperatures and buckets applied;
reference rounded to 4 digits) with the reference `system_one` answers on 311 questions.

| path, GGUF, kernels | argmax | max \|dlogit\| | mean \|dlogit\| | max TVD | calibrated max \|dP\| |
|---|---|---|---|---|---|
| CLI, `laya` F32, default | 1491/1491 | 4.2e-4 | 5.8e-6 | 2.0e-5 | 1e-4 |
| CLI, `laya` F16, default | 1489/1491 | 0.44 | 5.2e-3 | 1.7e-2 | 6e-3 |
| CLI, `laya` F16, blas | 1491/1491 | 1.3e-2 | 5.3e-4 | 1.1e-3 | 5e-4 |
| server, `laya` F16, default (the Linux / Windows default) | 1490/1491 | 0.24 | 5.2e-3 | 1.7e-2 | 5.8e-3 |
| server, `laya` F16, auto = blas (the macOS default) | 1491/1491 | 1.3e-2 | 5.3e-4 | 1.1e-3 | 4.6e-4 |
| CLI, `laya-typed-decisions` F32, default | 1491/1491 | 6.8e-5 | 1.3e-6 | 1.8e-5 | 1e-4 |
| CLI, `laya-typed-decisions` F16, default | 1491/1491 | 3.6e-2 | 1.3e-3 | 1.0e-2 | 6e-3 |
| CLI, `laya-typed-decisions` F16, blas | 1491/1491 | 3.9e-3 | 4.9e-4 | 1.1e-3 | 7e-4 |
| server, `laya-typed-decisions` F16, default | 1491/1491 | 3.8e-2 | 1.3e-3 | 1.1e-2 | 5.4e-3 |
| server, `laya-typed-decisions` F16, auto = blas | 1491/1491 | 4.6e-3 | 5.0e-4 | 1.2e-3 | 7.1e-4 |

Every CLI run of the final tree (F32, F16 `default`, F16 `blas`, both checkpoints) is bitwise
identical to the run made before the last changes to the Unicode tables and `laya.cpp`. The F16 `default` flips are near-ties (reference
top-2 gaps 0.009 and 0.005; the server path keeps one of them). The act-head logits of these
checkpoints are 169 to several thousand in magnitude; the relative difference is at most 8e-5
(F32), 8e-4 (F16 blas) and 6e-2 (F16 default), and the act probability (0 or 1 at these
magnitudes) never changes. Outputs: `build-p4/parity/final/` (`compare_en.json`,
`compare_td.json`).

Latency (`llama-decision-bench`, same suite and machine, F16, kernels `auto` = Accelerate BLAS,
p50 / p95 ms; `build-p4/bench/english-report.md`). Measured with BLAS threads = `-t`, before
the default became min(`-t`, 8); for this encoder an interleaved A/B at `-t 12` found 8 and 12
BLAS threads equal (router N=4 941 / 943 vs 940 / 927 ms, `build-p4/bench/m1/table-ab.md`), and
the logits do not depend on them. A rerun with the final binary under background load
(`build-p4/bench/english3`) was 5-10% slower at the p50, with identical logits hashes:

| group | multilingual t8 | multilingual t12 | `laya` t8 | `laya` t12 | `laya-typed-decisions` t12 |
|---|---|---|---|---|---|
| systemone 1q | 52 / 55 | 54 / 59 | 120 / 129 | 116 / 125 | 119 / 122 |
| systemone 5q | 280 / 301 | 292 / 321 | 635 / 688 | 619 / 656 | 605 / 654 |
| router N=1 | 123 / 127 | 111 / 115 | 245 / 250 | 221 / 229 | 216 / 220 |
| router N=4 | 460 / 479 | 426 / 436 | 998 / 1030 | 907 / 939 | 893 / 915 |
| router N=8 | 930 / 1001 | 862 / 922 | 1874 / 1991 | 1693 / 1835 | 1670 / 1789 |
| compute per 1k tokens | 230-255 | 228-246 | 543-571 | 507-553 | 493-547 |
| footprint after load / peak, MiB | 519 / 546 | 519 / 550 | 943 / 959 | 944 / 962 | 943 / 959 |
| server ready, ms | - | 342 | 288 | 306 | 306 |

- Per token the English encoder costs about 2.2x the multilingual one. The English BPE needs
  about 10% fewer tokens for the same English text (router N=1: 418 vs 467), so a request
  costs about 2x. Logits are identical at `-t 8` and `-t 12` for every model.
- D3 (p95 <= 300 ms for N=1-2, <= 1 s for N=4): the English checkpoints meet N=1 (229 ms)
  and N=4 (939 ms) at `-t 12`; N=2 (about 450 ms) does not.
- Memory: F16 needs about 950 MiB of footprint; the 400 MB target would need a quantized recipe, and its
  parity is not measured for these checkpoints.

## Converting without Python (`llama-laya-convert`)

```bash
./build/bin/llama-laya-convert <checkpoint dir> -o laya-f16.gguf [--outtype f32|f16|q8_0] [--model-name NAME] [--verify-against ref.gguf]
./build/bin/llama-laya-convert --compare a.gguf b.gguf
```

A C++ port of `convert_hf_to_gguf.py` for laya checkpoints (`conversion/laya.py` and the classes
it inherits, gguf-py `SpecialVocab` and `Metadata`); it needs no Python and links ggml (plus `laya` for
its Windows path helper). The Python converter is
the reference: for all three checkpoints and `f32` / `f16` / `q8_0`, with and without
`--model-name`, the output is byte-identical to `convert_hf_to_gguf.py <dir> --outfile X --outtype T
[--model-name NAME]` (18 of 18 files, same SHA-256 and size). An f16 file takes 0.2-1.8 s on an
M4 Max (idle to shared with other jobs), 6-19x less than the Python converter in the same runs.
`--verify-against` prints the first difference (KV, tensor info, tensor data, then raw bytes).
Exit codes: 0 done / identical, 1 error (a file cannot be read, the conversion failed), 2 the files
differ.

- `q8_0` is `convert_hf_to_gguf.py --outtype q8_0`: 1-D and `*_norm.weight` F32, rows that are not
  whole Q8_0 blocks F16, every other 2-D weight Q8_0, including `token_embd`, `type_emb`, `scorer.*`
  and `act_head.*`. That is not the Q8_0 recipe the accuracy figures of this document were measured
  on (`tests/laya/quantize.sh` keeps those five at F16), and its accuracy is not measured, so
  `-m DIR` does not offer it. For a served Q8_0 model, quantize the f16 GGUF with
  `tests/laya/quantize.sh`.
- `general.*` follows gguf-py: `--model-name`, the README front matter (a YAML subset), and the
  directory name. Like Python, an HF snapshot directory whose hash starts with a digit gives
  `general.finetune=<hash>`, and without `--model-name` the name is the title-cased hash.
- Inputs the port does not cover are refused with a message instead of giving a different file:
  a root `config.json`, `pytorch_model*.bin` (also a `model.safetensors.index.json` without any
  `model*.safetensors` file: Python then takes the `.bin` path), non-float dtypes, tensor names
  outside the laya table, rope scaling / experts / `quantization_config` / `id2label` in the
  encoder config, `modules.json`, a `tokenizer_class` other than `PreTrainedTokenizerFast` /
  `TokenizersBackend` (or, without one, a `tokenizer/config.json` with `model_type` or
  `tokenizer_class`: AutoTokenizer would pick a model-specific class), `auto_map`, added tokens
  whose AutoTokenizer round trip is not a known identity or U+2581 -> space, special tokens
  AutoTokenizer would add, list-form chat templates, YAML beyond the subset for a used key,
  non-finite weights with `q8_0`, absurd sizes Python cannot finish either (`num_hidden_layers` >
  65536, `vocab_size` > 2^24, `vocab_size` above the `token_embd` rows), JSON nested deeper than 127
  levels, and a `generation_config.json` with NaN / Infinity / lone surrogate escapes (Python's
  `json` reads them; a plain syntax error is ignored, as in Python). The full list is in
  `tools/laya/laya-convert.h`.
- Where Python loads a file through a strict library, the fields the port reads are held to the
  same rules: every `added_tokens` flag must be a JSON bool and a `TemplateProcessing` must have
  the `tokenizers` schema (as tokenizers 0.23 requires), and the README front matter must not
  contain characters PyYAML's reader rejects. Fields the port does not read are not validated,
  so Python can still reject a file this tool converts (example: a bad `padding_side` in
  `tokenizer_config.json`, which transformers checks).
- Behaviour pinned to the reference environment (transformers 5.17, tokenizers 0.23): the named
  special tokens of `tokenizer_config.json` and the values of a dict `extra_special_tokens` become
  CONTROL; `clean_up_tokenization_spaces` is ignored for BPE. F32 -> F16 rounds like numpy on
  AArch64 / x86-64 F16C (checked on all 2^32 inputs), NaN quieted with the top payload bits.
- Checkpoint files are untrusted: safetensors header length, offsets, dtype / shape / byte
  counts are checked (overflow-safe) before any read; JSON nesting is limited before parsing
  (copying a deeply nested value would overflow the stack); text processing is linear (a 4M-line
  CRLF README or 300k YAML keys convert in 0.1-0.2 s); paths are UTF-8 (UTF-16 APIs on Windows).
- Output: written to a new, uniquely named `<out>.<hex>.tmp` (created exclusively, so never through
  an existing file or symlink), read back with the gguf API, synced to disk (`F_FULLFSYNC` on
  macOS: 0.08-0.15 s for 660 MB) and renamed. Nothing is left on failure.

Tests: `ctest -R laya-convert` converts the tiny fixtures of `tests/laya/convert` (both tokenizer
families) against the SHA-256 of the Python output and runs the broken-input cases (CI runs it on
Linux and Windows); `test-laya-convert-py` (label `python`) converts fresh fixtures with both
converters (`LAYA_REF_PYTHON` names the interpreter; skipped without torch or without transformers
5.17 / tokenizers 0.23). To check the real checkpoints yourself:

```bash
python convert_hf_to_gguf.py <snapshot> --outfile py.gguf --outtype f16
./build/bin/llama-laya-convert <snapshot> -o cpp.gguf --outtype f16 --verify-against py.gguf   # IDENTICAL to py.gguf
```

## Loading a Hugging Face checkpoint directly

```bash
./build/bin/llama-server --decision -m ~/models/laya-multilingual --port 8090     # converts once, then loads
./build/bin/llama-server --decision -m ~/models/laya-multilingual --port 8090     # cache hit
./build/bin/llama-laya-cli -m ~/models/laya-multilingual -f tests/laya/demo_input.json
# options (server and CLI): --decision-convert-cache DIR   --decision-convert-type f16|f32
```

`-m` may name a laya checkpoint directory (the Hugging Face layout: `rl_agent_config.json`,
`encoder/config.json`, `tokenizer/`, `model*.safetensors`; an HF snapshot directory works as it
is). It is converted once with the C++ converter (`llama-laya-convert`, previous section) into a
GGUF cache, and the cached GGUF is then loaded exactly like `-m FILE`: one execution path, the
engine never reads the checkpoint. GGUF inputs work as before.

- **Cache**: `<cache>/<key>/<name>.gguf`. `<cache>` is `--decision-convert-cache`, else
  `$LLAMA_CACHE/laya/gguf-cache`, else the user cache: macOS
  `~/Library/Caches/llama.cpp/laya/gguf-cache`, Linux `$XDG_CACHE_HOME` or `~/.cache` +
  `/llama.cpp/laya/gguf-cache`, Windows `%LOCALAPPDATA%\llama.cpp\laya\gguf-cache` (the layout of
  common's `fs_get_cache_directory`, which `-mu` URL and docker downloads use; `-hf` uses the
  Hugging Face hub cache instead). Nothing is written into the checkpoint directory: a cache inside
  it is refused before anything is created. `<name>` is the directory name, so the default model
  name and spec `model_id` are the directory name (for an HF snapshot: the commit hash; set `-a`
  for another alias).
- **Key**: the first 32 hex digits of a SHA-256 over the cache format, the converter version
  (`LAYA_CONVERT_VERSION`, bumped with every change of the written bytes and with every input a
  newer converter refuses, so older cache entries are not reused), the outtype, the directory name
  (the converter derives `general.*` from it) and the files the converter can read: those directly
  in the checkpoint root, in `encoder/` and in `tokenizer/` (sorted relative paths; dot-entries
  such as `.git` or `.cache` skipped; at most 20000 entries): files up to 8 MiB by size + SHA-256 of
  the content, larger files by size + mtime in ns + symlink target. Any other subdirectory counts by
  its name only: the converter never reads below it (the `laya` repository keeps two more
  checkpoints and eval data there). Files over 8 MiB are keyed by metadata because hashing them at
  each start would cost more than converting: the in-tree SHA-256 takes 1.43 s for the 644 MB
  `laya-multilingual` weights and 1.87 s for the 843 MB English ones. That covers the weights and
  also the 34 MB `tokenizer.json` of `laya-multilingual`. Size + ns mtime change with every rewrite,
  and in the HF cache the symlink target is the blob name, which is the content hash. Small files,
  where an edit can keep the size and fall within the mtime granularity (a digit in a config), are
  keyed by content. What the key does not see: a same-size rewrite of a file over 8 MiB that also
  restores its mtime.
- **Writes are atomic and durable**: the conversion goes to a unique temporary name in the key
  directory, is synced to disk and renamed into place (the directory is synced too), so a start
  never sees a partial file and a power loss cannot leave a complete-looking file with zeroed data.
  Concurrent cold starts on the same checkpoint each convert and the last rename wins with the same
  bytes (8 simultaneous starts on an empty cache: 8 conversions, 1 file, identical outputs).
  Temporary files older than 10 minutes in the key directory (a killed conversion) are removed
  before the next conversion there. A checkpoint that changes while it is converted is refused (key
  computed again afterwards). A cache entry that is not a readable laya GGUF with all its tensor
  data is converted again; entries are never deleted, remove old key directories by hand. On POSIX,
  a key directory or cached file owned by another user, writable by others, or a symlink is refused
  (a shared `LLAMA_CACHE` could otherwise hand over another model).
- **Logs**: `decision: converted: <path> in 1.04 s (...)` or `decision: cache hit: <path> (...)`
  (`laya:` in the CLI). `/props.decision` has `source` (`gguf` or `checkpoint-dir`), `cache_path`
  and `checkpoint {dir, cache_dir, key, outtype, cache_hit, convert_ms, converter}`: `convert_ms`
  is 0 on a cache hit, `converter` is the integer `LAYA_CONVERT_VERSION`.
- **Errors**: a directory without `rl_agent_config.json` ("is a directory but not a laya
  checkpoint"), a checkpoint the converter refuses ("cannot convert checkpoint directory ...:
  <converter message>"), `--decision-convert-type q8_0`, and an unusable cache directory stop the
  load with exit code 1 and leave nothing in the cache.
- **Cost** (M4 Max shared with other jobs, `laya-multilingual` f16, `llama-laya-cli --tokenize` of
  one string, 3 rounds): first start 1.16-1.34 s including the 0.86-1.04 s conversion, cache hit
  0.32-0.36 s, the same GGUF with `-m FILE` 0.29-0.30 s (the key and the cache-entry check cost
  about 40-60 ms). Peak RSS: 513-517 MiB for a converting start, 372-373 MiB for a cache hit or
  `-m FILE`.

**Identity with the Python converter.** The cached file is the output of `llama-laya-convert`:
for the three published checkpoints its bytes equal `convert_hf_to_gguf.py` without
`--model-name` (what `-m DIR` converts) in f32, f16 and q8_0, and with `--model-name` too (18 of 18
files). On 67 inputs per cell (the 7 golden cases + 60 sampled from the parity sets), `llama-laya-cli
-m DIR` (the C++ cache, this build's loader) and the previous CLI and loader on the Python GGUF
print the same bytes apart from the `"model"` line: 6 of 6 cells (3 checkpoints x f16 / f32, 402 of
402). GGUF inputs are unchanged: on the 7 golden cases the CLI before and after this change prints
the same bytes (28 of 28: `laya-multilingual` f16 / q8_0 at `-t 1` / `-t 8`). 989 malformed
checkpoints (fuzzing under ASan + UBSan) gave no crash: 830 clean refusals and 159 accepted; the
Python converter wrote the same bytes for 158 of those and rejected the other (a bad
`padding_side`, a field the port does not read; see the previous section). `test_decision.py` and `test-decision-checkpoint-cli` check the same on the tiny
fixtures: the first start converts, the second hits the cache, the cached file has the SHA-256 of
the Python output (`tests/laya/convert/golden.sha256`; with `LAYA_REF_PYTHON` set the Python
converter also runs), and the answers with debug logits are identical to loading the Python GGUF.

Known gaps: the conversion runs in the server process, and the allocator keeps what it freed:
a converting start peaks at 513-517 MiB against 372-373 MiB for a cache hit or `-m FILE`
(`laya-multilingual` f16, `llama-laya-cli`; mostly the 34 MB `tokenizer.json` parsed into a JSON
tree), and a server keeps part of that until the next start. Convert ahead (`llama-laya-cli -m DIR --tokenize` of one string, or one start) where that
matters. There is no cache eviction. On Windows the converter and the cache open their paths as
UTF-8 through UTF-16 with the helpers of the Windows fixes (`laya_utf8_to_wide`, `ggml_fopen`;
`llama-laya-convert` takes its arguments through `decision_utf8_args`, like `llama-laya-cli`; see
"Status and known gaps"), but `-m DIR` and `llama-laya-convert` have not been built or run there yet.

## Status and known gaps

- Router calibration: the tools exist (`fit-router-calibration.py`, `router-baselines.py`),
  but no Platt block has been fitted on real outcome data yet. That needs the model team's
  labelled router data and a router checkpoint. The Phase 2 acceptance numbers (decision
  agreement at p = 0.95 against PyTorch, mean |dp|, ECE shift on >= 2000 held-out) are not
  measured.

- `semif-letters` (Arbiter-4B, JevK5) is not implemented; such a spec fails at load.
- English checkpoints: only F32 and F16 are measured; no quantized recipe, no router calibration
  and no x86 run yet.
- laya: the reference's per-language temperatures and `answer_confidence` are not
  implemented in the server; the act head (`action`) is not exposed. `llama-laya-cli` prints
  both, in the reference answer shape.
- The ggml threadpool workers keep QoS `DEFAULT` on macOS (ggml sets no QoS on the threads it
  creates; only the calling worker thread is raised). The first request after a long idle
  pays a fixed ~20 ms (cores and caches waking up), independent of the threadpool.
- ggml-blas (without OpenMP) still starts `n_threads_blas - 1` threads for every weight
  conversion and converts the same weights on every pass; caching the F32 weights or reusing
  threads would need a change inside ggml (not done: ggml is left untouched).
- Windows: MSVC 2022 and 2026 (without `/WX`: upstream code has warnings) built the tree
  *before* the Windows fixes on a Windows x64 runner and passed the decision gates there
  (ctest, pytest, `verify_precision.py` with `PYTHONUTF8=1`, CLI t1 = t8). The fixes: model,
  spec and input paths are opened as UTF-8 through UTF-16 (`CreateFileW`,
  `std::filesystem::path`, `ggml_fopen`), `llama-laya-cli` takes its arguments from
  `GetCommandLineW` and writes LF, and the Python helpers read child output as UTF-8. They
  compile with MinGW-w64 GCC 15.2 (`-fsyntax-only -Wall -Wextra -Werror`, `_WIN32_WINNT`
  0x0601 and 0x0A00) and pass on macOS, but have not been run on Windows yet.
