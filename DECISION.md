# Decision mode (`llama-server --decision`)

`llama-server --decision -m model.gguf` serves a *decision model*: a small model that
returns calibrated probabilities over a fixed set of options in one forward pass, with
no generation. It is a separate process role of this fork. It has no chat routes, no
slots, no KV cache and no web UI.

Engines (layouts):

| layout | models | runtime | status |
|---|---|---|---|
| `laya` | Laya family (`laya-multilingual`, mmBERT encoder + marker head) | `tools/laya`, own ggml graph, CPU only | done |
| `semif-letters` | Arbiter-4B, JevK5 (Qwen3.5 + letter readout) | libllama | not supported yet: load fails with a clear error |

Code map:

| path | role |
|---|---|
| `tools/server/server-decision.{h,cpp}` | HTTP routes, FIFO worker, error envelope, info routes |
| `tools/decision/` | static library: py-json, `decision.spec`, request checks, calibration, executor cards, engines, laya input building (`laya-decide`) |
| `tools/laya/` | laya runtime (port of upstream PR #29363; links only ggml) and `llama-laya-cli` |
| `gguf-py/gguf/scripts/gguf_decision_spec.py` | get / set / verify `decision.spec` in a GGUF |
| `tests/test-decision-*.cpp`, `tests/decision/` | C++ unit tests, golden files, tiny model generator |
| `tools/server/tests/unit/test_decision.py` | server contract tests (offline) |

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
loads), then the spec and the engine load, then the server is ready. A load error stops
the process with exit code 1.

## Flags

Environment variables use the `LLAMA_ARG_DECISION_*` names.

| flag | default | meaning |
|---|---|---|
| `--decision` | off | decision mode. Requires `-m FILE`; `-hf`, `-dr` and `-mu` are rejected (without `-m` the server would start in router mode). Explicit only: a stamped GGUF never switches mode on its own. |
| `--decision-spec FILE` | from the GGUF | JSON that replaces the embedded `decision.spec` as a whole. A bare Arbiter `calibration.json` (`{"temperature": x}` or `{"temperature": {...}}`) is accepted too: it is a required calibration, keeps the layout clamp (laya [0.5, 5]) and replaces the `laya.temperature` calibration; its `calibration.version` is its own `version` field (else `cal-<hash>`), never `gguf:laya.temperature`. |
| `--decision-plan NAME` | from the spec | compute plan, laya: `sequential`, `packed` (systemone only; the router always runs `sequential`) |
| `--decision-queue N` | 4 | requests that may wait while one runs; more get 429 |
| `--decision-max-items N` | 16 | questions per `/v1/systemone` request, candidates per `/v1/router/score` request (router is also capped at 16) |
| `--decision-allow-uncalibrated` | off | serve `/v1/router/score` without a router calibration |
| `--decision-debug` | off | raw logits and token ids in answers, `POST /v1/decision/render`, request logging, test delays (below) |

Reused: `-m`, `-a/--alias` (default: the model file name without its directories and a trailing `.gguf`, the same name as the default spec `model_id`), `--host`, `--port`, `--api-key` / `LLAMA_API_KEY`, `-t`
(engine threads, fixed at load), `--metrics`, `--timeout`, SSL flags. `--device` is
ignored by the laya engine (CPU only).

Ignored with a warning: `--parallel`, `-ctk/-ctv`, `--ctx-checkpoints`, `--spec-*` / `-md`,
`--embedding` / `--pooling`, `--mmproj`, `--lora`, `--sleep-idle-seconds`,
`--mcp-servers-*`, `--tools`, `--ui-mcp-proxy`, `--path`. The UI is always off.

A laya GGUF started without `--decision` fails fast with a hint. `--decision` inside a
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
HTML tags) stay tokens. The strings cannot overlap and a space cannot form a new one, so
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
  `plan {name, router, plans, n_threads, state_split, recipe, fa, n_batch, n_ubatch}` (`router`: the plan
  of `/v1/router/score`; `state_split`: router state pieces are tokenized separately, see
  Scheduling), `device`, `kv_type`,
  `queue_capacity`, `input_contract`, `special_tokens`, `debug`.
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
and costs O(n log n) in the text length, so long words without spaces (CJK, base64, URLs)
stay cheap: a 1 MB input takes well under a second. Each request tokenizes its state once:
all questions of a systemone request share the state tokens, and router candidates share
the tokens of the criterion and the task. For that the router state splits after the card
and after `task:`, and each piece is tokenized on its own. That gives the tokens of the
whole text only where both points are token boundaries of the vocabulary (for
`laya-multilingual`: `\n\n` is an added token and a space starts a Metaspace word), so at
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
numerics (plan, threads, recipe) are shown in `/props.decision.plan`.

## Model metadata: `decision.spec`

`decision.spec` is a GGUF string holding UTF-8 JSON. It is the source of truth. The mirror
keys `decision.spec_version` (u32), `decision.layout`, `decision.model_id` and
`decision.model_version` are for readers that skip the JSON; the server checks that they
agree with the spec. `laya.*` keys hold architecture hyperparameters only.

Where the spec comes from, in order:

1. `--decision-spec FILE` replaces it as a whole (`spec_source: file`).
2. The GGUF `decision.spec` (`spec_source: gguf`).
3. Defaults for an unstamped laya GGUF (`spec_source: default`): calibration from
   `laya.temperature` unless all values are 1.0 (then `calibrated: false`, which is the
   case for `laya-multilingual`), no router calibration. A non-laya GGUF without a spec
   does not load.

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
  GGUF, recipe and plan.

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

## Tests

```bash
ctest --test-dir build -R decision        # py-json (golden + splitmix64 hashes), request, calib + spec cases, router
cd tools/server/tests
LLAMA_SERVER_BIN_PATH=../../../build/bin/llama-server python -m pytest -q unit/test_decision.py
# tokenizer parity with the checkpoint's HF tokenizer (needs transformers and the real GGUF)
python tests/laya/verify_tokenizer.py build/bin/llama-laya-cli laya-multilingual-f16.gguf <hf-snapshot>/tokenizer
```

The server tests are offline. The first `tiny_laya_decision_server()` call generates a random
tiny laya GGUF (`tests/decision/make_tiny_laya.py`, numpy + gguf-py) into `tools/server/tests/tmp`;
it is not a `ServerPreset`, so other test modules never build it. `tests/decision/gen_golden.py`
writes the py-json golden files: `repr` of every power of two, 150 objects, and the SHA-256 of
30000 doubles and 1000 nested objects drawn from a splitmix64 stream that
`test-decision-json` draws the same way. CI runs both suites in the `decision-tests` job of
`.github/workflows/dev-build.yml` (Linux and Windows; warnings are errors on Linux). They cover
response shapes, error codes 400/401/404/413/422/429/501/503, determinism, candidate
independence, start-up checks and the info routes. With `--decision-debug`, two
environment variables make timing tests deterministic:
`LLAMA_DECISION_DEBUG_LOAD_DELAY_MS` (delay before the engine loads, for the 503 test) and
`LLAMA_DECISION_DEBUG_JOB_DELAY_MS` (delay before each job, for the 429 and shutdown tests).

## Status and known gaps

- `semif-letters` (Arbiter-4B, JevK5) is not implemented; such a spec fails at load.
- laya: the reference's per-language temperatures and `answer_confidence` are not
  implemented in the server; the act head (`action`) is not exposed. `llama-laya-cli` prints
  both, in the reference answer shape.
- The laya engine loads weights with `ifstream` into RAM, with no mmap and no persistent
  threadpool (Phase 4).
