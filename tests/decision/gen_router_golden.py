#!/usr/bin/env python3
"""Write golden/router_cases.jsonl: router requests and what tools/decision/reference.py makes of them.

    python3 tests/decision/gen_router_golden.py            # rewrite the golden file (stdlib only, deterministic)
    python3 tests/decision/gen_router_golden.py --check    # exit 1 if the file differs from a fresh run

test-decision-reference (ctest -R decision) reads the file and checks that the engine's request
parser, card-v1 renderer, router state and special-token handling give the same bytes, and the
same 400 reason and param on bad requests. So the file ties the Python reference (which the
training pipeline imports) to the C++ engine.

Line 1 is a header: {"format", "control_sets": {"ml": {"mask", "control"}, "en": {...}}}. The sets
are the escape sets of laya-multilingual and of the English checkpoints (DECISION.md; the
test test_reference.py compares them with the real GGUFs when those are present). Every other
line is one case:
    {"name", "control": "ml"|"en", "special_tokens", "router": null | spec router block,
     "body_text": exact request body, "expect": {"ok": false, "reason", "param"} |
     {"ok": true, "n_items", "digest"}}
digest is a SHA-256 over the rendering (see digest()): the file stays small, and on a mismatch
test-decision-reference prints its own rendering of the case next to `reference.py router`.
The random part draws cards with unicode, C0/C1 controls, control-token text, long fields,
missing checks and unknown fields; the fixed part covers each rule once.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "decision"))
import reference as ref  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden" / "router_cases.jsonl"
N_RANDOM = 420

CONTROL_SETS = {
    "ml": {"mask": "<mask>", "control": sorted(["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<2mass>", "[@BOS@]", "<start_of_turn>",
                                                "<end_of_turn>"] + ["<unused%d>" % i for i in range(100)])},
    "en": {"mask": "[MASK]", "control": sorted(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<|padding|>", "<|endoftext|>"] +
                                               ["[unused%d]" % i for i in range(83)])},
}

ROUTERS = [
    None,
    {"question": {"type": "noul", "instructions": "Will it work?"}},
    {"question": {"type": "noul", "instructions": "Does the executor pass?", "criteria": {"TRUE": "it passes", "false": ""}}},
    {"question": {"type": "noul", "instructions": "Pass <mask> fail?", "criteria": {"true": 1.5, "false": None},
                  "labels": {"false": " fail ", "true": "pass"}}},
]

PIECES = [
    "a", "Z", "7", " ", "  ", "-", ":", ";", "/", "(Q4_K_M)", "field extraction", "exact match", "atomic-evals/extract",
    "\t", "\n", "\r\n", "\x00", "\x01", "\x1f", "\x7f", "\x0b", "\x0c",
    "\u0085", "\u00a0", "\u2028", "\u2029", "\u3000", "\u200b", "\ufeff",
    "\u00e9", "e\u0301", "\u4e2d\u6587", "\u0438\u0442\u043e\u0433", "\u05e9\u05dc\u05d5\u05dd", "\U0001F600", "\U0001F469\u200d\U0001F4bb",
    "\u2581", "\u2581\u2581", "<", ">", "[", "]", "<eos>", "<mask>", "<unused7>", "<unused99>", "<unused100>", "<unused>", "<eo",
    "<<eos>>", "[MASK]", "[unused3]", "[unused82]", "[unused83]", "<|endoftext|>", "<|padding|>", "[@BOS@]", "<start_of_turn>",
    "<2mass>", "<pad><pad>", "<unk", "<unk>", "<bos>", "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[unused0]", "   ",
    "\"", "\\", "{\"k\": 1}", "1.0", "188/200", "&amp;", "<b>bold</b>",
]


def rand_text(rng, lo=0, hi=12):
    return "".join(rng.choice(PIECES) for _ in range(rng.randint(lo, hi)))


def rand_nonblank(rng):
    while True:
        s = rand_text(rng, 1, 10)
        if ref.canon_line(s):
            return s


def long_text(rng, n_bytes):
    # exactly n_bytes of UTF-8, mixing 1- to 4-byte characters
    out, size = [], 0
    alphabet = ["x", " ", "\u00e9", "\u4e2d", "\U0001F600", "\t", "<eos>"]
    while size < n_bytes:
        ch = rng.choice(alphabet)
        b = len(ch.encode("utf-8"))
        if size + b > n_bytes:
            ch, b = "y", 1
        out.append(ch)
        size += b
    return "".join(out)


def rand_int(rng, lo, hi):
    return rng.choice([lo, hi, rng.randint(lo, hi)])


def rand_check(rng):
    c = {"skill": rand_nonblank(rng)}
    if rng.random() < 0.6:
        total = rng.choice([1, 2, 10, 200, 1000, 2**31, 2**53 + 1, ref.INT64_MAX])
        c.update(status="measured", passed=rand_int(rng, 0, total), total=total)
        if rng.random() < 0.7:
            c["criterion"] = rand_text(rng)
    else:
        c["status"] = "missing"
    if rng.random() < 0.7:
        c["source"] = rand_text(rng, 0, 4)
    if rng.random() < 0.6:
        c["version"] = rng.choice(["2026-09", "v2", "", " ", rand_text(rng, 0, 3)])
    if rng.random() < 0.1:
        c["note"] = rand_text(rng)
    items = list(c.items())
    rng.shuffle(items)
    return dict(items)


def rand_card(rng):
    card = {"schema": ref.CARD_SCHEMA, "name": rand_nonblank(rng), "kind": rng.choice(["local", "cloud", "api", rand_nonblank(rng)])}
    if rng.random() < 0.6:
        card["description"] = rand_text(rng, 0, 10)
    r = rng.random()
    if r < 0.02:
        card["name"] = long_text(rng, rng.choice([4096, 4000]))
    elif r < 0.03:
        card["description"] = long_text(rng, 4096)
    n = (32 if rng.random() < 0.03 else rng.choice([0, 0, 1, 2, 3, 5, 8])) if rng.random() < 0.9 else None
    if n is not None:
        card["checks"] = [rand_check(rng) for _ in range(n)]
    if rng.random() < 0.1:
        card["owner"] = {"team": rand_text(rng)}
    items = list(card.items())
    rng.shuffle(items)
    return dict(items)


def rand_id(rng, i):
    return rng.choice(["local/qwen3.5-4b-q4_k_m", "cloud@provider:model+v2", "a", "x" * 125, "A.b_c-d"]) + "-%d" % i


# faults: (name, function(body, rng) -> body or body text)
def _cand(body, rng):
    return rng.choice(body["candidates"])


def _check(body, rng):
    for c in body["candidates"]:
        if c["card"].get("checks"):
            return rng.choice(c["card"]["checks"])
    card = body["candidates"][0]["card"]
    card["checks"] = [rand_check(rng)]
    return card["checks"][0]


def f_schema(b, r): _cand(b, r)["card"]["schema"] = r.choice(["atomic.executor-card/2", 1, None, ""])
def f_no_schema(b, r): _cand(b, r)["card"].pop("schema")
def f_name_blank(b, r): _cand(b, r)["card"]["name"] = r.choice(["", " \t\n", "\x00\x7f", 5, None])
def f_no_kind(b, r): _cand(b, r)["card"].pop("kind")
def f_desc_type(b, r): _cand(b, r)["card"]["description"] = r.choice([1, None, ["x"], {"a": 1}])
def f_checks_type(b, r): _cand(b, r)["card"]["checks"] = r.choice([{}, "none", 3, None])
def f_many_checks(b, r): _cand(b, r)["card"]["checks"] = [rand_check(r) for _ in range(33)]
def f_check_type(b, r): _cand(b, r)["card"]["checks"] = [rand_check(r), r.choice(["x", 1, None, []])]
def f_skill(b, r): _check(b, r)["skill"] = r.choice(["", "  ", 1])
def f_status(b, r): _check(b, r)["status"] = r.choice(["Measured", "", None, 1, "unknown"])
def f_no_status(b, r): _check(b, r).pop("status")
def f_card_long(b, r): _cand(b, r)["card"][r.choice(["name", "kind", "description"])] = long_text(r, 4097)
def f_check_long(b, r): _check(b, r)[r.choice(["skill", "criterion", "source", "version"])] = long_text(r, 4097)
def f_multibyte_long(b, r): _cand(b, r)["card"]["name"] = "\u4e2d" * 1366  # 4098 bytes, 1366 characters
def f_crit_type(b, r): _check(b, r)["criterion"] = r.choice([1, None, True])


def f_measured(b, r):
    c = _check(b, r)
    c["status"] = "measured"
    c["passed"], c["total"] = r.choice([
        (201, 200), (1, 0), (0, 0), (-1, 5), (5, -1), (188.0, 200), (188, 200.0), (True, 2), (1, None), ("1", "2"),
        (ref.INT64_MAX + 1, ref.INT64_MAX + 1), (0, ref.UINT64_MAX)])


def f_measured_missing_num(b, r):
    c = _check(b, r)
    c["status"] = "measured"
    c.pop(r.choice(["passed", "total"]), None)


def f_missing_num(b, r):
    c = _check(b, r)
    c["status"] = "missing"
    c[r.choice(["passed", "total"])] = r.choice([0, None, 3])


def f_cand_type(b, r): b["candidates"][r.randrange(len(b["candidates"]))] = r.choice(["x", 1, None, []])
def f_bad_id(b, r): _cand(b, r)["id"] = r.choice(["", "x" * 129, "a b", "a\u00e9", "a#b", 7, None, "\u0661"])
def f_no_id(b, r): _cand(b, r).pop("id")
def f_dup_id(b, r): b["candidates"].append({"id": b["candidates"][0]["id"], "card": rand_card(r)})
def f_no_card(b, r): _cand(b, r).pop("card")
def f_card_type(b, r): _cand(b, r)["card"] = r.choice(["card", [], 1, None])
def f_cands(b, r): b["candidates"] = r.choice([[], {}, "x", None])
def f_many_cands(b, r): b["candidates"] = [{"id": "c%d" % i, "card": rand_card(r)} for i in range(17)]
def f_task(b, r): b["task"] = r.choice(["", " \u3000\u2028", "\u0085", 5, None])
def f_criterion(b, r): b["criterion"] = r.choice(["", "\t\n", ["x"], None])
def f_no_task(b, r): b.pop("task")
def f_truncation(b, r): b["truncation"] = r.choice(["Allow", "none", 1, True, ""])


FAULTS = [f_schema, f_no_schema, f_name_blank, f_no_kind, f_desc_type, f_checks_type, f_many_checks, f_check_type, f_skill,
          f_status, f_no_status, f_card_long, f_check_long, f_multibyte_long, f_crit_type, f_measured, f_measured_missing_num,
          f_missing_num, f_cand_type, f_bad_id, f_no_id, f_dup_id, f_no_card, f_card_type, f_cands, f_many_cands, f_task,
          f_criterion, f_no_task, f_truncation]


def rand_body(rng):
    n = 16 if rng.random() < 0.03 else rng.choice([1, 1, 2, 3, 4])
    body = {"model": rng.choice(["opt", "", "router"]), "task": rand_nonblank(rng) + rand_text(rng, 0, 15),
            "criterion": rand_nonblank(rng),
            "candidates": [{"id": rand_id(rng, i), "card": rand_card(rng)} for i in range(n)]}
    if rng.random() < 0.3:
        body["truncation"] = rng.choice(["allow", "error", None])
    if rng.random() < 0.1:
        body["candidates"][0]["weight"] = 2
    if rng.random() < 0.03:
        body["task"] = long_text(rng, 6000)
    if rng.random() < 0.05:
        body.pop("model")
    return body


# fixed body texts: parser rules and cases that only a raw text can express
FIXED_TEXTS = [
    ("array-body", "[1, 2]"),
    ("malformed", '{"task": "t", '),
    ("nan", '{"task": "t", "criterion": "c", "candidates": [], "x": NaN}'),
    ("infinity", '{"x": -Infinity}'),
    ("big-int", '{"x": 18446744073709551616, "task": "t"}'),
    ("float-overflow", '{"x": 1e400}'),
    ("lone-surrogate", '{"task": "\\ud800"}'),
    ("dup-task", '{"task": "first", "criterion": "c", "task": "second", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "local"}}]}'),
    ("escaped-unicode", '{"task": "\\u4e2d\\ud83d\\ude00\\u0000x", "criterion": "\\t\\u2028c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "\\u0000A\\u001fB\\u007f", "kind": "\\u00a0local\\u0085"}}]}'),
    ("uint64-total", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "checks": [{"skill": "s", "status": "measured", "passed": 1, "total": 9223372036854775808}]}}]}'),
    ("int64-max", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "checks": [{"skill": "s", "status": "measured", "passed": 9223372036854775807, "total": 9223372036854775807}]}}]}'),
    ("float-exp-passed", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "checks": [{"skill": "s", "status": "measured", "passed": 1e2, "total": 200}]}}]}'),
    ("neg-zero", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "checks": [{"skill": "s", "status": "measured", "passed": -0, "total": 1}]}}]}'),
    ("dup-card-key", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"name": "B", "schema": "atomic.executor-card/1", "name": "A", "kind": "k", "zz": 1, "name": "C"}}]}'),
    ("deep", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "x": ' + "[" * 124 + "]" * 124 + "}}]}"),
    ("too-deep", '{"x": ' + "[" * 128 + "]" * 128 + "}"),
    ("control-in-string", '{"task": "a\tb"}'),
    ("bom", '\ufeff{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k"}}]}'),
    ("neg-nan", '{"x": -NaN}'),
    ("trailing", '{"task": "t"} x'),
    ("underflow-ws", ' \n\t{"task": "t", "criterion": "c", "u": 1e-400, "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "\\u0000k": -0.0}}]}\r\n '),
    ("neg-zero-float", '{"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k", "checks": [{"skill": "s", "status": "measured", "passed": -0.0, "total": 1}]}}]}'),
    ("bad-escape", '{"task": "\\x41"}'),
    ("dup-candidates", '{"candidates": [], "task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "k"}}]}'),
    # the first failure in document order decides the reason, as in the engine's SAX parser
    ("too-deep-then-bad-number", '{"x": ' + "[" * 128 + "]" * 128 + ', "y": 1e400}'),
    ("bad-number-then-too-deep", '{"y": 1e400, "x": ' + "[" * 128 + "]" * 128 + "}"),
    ("lone-surrogate-then-nan", '{"t": "\\ud800", "y": NaN}'),
    ("nan-then-lone-surrogate", '{"y": NaN, "t": "\\ud800"}'),
    ("split-surrogate-then-big-int", '{"t": "\\ud83dx\\ude00", "y": 18446744073709551616}'),
    ("surrogate-pair-then-nan", '{"t": "\\ud83d\\ude00", "y": -Infinity}'),
]


def netstring(s):
    b = s.encode("utf-8")
    return str(len(b)).encode() + b":" + b


def digest(r):
    """SHA-256 over length-prefixed fields: warnings, truncation_error, the question's type and
    instructions, options, then per item id, state, state_splits, model_text.
    test-decision-reference hashes its output the same way."""
    fields = [str(len(r["warnings"]))] + r["warnings"] + ["1" if r["truncation_error"] else "0"]
    fields += [r["question"]["type"], r["question"]["instructions"]] + r["options"]
    for it in r["items"]:
        fields += [it["id"], it["state"], "%d,%d" % tuple(it["state_splits"]), it["model_text"]]
    return hashlib.sha256(b"".join(netstring(f) for f in fields)).hexdigest()


def expect(body_text, control, special_tokens, router):
    esc = ref.Escape(CONTROL_SETS[control]["mask"], CONTROL_SETS[control]["control"])
    spec = {"special_tokens": special_tokens, "router": router}
    try:
        r = ref.render_router(body_text, spec, esc)
    except ref.DecisionError as e:
        return {"ok": False, "reason": e.reason, "param": e.param}
    return {"ok": True, "n_items": len(r["items"]), "digest": digest(r)}


def cases():
    rng = random.Random(20260930)
    out = []

    def add(name, body_text):
        control = rng.choice(["ml", "en"])
        st = rng.choice(["escape-control", "escape-control", "mask-to-space"])
        router = rng.choice(ROUTERS)
        out.append({"name": name, "control": control, "special_tokens": st, "router": router, "body_text": body_text,
                    "expect": expect(body_text, control, st, router)})

    for name, text in FIXED_TEXTS:
        add(name, text)
    for f in FAULTS:
        for k in range(3):
            body = rand_body(rng)
            f(body, rng)
            add("%s-%d" % (f.__name__, k), ref.py_dumps(body))
    for i in range(N_RANDOM):
        body = rand_body(rng)
        text = json.dumps(body, ensure_ascii=rng.random() < 0.3)
        add("random-%04d" % i, text)
    return out


def render():
    lines = [json.dumps({"format": "router-cases-v1", "control_sets": CONTROL_SETS}, ensure_ascii=True)]
    # raw UTF-8, but the characters str.splitlines() breaks on stay escaped, so any line reader works
    lines += [json.dumps(c, ensure_ascii=False).replace("\x85", "\\u0085").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
              for c in cases()]
    return "\n".join(lines) + "\n"


def read_golden():
    # newline="": a CRLF file stays CRLF and fails the check (read_text would hide it)
    with open(GOLDEN, encoding="utf-8", newline="") as f:
        return f.read()


def main():
    text = render()
    if "--check" in sys.argv:
        old = read_golden() if GOLDEN.exists() else ""
        if old != text:
            print("%s is stale: run python3 tests/decision/gen_router_golden.py" % GOLDEN, flush=True)
            return 1
        print("%s is up to date" % GOLDEN, flush=True)
        return 0
    with open(GOLDEN, "w", encoding="utf-8", newline="\n") as f:  # LF on Windows too: golden/ is -text
        f.write(text)
    n = text.count("\n") - 1
    n_ok = text.count('"ok": true')
    print("wrote %s: %d cases (%d accepted)" % (GOLDEN, n, n_ok), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
