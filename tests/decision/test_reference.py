"""Offline tests of the router training and calibration tools.

    python3 -m pytest -q tests/decision/test_reference.py       (needs pytest and numpy)

Covers tools/decision/reference.py (py-json, cards, router state, escape sets), the card JSON
Schema against the reference validator, scripts/fit-router-calibration.py fit and
scripts/router-baselines.py. The engine side of the reference is test-decision-reference (ctest,
golden/router_cases.jsonl) and test_decision.py (the server's render route, collect, stamping).
Optional: TEST_DECISION_REFERENCE_BIN=build/bin/test-decision-reference also compares the escape
set of the real laya GGUFs (models-local/, build*/models/) with the engine's.
"""
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools" / "decision"))
sys.path.insert(0, str(HERE))

import reference as ref  # noqa: E402
import gen_router_golden as gen  # noqa: E402

SCHEMA = json.loads((ROOT / "tools" / "decision" / "schema" / "executor-card-v1.schema.json").read_text(encoding="utf-8"))
FIT = ROOT / "scripts" / "fit-router-calibration.py"
BASELINES = ROOT / "scripts" / "router-baselines.py"

CARD = {"schema": "atomic.executor-card/1", "name": "Qwen3.5-4B local", "kind": "local",
        "description": "4B general model on this laptop (Q4_K_M)",
        "checks": [{"skill": "field extraction", "status": "measured", "passed": 188, "total": 200, "criterion": "exact match",
                    "source": "atomic-evals/extract", "version": "2026-09"},
                   {"skill": "long-document QA", "status": "missing", "source": "atomic-evals/longqa", "version": "2026-09"}]}
CARD_TEXT = ("executor: Qwen3.5-4B local\n"
             "kind: local\n"
             "description: 4B general model on this laptop (Q4_K_M)\n"
             "checks:\n"
             "- field extraction: passed 188 of 200; criterion: exact match; source: atomic-evals/extract 2026-09\n"
             "- long-document QA: not measured; source: atomic-evals/longqa 2026-09")


#
# py-json
#

def test_py_json_golden():
    # the py-json golden of test-decision-json: strict parse + dump gives the same text
    lines = (HERE / "golden" / "py_json.jsonl").read_text(encoding="utf-8").split("\n")[:-1]
    assert len(lines) >= 100
    for line in lines:
        assert ref.py_dumps(ref.loads_strict(line)) == line
    for line in (HERE / "golden" / "py_float.txt").read_text(encoding="utf-8").split("\n")[:-1]:
        assert ref.py_dumps(ref.loads_strict(line)) == line


@pytest.mark.parametrize("text,reason", [
    ("NaN", "UNSUPPORTED_NUMBER"), ('{"x": Infinity}', "UNSUPPORTED_NUMBER"), ("[-Infinity]", "UNSUPPORTED_NUMBER"),
    ("1e400", "UNSUPPORTED_NUMBER"), ("18446744073709551616", "UNSUPPORTED_NUMBER"), ("-9223372036854775809", "UNSUPPORTED_NUMBER"),
    ('"\\ud800"', "MALFORMED_JSON"), ('{"a": "\\udc00x"}', "MALFORMED_JSON"), ("[" * 129 + "]" * 129, "MALFORMED_JSON"),
    (b'"\xff"', "MALFORMED_JSON"), ('"a\tb"', "MALFORMED_JSON"), ("{'a': 1}", "MALFORMED_JSON"), ("", "MALFORMED_JSON"),
    ("[1,]", "MALFORMED_JSON"),
    # the first failure in document order decides, as in the engine's SAX parser
    ('{"x": ' + "[" * 128 + "]" * 128 + ', "y": 1e400}', "MALFORMED_JSON"),
    ('{"y": 1e400, "x": ' + "[" * 128 + "]" * 128 + "}", "UNSUPPORTED_NUMBER"),
    ('{"t": "\\ud800", "y": NaN}', "MALFORMED_JSON"), ('{"y": NaN, "t": "\\ud800"}', "UNSUPPORTED_NUMBER"),
    ('["\\ud83d\\ude00", -Infinity]', "UNSUPPORTED_NUMBER"), ('["\\ud83d\\u0041", 1e400]', "MALFORMED_JSON"),
    ('["\\\\ud800", 1e400]', "UNSUPPORTED_NUMBER"),
])
def test_loads_strict_rejects(text, reason):
    with pytest.raises(ref.DecisionError) as e:
        ref.loads_strict(text)
    assert e.value.reason == reason


def test_loads_strict_accepts():
    assert ref.loads_strict("18446744073709551615") == 2**64 - 1
    assert ref.loads_strict("-9223372036854775808") == -(2**63)
    assert ref.loads_strict("[" * 128 + "]" * 128) is not None
    assert ref.loads_strict('"\\ud83d\\ude00"') == "\U0001F600"
    d = ref.loads_strict('{"a": 1, "b": 2, "a": 3}')
    assert list(d.items()) == [("a", 3), ("b", 2)]  # last value, first position (like decision_json_parse)
    assert ref.py_dumps(ref.loads_strict('{"x": 1.0, "y": 1e16, "z": -0.0, "s": "\\u00e9\\n"}')) == '{"x": 1.0, "y": 1e+16, "z": -0.0, "s": "\u00e9\\n"}'


def test_parse_body():
    with pytest.raises(ref.DecisionError) as e:
        ref.parse_body("[1]")
    assert e.value.reason == "BODY_NOT_OBJECT"
    with pytest.raises(ref.DecisionError) as e:
        ref.parse_body(b" " * (ref.MAX_BODY_BYTES + 1))
    assert e.value.reason == "BODY_TOO_LARGE"


#
# cards and router state
#

def test_card_render_example():
    assert ref.card_validate(CARD) == []
    assert ref.card_render(CARD) == CARD_TEXT
    card = {"schema": ref.CARD_SCHEMA, "name": " \tA\x00\x7f  B\n", "kind": "k", "description": " \n ", "checks": []}
    assert ref.card_render(card) == "executor: A B\nkind: k\nchecks: none"
    c = {"skill": "s", "status": "missing", "version": "v1"}
    assert ref.card_render({**card, "checks": [c]}).endswith("- s: not measured; source: v1")
    c = {"skill": "s", "status": "measured", "passed": 0, "total": 1, "criterion": "", "source": "src"}
    assert ref.card_render({**card, "checks": [c]}).endswith("- s: passed 0 of 1; source: src")


@pytest.mark.parametrize("patch,param", [
    ({"schema": "atomic.executor-card/2"}, "card.schema"),
    ({"name": " \n"}, "card.name"),
    ({"kind": None}, "card.kind"),
    ({"description": 1}, "card.description"),
    ({"checks": {}}, "card.checks"),
    ({"checks": [{"skill": "s", "status": "measured", "passed": 201, "total": 200}]}, "card.checks[0]"),
    ({"checks": [{"skill": "s", "status": "measured", "passed": 188.0, "total": 200}]}, "card.checks[0]"),
    ({"checks": [{"skill": "s", "status": "measured", "passed": True, "total": 2}]}, "card.checks[0]"),
    ({"checks": [{"skill": "s", "status": "measured", "passed": 1, "total": 2**63}]}, "card.checks[0]"),
    ({"checks": [{"skill": "s", "status": "missing", "total": None}]}, "card.checks[0]"),
    ({"checks": [{"skill": "s", "status": "Missing"}]}, "card.checks[0].status"),
    ({"name": "\u4e2d" * 1366}, "card.name"),  # 4098 UTF-8 bytes
])
def test_card_invalid(patch, param):
    card = {**CARD, **patch}
    with pytest.raises(ref.DecisionError) as e:
        ref.card_validate(card)
    assert e.value.reason == "INVALID_CARD" and e.value.param == param


def test_router_state_and_splits():
    body = {"task": "Sum \u4e2d", "criterion": "is 4", "candidates": [{"id": "a", "card": CARD, "x": 1}], "truncation": "error"}
    r = ref.render_router(body)
    assert r["warnings"] == ["candidates[0].x: unknown field ignored"] and r["truncation_error"] is True
    it = r["items"][0]
    assert it["state"] == CARD_TEXT + "\n\nsuccess criterion: is 4\n\ntask: Sum \u4e2d"
    assert it["state_splits"] == [len(CARD_TEXT.encode()), len((CARD_TEXT + "\n\nsuccess criterion: is 4\n\ntask:").encode())]
    assert r["options"] == ["false: does not meet the criterion", "true: meets the criterion"]
    q = {"type": "noul", "instructions": "Ok?", "criteria": {"TRUE": "works", "false": ""}, "labels": {"false": " no ", "true": "yes"}}
    assert ref.noul_options(q) == ["no: no, the statement does not hold", "yes: works"]
    assert ref.router_question({"router": {"question": {"type": "noul", "instructions": "Ok?"}}})["criteria"] == ref.DEFAULT_QUESTION["criteria"]


def test_escape_equals_str_replace():
    # DECISION.md: the strings cannot overlap and a space cannot form one, so replacing them one
    # after another in any order gives the escape-control text
    rng = random.Random(7)
    for name, cs in gen.CONTROL_SETS.items():
        esc = ref.Escape(cs["mask"], cs["control"])
        order = list(esc.control)
        for _ in range(300):
            text = gen.rand_text(rng, 0, 30)
            rng.shuffle(order)
            want = text
            for s in order:
                want = want.replace(s, " ")
            assert esc.text(text, True) == want
            assert esc.text(text, False) == text.replace(cs["mask"], " ")


def test_escape_control_needs_vocab():
    body = {"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": CARD}]}
    with pytest.raises(ValueError):
        ref.render_router(body, {"special_tokens": "escape-control"})
    ex = ref.laya_router_examples(body, {"special_tokens": "escape-control"}, ref.Escape("<mask>", ["<eos>"]))
    assert ex == [{"id": "a", "type": "noul", "instructions": ref.DEFAULT_QUESTION["instructions"],
                   "criteria": ref.DEFAULT_QUESTION["criteria"], "state": ref.router_state(CARD_TEXT, "c", "t")}]


def test_golden_is_current():
    # the engine is checked against this file (test-decision-reference); it must be what reference.py makes now
    assert gen.read_golden() == gen.render(), \
        "run python3 tests/decision/gen_router_golden.py"


def test_cli(tmp_path):
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"task": "t <eos>", "criterion": "c", "candidates": [{"id": "a", "card": CARD}]}), encoding="utf-8")
    ctl = tmp_path / "ctl.json"
    ctl.write_text(json.dumps(gen.CONTROL_SETS["ml"]), encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"special_tokens": "escape-control"}), encoding="utf-8")
    cli = [sys.executable, str(ROOT / "tools" / "decision" / "reference.py")]
    out = subprocess.run(cli + ["router", str(req), "--spec", str(spec), "--control", str(ctl)], capture_output=True, check=True)
    r = json.loads(out.stdout)
    assert r["items"][0]["model_text"] == ref.router_state(CARD_TEXT, "c", "t  ")
    raw = subprocess.run(cli + ["router", str(req), "--candidate", "a", "--text", "state"], capture_output=True, check=True).stdout
    assert raw == ref.router_state(CARD_TEXT, "c", "t <eos>").encode()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"task": "t", "criterion": "c", "candidates": [{"id": "a b", "card": CARD}]}), encoding="utf-8")
    res = subprocess.run(cli + ["router", str(bad)], capture_output=True)
    assert res.returncode == 1 and json.loads(res.stderr)["error"]["reason"] == "INVALID_CANDIDATE_ID"
    card = tmp_path / "card.json"
    card.write_text(json.dumps(CARD), encoding="utf-8")
    assert subprocess.run(cli + ["card", str(card)], capture_output=True, check=True).stdout.decode() == CARD_TEXT + "\n"


def real_ggufs():
    out = []
    for pat in ("models-local/laya-f16.gguf", "build-p4/models/laya-*-f16.gguf", "build*/models/laya-*-f16.gguf"):
        out += sorted(ROOT.glob(pat))
    return sorted(set(out))


@pytest.mark.skipif(not real_ggufs(), reason="no laya GGUF in models-local/ or build*/models/")
def test_escape_sets_of_real_models():
    # the sets the golden file uses are the ones of the real checkpoints (DECISION.md lists them)
    engine = os.environ.get("TEST_DECISION_REFERENCE_BIN")
    for path in real_ggufs():
        esc = ref.control_strings_from_gguf(str(path))
        name = "en" if esc.mask == "[MASK]" else "ml"
        want = ref.Escape(gen.CONTROL_SETS[name]["mask"], gen.CONTROL_SETS[name]["control"])
        assert esc.as_dict() == want.as_dict(), path
        if engine:
            out = subprocess.run([engine, "--control", str(path)], capture_output=True, check=True).stdout
            assert json.loads(out) == esc.as_dict(), path


#
# JSON Schema of the card vs the reference validator
#

def json_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(json_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(json_eq(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def json_type(v, t):
    return {"object": isinstance(v, dict), "array": isinstance(v, list), "string": isinstance(v, str), "null": v is None,
            "boolean": isinstance(v, bool),
            "number": isinstance(v, (int, float)) and not isinstance(v, bool),
            "integer": (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())}[t]


ANNOTATIONS = {"$schema", "$id", "title", "description", "$defs"}


def schema_valid(v, s, root=SCHEMA):
    """Draft 2020-12 evaluation for the keywords the card schema uses; any other keyword fails loudly."""
    for k, x in s.items():
        if k in ANNOTATIONS or k in ("then", "else"):
            continue
        if k == "$ref":
            assert x.startswith("#/$defs/")
            ok = schema_valid(v, root["$defs"][x[len("#/$defs/"):]], root)
        elif k == "type":
            ok = json_type(v, x)
        elif k == "const":
            ok = json_eq(v, x)
        elif k == "enum":
            ok = any(json_eq(v, e) for e in x)
        elif k == "required":
            ok = not isinstance(v, dict) or all(r in v for r in x)
        elif k == "properties":
            ok = not isinstance(v, dict) or all(schema_valid(v[p], ps, root) for p, ps in x.items() if p in v)
        elif k == "items":
            ok = not isinstance(v, list) or all(schema_valid(e, x, root) for e in v)
        elif k == "maxItems":
            ok = not isinstance(v, list) or len(v) <= x
        elif k == "maxLength":
            ok = not isinstance(v, str) or len(v) <= x
        elif k == "pattern":
            ok = not isinstance(v, str) or re.search(x, v) is not None
        elif k == "minimum":
            ok = not json_type(v, "number") or v >= x
        elif k == "maximum":
            ok = not json_type(v, "number") or v <= x
        elif k == "exclusiveMaximum":
            ok = not json_type(v, "number") or v < x
        elif k == "allOf":
            ok = all(schema_valid(v, e, root) for e in x)
        elif k == "anyOf":
            ok = any(schema_valid(v, e, root) for e in x)
        elif k == "not":
            ok = not schema_valid(v, x, root)
        elif k == "if":
            branch = "then" if schema_valid(v, x, root) else "else"
            ok = branch not in s or schema_valid(v, s[branch], root)
        else:
            raise AssertionError("keyword %r not handled" % k)
        if not ok:
            return False
    return True


def ref_valid(card):
    try:
        ref.card_validate(card)
        return True
    except ref.DecisionError:
        return False


def _strings(card):
    if not isinstance(card, dict):
        return []
    out = [v for v in card.values() if isinstance(v, str)]
    for c in card.get("checks", []) if isinstance(card.get("checks"), list) else []:
        if isinstance(c, dict):
            out += [v for v in c.values() if isinstance(v, str)]
    return out


def schema_gap(card):
    """What JSON Schema cannot express (the schema description lists it)."""
    if not isinstance(card, dict):
        return None
    if any(len(s) <= 4096 < len(s.encode("utf-8")) for s in _strings(card)):
        return "bytes"
    for c in card.get("checks", []) if isinstance(card.get("checks"), list) else []:
        if not isinstance(c, dict):
            continue
        p, t = c.get("passed"), c.get("total")
        if any(isinstance(v, float) and v.is_integer() for v in (p, t)):
            return "integral-float"
        if all(isinstance(v, int) and not isinstance(v, bool) for v in (p, t)) and p > t:
            return "passed>total"
    return None


def schema_cases():
    cards = [CARD, {**CARD, "checks": []}, {k: v for k, v in CARD.items() if k != "checks"}, {**CARD, "extra": 1}]
    for patch in [{"schema": 1}, {"name": ""}, {"name": "\u3000"}, {"name": "\x00 \x7f"}, {"kind": 5}, {"description": None},
                  {"checks": [{}]}, {"checks": [None]}, {"checks": "x"}, {"checks": [CARD["checks"][0]] * 33},
                  {"checks": [CARD["checks"][0]] * 32}, {"name": "x" * 4096}, {"name": "x" * 4097}]:
        cards.append({**CARD, **patch})
    for m in [{"passed": -1}, {"total": 0}, {"passed": 2**63 - 1, "total": 2**63 - 1}, {"total": 2**63}, {"passed": "1"},
              {"passed": None}, {"status": "missing"}, {"status": "missing", "passed": 1, "total": 2}, {"note": 1}]:
        c = {**CARD["checks"][0], **m}
        cards.append({**CARD, "checks": [c]})
    cards.append({**CARD, "checks": [{"skill": "s", "status": "measured", "total": 2}]})
    cards.append("not a card")
    # every card of the golden requests, valid and faulted
    for line in (HERE / "golden" / "router_cases.jsonl").read_text(encoding="utf-8").split("\n")[1:-1]:
        c = json.loads(line)
        try:
            body = ref.loads_strict(c["body_text"])
        except ref.DecisionError:
            continue
        cands = body.get("candidates") if isinstance(body, dict) else None
        for cand in cands if isinstance(cands, list) else []:
            if isinstance(cand, dict) and "card" in cand:
                cards.append(cand["card"])
    return cards


def test_schema_agrees_with_reference():
    n = n_valid = 0
    gaps = {}
    for card in schema_cases():
        n += 1
        r, s = ref_valid(card), schema_valid(card, SCHEMA)
        gap = schema_gap(card)
        if gap:
            gaps[gap] = gaps.get(gap, 0) + 1
            assert not r, card
            continue
        assert r == s, (card, "reference", r, "schema", s)
        n_valid += r
    assert n >= 1000 and n_valid >= 500, (n, n_valid)


@pytest.mark.parametrize("card,gap", [
    ({**CARD, "checks": [{"skill": "s", "status": "measured", "passed": 3, "total": 2}]}, "passed>total"),
    ({**CARD, "checks": [{"skill": "s", "status": "measured", "passed": 188.0, "total": 200}]}, "integral-float"),
    ({**CARD, "name": "\u4e2d" * 2000}, "bytes"),
])
def test_schema_gaps(card, gap):
    # documented gaps: the schema accepts, the engine and the reference refuse
    assert schema_gap(card) == gap
    assert schema_valid(card, SCHEMA) and not ref_valid(card)


def test_schema_with_jsonschema_package():
    jsonschema = pytest.importorskip("jsonschema")
    v = jsonschema.Draft202012Validator(SCHEMA)
    for card in schema_cases():
        assert v.is_valid(card) == schema_valid(card, SCHEMA), card


#
# calibration and baselines
#

def synth_records(n_rec, a=1.7, b=-0.4, seed=1, logits=True, mixed=False):
    """Records whose outcome depends on the card pass rate; logit z is a noisy view of the same signal."""
    rng = random.Random(seed)
    recs = []
    for i in range(n_rec):
        cands = []
        for j in range(rng.randint(1, 4)):
            q = rng.random()
            total = rng.choice([20, 50, 200])
            card = {"schema": ref.CARD_SCHEMA, "name": "exec %d" % j, "kind": rng.choice(["local", "cloud"]),
                    "checks": [{"skill": "x", "status": "measured", "passed": int(q * total), "total": total}]}
            z = 4.0 * (q - 0.5) + rng.gauss(0, 0.5)
            y = 1 if rng.random() < 1 / (1 + math.exp(-(a * z + b))) else 0
            c = {"id": "c%d" % j, "card": card, "outcome": y}
            if logits:
                c["logit"] = z
            cands.append(c)
        rec = {"id": "r%05d" % i, "task": "task %d" % i, "criterion": "correct", "candidates": cands}
        if mixed:
            rec["engine"] = {"spec_sha256": "a" if i % 2 else "b"}
        recs.append(rec)
    return recs


def write_jsonl(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    return str(path)


def run(script, *args):
    # UTF-8 both ways, also under a Windows ANSI code page
    return subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True, text=True, encoding="utf-8",
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))


def router_spec():
    return {"spec_version": 1, "model_id": "test/router", "model_version": "0.1", "layout": "laya",
            "input_contract": "laya-router-v1", "special_tokens": "escape-control",
            "router": {"card_schema": ref.CARD_SCHEMA, "card_renderer": "card-v1"}}


def test_fit_recovers_platt(tmp_path):
    pytest.importorskip("numpy")
    data = write_jsonl(tmp_path / "d.jsonl", synth_records(3000))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(router_spec()), encoding="utf-8")
    res = run(FIT, "fit", data, "--spec", spec, "--spec-out", tmp_path / "out.json", "-o", tmp_path / "rep.json", "--resamples", 300)
    assert res.returncode == 0, res.stdout + res.stderr
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    cal = rep["calibration"]
    assert cal["method"] == "platt" and abs(cal["a"] - 1.7) < 0.2 and abs(cal["b"] + 0.4) < 0.15
    # sigmoid(z) is miscalibrated for this data; the fit must clearly lower ECE on the eval split
    d = rep["eval"]["delta"]["ece"]
    assert d["delta"] < 0 and d["clear"]
    assert rep["eval"]["after"]["auc"] == pytest.approx(rep["eval"]["before"]["auc"])  # monotone map: same ranking
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["router"]["calibration"] == cal
    assert "gguf_decision_spec.py set MODEL.gguf" in res.stdout and json.dumps(cal) in res.stdout
    sys.path.insert(0, str(ROOT / "gguf-py" / "gguf" / "scripts"))
    from gguf_decision_spec import validate_spec
    validate_spec(out)


def test_fit_refuses_small_or_mixed(tmp_path):
    pytest.importorskip("numpy")
    small = write_jsonl(tmp_path / "s.jsonl", synth_records(100))
    res = run(FIT, "fit", small)
    assert res.returncode == 2 and "refusing to fit" in res.stdout
    mixed = write_jsonl(tmp_path / "m.jsonl", synth_records(3000, mixed=True))
    res = run(FIT, "fit", mixed)
    assert res.returncode == 2 and "engine identities" in res.stdout
    # collected logits mixed with logits of unknown origin: a missing engine block is its own identity
    part = synth_records(3000)
    part[0]["engine"] = {"spec_sha256": "a"}
    res = run(FIT, "fit", write_jsonl(tmp_path / "p.jsonl", part))
    assert res.returncode == 2 and "2 engine identities" in res.stdout and "(no engine block)" in res.stdout
    res = run(BASELINES, tmp_path / "p.jsonl", "-o", tmp_path / "prep.json", "--resamples", 50)
    assert res.returncode == 0 and "engine-raw and engine-platt left out" in res.stdout, res.stdout + res.stderr
    assert "engine-raw" not in json.loads((tmp_path / "prep.json").read_text(encoding="utf-8"))["eval"]
    nolog = write_jsonl(tmp_path / "n.jsonl", synth_records(50, logits=False))
    res = run(FIT, "fit", nolog)
    assert res.returncode == 2 and "has no logit" in res.stdout
    bad = synth_records(10)
    bad[3]["candidates"][0]["card"]["checks"][0]["passed"] = 999
    res = run(FIT, "fit", write_jsonl(tmp_path / "b.jsonl", bad))
    assert res.returncode == 2 and "b.jsonl:4" in res.stdout and "INVALID_CARD" in res.stdout
    res = run(FIT, "fit", small, "--spec-out", tmp_path / "x.json")
    assert res.returncode == 2 and "needs --spec" in res.stdout


def test_fit_before_and_kernels(tmp_path):
    pytest.importorskip("numpy")
    recs = synth_records(3000)
    for r in recs:
        r["engine"] = {"spec_sha256": "s", "plan": {"router": "sequential", "kernels": "cpu+blas", "recipe": None}}
    data = write_jsonl(tmp_path / "d.jsonl", recs)
    spec = router_spec()
    spec["router"]["calibration"] = {"method": "platt", "a": 1.5, "b": -0.3}
    spec["plan"] = {"name": "sequential", "kernels": "auto"}
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    res = run(FIT, "fit", data, "--spec", tmp_path / "spec.json", "--spec-out", tmp_path / "out.json", "-o", tmp_path / "rep.json",
              "--resamples", 100)
    assert res.returncode == 0, res.stdout + res.stderr
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    # "before" is the calibration that ships, not sigmoid(z)
    assert rep["before"] == {"a": 1.5, "b": -0.3, "source": "spec router.calibration"}
    # the kernels the logits came from are pinned: auto would be default on Linux and Windows
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["plan"] == {"name": "sequential", "kernels": "blas"} and rep["plan_kernels"] == "blas"
    assert 'plan.kernels: "auto" -> blas' in res.stdout
    res = run(FIT, "fit", data, "--spec", tmp_path / "spec.json", "--before", "1,0", "-o", tmp_path / "rep2.json", "--resamples", 100)
    assert res.returncode == 0, res.stdout + res.stderr
    assert json.loads((tmp_path / "rep2.json").read_text(encoding="utf-8"))["before"]["source"] == "--before"
    # logits without an engine block: nothing to pin, say so
    res = run(FIT, "fit", write_jsonl(tmp_path / "n.jsonl", synth_records(3000)), "--spec", tmp_path / "spec.json",
              "--spec-out", tmp_path / "out2.json", "--resamples", 100)
    assert res.returncode == 0 and "plan.kernels left as it is" in res.stdout, res.stdout + res.stderr
    assert json.loads((tmp_path / "out2.json").read_text(encoding="utf-8"))["plan"]["kernels"] == "auto"


def test_baselines(tmp_path):
    pytest.importorskip("numpy")
    data = write_jsonl(tmp_path / "d.jsonl", synth_records(3000))
    res = run(BASELINES, data, "-o", tmp_path / "rep.json", "--resamples", 300)
    assert res.returncode == 0, res.stdout + res.stderr
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    assert rep["reference"] == "engine-platt"
    assert set(rep["eval"]) == {"base-rate", "pass-rate", "pass-rate-platt", "logreg", "engine-raw", "engine-platt"}
    # the outcome follows the pass rate, so the card baselines must clearly beat the base rate
    base = rep["eval"]["base-rate"]
    for name in ("pass-rate-platt", "logreg"):
        assert rep["eval"][name]["brier"] < base["brier"] and rep["eval"][name]["auc"] > 0.75
    assert rep["delta_vs_reference"]["base-rate"]["brier"]["clear"]
    assert base["auc"] == 0.5
    # without logits the engine rows disappear and the reference falls back
    data = write_jsonl(tmp_path / "n.jsonl", synth_records(3000, logits=False))
    res = run(BASELINES, data, "-o", tmp_path / "rep2.json", "--resamples", 100)
    assert res.returncode == 0, res.stdout + res.stderr
    rep = json.loads((tmp_path / "rep2.json").read_text(encoding="utf-8"))
    assert rep["reference"] == "pass-rate-platt" and "engine-raw" not in rep["eval"]
    res = run(BASELINES, write_jsonl(tmp_path / "s.jsonl", synth_records(50)))
    assert res.returncode == 2 and "refusing to fit" in res.stdout


def test_same_split_in_both_scripts(tmp_path):
    pytest.importorskip("numpy")
    import router_eval as E
    recs = synth_records(500)
    fit, ev = E.split_records(recs, 0.3, "router")
    assert fit and ev and len(fit) + len(ev) == len(recs)
    h = hashlib.sha256(b"router:r00000").digest()
    assert (int.from_bytes(h[:8], "big") / 2**64 < 0.3) == (recs[0] in ev)
    recs[0]["split"] = "eval"
    with pytest.raises(E.DataError):
        E.split_records(recs)
