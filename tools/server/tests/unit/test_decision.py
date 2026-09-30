import importlib.util
import json
import math
import os
import signal
import threading
import time

import pytest
import requests
from utils import *

# llama-server --decision on a random tiny laya GGUF (generated, no network); see DECISION.md

server: ServerProcess

CARD = {"schema": "atomic.executor-card/1", "name": "A", "kind": "local"}
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../..")


# overrides the conftest fixture: the other presets download models, this module stays offline
@pytest.fixture(scope="module", autouse=True)
def do_something():
    yield


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = tiny_laya_decision_server()


def url(path: str) -> str:
    return f"http://{server.server_host}:{server.server_port}{path}"


def post_raw(path: str, data: bytes) -> requests.Response:
    return requests.post(url(path), data=data, headers={"Content-Type": "application/json"}, timeout=60)


def assert_error(res: ServerResponse, status: int, reason: str):
    assert res.status_code == status, res.body
    err = res.body["error"]
    assert err["code"] == status
    assert err["reason"] == reason
    assert isinstance(err["message"], str) and err["type"]


def write_spec(tmp_path, spec: dict) -> str:
    path = os.path.join(tmp_path, "spec.json")
    with open(path, "w") as f:
        json.dump(spec, f)
    return path


def router_spec(**router) -> dict:
    return {
        "spec_version": 1, "model_id": "test/router", "model_version": "0.1", "layout": "laya",
        "router": {
            "card_schema": "atomic.executor-card/1", "card_renderer": "card-v1",
            "question": {"type": "noul", "instructions": "Will it work?", "criteria": {"true": "yes", "false": "no"}},
            **router,
        },
    }


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Asks for money back?"},
    "cat": {"type": "choice", "instructions": "Category?", "criteria": {"billing": "about money", "tech": None, "other": "anything else"}},
    "sev": {"type": "score", "instructions": "Severity?", "criteria": ["low", "mid", "high"]},
}


#
# info routes
#

def test_health():
    server.start()
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    assert res.body == {"status": "ok", "ok": True, "model": "tiny-laya", "layout": "laya"}


def test_default_model_name():
    server.model_alias = None
    server.start()
    assert server.make_request("GET", "/health").body["model"] == "tiny-laya-decision"


def test_default_model_name_is_model_id(tmp_path):
    # one derivation for the alias and the default spec model_id: the file name without .gguf
    path = os.path.join(tmp_path, "tiny.laya.bin")
    with open(tiny_laya_gguf(), "rb") as src, open(path, "wb") as dst:
        dst.write(src.read())
    server.model_file = path
    server.model_alias = None
    server.start()
    assert server.make_request("GET", "/health").body["model"] == "tiny.laya.bin"
    assert server.make_request("GET", "/props").body["decision"]["model_id"] == "tiny.laya.bin"


def test_bare_calibration_sidecar(tmp_path):
    # a bare calibration.json replaces laya.temperature and does not keep its version
    server.decision_spec = write_spec(tmp_path, {"temperature": 1.3})
    server.start()
    cal = server.make_request("GET", "/props").body["decision"]["calibration"]
    assert cal["method"] == "temperature" and cal["calibrated"] is True
    assert cal["version"].startswith("cal-")


def test_props():
    server.start()
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    d = res.body["decision"]
    assert d["api_version"] == 1
    assert d["layout"] == "laya" and d["format"] == "laya-v1"
    assert d["endpoints"] == ["/v1/systemone"]
    assert d["spec_source"] == "default"
    assert len(d["spec_sha256"]) == 64
    assert d["question_types"] == ["noul", "choice", "score"]
    assert d["limits"]["max_options"] == 20
    assert d["limits"]["max_questions"] == 16
    assert d["limits"]["max_body_bytes"] == 1024 * 1024
    assert d["limits"]["max_card_tokens"] == 512 * 3 // 8  # layout default: 3/8 of max_len
    assert d["limits"]["max_card_field_bytes"] == 4096
    assert d["calibration"]["calibrated"] is True  # tiny model has laya.temperature != 1
    assert d["router"]["available"] is False
    assert d["plan"]["name"] == "sequential" and d["plan"]["n_threads"] == 2
    # the tiny vocabulary has no added "\n\n" token, so splitting router states is not exact
    assert d["plan"]["state_split"] is False
    assert d["device"] == "cpu"
    assert d["queue_capacity"] == 4


def test_models():
    server.start()
    res = server.make_request("GET", "/v1/models")
    assert res.status_code == 200
    m = res.body["data"][0]
    assert m["id"] == "tiny-laya"
    assert m["capabilities"] == ["decision", "systemone"]
    assert m["decision"]["api_version"] == 1
    assert m["decision"]["layout"] == "laya"


def test_models_router_calibrated(tmp_path):
    server.decision_spec = write_spec(tmp_path, router_spec(calibration={"method": "platt", "a": 1.0, "b": 0.0}))
    server.start()
    res = server.make_request("GET", "/v1/models")
    assert res.body["data"][0]["capabilities"] == ["decision", "systemone", "router_score"]
    assert res.body["data"][0]["decision"]["model_id"] == "test/router"
    props = server.make_request("GET", "/props").body["decision"]
    assert props["spec_source"] == "file"
    assert props["router"]["calibrated"] is True
    assert "/v1/router/score" in props["endpoints"]
    # a spec without a calibration block keeps the GGUF laya.temperature calibration
    assert props["calibration"] == {"method": "temperature", "calibrated": True, "version": "gguf:laya.temperature"}


#
# systemone
#

def test_systemone_answers():
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "Billed twice, please refund", "questions": QUESTIONS})
    assert res.status_code == 200, res.body
    ans = res.body["answers"]
    assert list(ans.keys()) == ["refund", "cat", "sev"]

    assert ans["refund"]["type"] == "noul"
    assert 0.0 < ans["refund"]["noul"] < 1.0

    cat = ans["cat"]
    assert list(cat["probabilities"].keys()) == ["billing", "tech", "other"]
    assert abs(sum(cat["probabilities"].values()) - 1.0) < 1e-12
    assert cat["choice"] == max(cat["probabilities"], key=cat["probabilities"].get)

    sev = ans["sev"]
    probs = [sev["probabilities"][str(i)] for i in range(3)]
    assert abs(sum(probs) - 1.0) < 1e-12
    assert abs(sev["score"] - sum(i * p for i, p in enumerate(probs))) < 1e-12
    assert sev["legend"] == {"0": "low", "1": "mid", "2": "high"}

    for a in ans.values():
        assert 0.0 <= a["confidence"] <= 1.0
        assert "debug" not in a

    assert res.body["model"] == "tiny-laya"
    assert res.body["usage"]["output_tokens"] == 0
    assert res.body["usage"]["input_tokens"] >= res.body["usage"]["evaluated_tokens"] > 0
    rt = res.body["runtime"]
    assert rt["layout"] == "laya" and rt["plan"] == "sequential"
    assert rt["spec_sha256"] == server.make_request("GET", "/props").body["decision"]["spec_sha256"]
    assert set(res.body["timings"].keys()) == {"queue_ms", "render_ms", "compute_ms"}


def test_systemone_list_criteria():
    # layout laya: list labels as in the Laya reference (keys as json.dumps writes them, the
    # "choice" value is the label as given, no repeats by Python ==)
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {
        "q": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b", 3, True]},
    }})
    assert res.status_code == 200, res.body
    ans = res.body["answers"]["q"]
    assert list(ans["probabilities"].keys()) == ["a", "b", "3", "true"]
    labels = {"a": "a", "b": "b", "3": 3, "true": True}
    best = max(ans["probabilities"], key=ans["probabilities"].get)
    assert ans["choice"] == labels[best] and type(ans["choice"]) is type(labels[best])

    for crit in (["a", "a"], [1, 1.0], [True, 1], ["a", None], ["a", ["b"]], ["1", 1]):
        res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {
            "q": {"type": "choice", "instructions": "Pick", "criteria": crit},
        }})
        assert_error(res, 400, "UNSUPPORTED_CRITERIA_VALUE")
        assert res.body["error"]["param"] == "questions.q.criteria"


def render_tokens(questions: dict, state="Billed twice") -> list:
    res = server.make_request("POST", "/v1/decision/render", data={"state": state, "questions": questions})
    assert res.status_code == 200, res.body
    return [item["tokens"] for item in res.body["items"]]


def test_systemone_laya_reference_semantics():
    # the laya layout reads questions like the Laya reference (laya 0.3.21), see DECISION.md
    server.decision_debug = True
    server.start()
    noul = {"type": "noul", "instructions": "Urgent?"}
    # capitalised noul keys are lowercased
    assert render_tokens({"q": {**noul, "criteria": {"True": "today", "FALSE": "later"}}}) == \
        render_tokens({"q": {**noul, "criteria": {"true": "today", "false": "later"}}})
    # noul labels replace the false/true words (stripped like str.strip)
    labelled = render_tokens({"q": {**noul, "labels": {"false": " calm\u2028", "true": "angry"}}})
    assert labelled != render_tokens({"q": noul})
    assert labelled == render_tokens({"q": {**noul, "labels": {"false": "calm", "true": "angry"}}})
    # non-string instructions are json.dumps'd
    assert render_tokens({"q": {"type": "noul", "instructions": {"q": "Hostile?"}}}) == \
        render_tokens({"q": {"type": "noul", "instructions": '{"q": "Hostile?"}'}})
    # blank instructions, one choice option and one score level are valid questions
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {
        "blank": {"type": "noul", "instructions": " \u2028"},
        "one": {"type": "choice", "instructions": "Pick", "criteria": {"only": None}},
        "lvl": {"type": "score", "instructions": "Rate", "criteria": ["only"]},
    }})
    assert res.status_code == 200, res.body
    assert res.body["answers"]["one"]["probabilities"] == {"only": 1.0}
    assert res.body["answers"]["lvl"]["score"] == 0.0
    # no questions: empty answers
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {}})
    assert res.status_code == 200, res.body
    assert res.body["answers"] == {} and res.body["usage"]["input_tokens"] == 0
    # a missing state is refused
    for body in ({"questions": {"q": noul}}, {"state": None, "questions": {"q": noul}}):
        res = server.make_request("POST", "/v1/systemone", data=body)
        assert_error(res, 400, "INVALID_REQUEST")
        assert res.body["error"]["param"] == "state"


def test_systemone_deterministic():
    server.decision_debug = True
    server.start()
    body = {"state": {"ticket": "Billed twice", "amount": 12.5}, "questions": QUESTIONS}
    a = server.make_request("POST", "/v1/systemone", data=body).body
    b = server.make_request("POST", "/v1/systemone", data=body).body
    assert json.dumps(a["answers"]) == json.dumps(b["answers"])
    assert "logits" in a["answers"]["cat"]["debug"] and "tokens" in a["answers"]["cat"]["debug"]


@pytest.mark.parametrize("questions,reason", [
    ({"q": {"type": "maybe", "instructions": "x"}},                                           "UNKNOWN_QUESTION_TYPE"),
    ({"q": {"instructions": "x"}},                                                            "UNKNOWN_QUESTION_TYPE"),
    ({"q": {"type": "noul"}},                                                                 "EMPTY_INSTRUCTIONS"),
    ({"q": {"type": "choice", "instructions": "x", "criteria": {}}},                          "TOO_FEW_OPTIONS"),
    ({"q": {"type": "score", "instructions": "x", "criteria": []}},                           "TOO_FEW_OPTIONS"),
    ({"q": {"type": "score", "instructions": "x", "criteria": ["a", None]}},                  "UNSUPPORTED_CRITERIA_VALUE"),
    ({"q": {"type": "noul", "instructions": "x", "criteria": {"true": "a", "maybe": "b"}}},   "INVALID_NOUL_CRITERIA"),
    ({"q": {"type": "noul", "instructions": "x", "labels": {"false": "y", "true": "y "}}},    "INVALID_NOUL_CRITERIA"),
    ({"q": {"type": "choice", "instructions": "x", "criteria": ["a"], "labels": None}},       "UNSUPPORTED_CRITERIA_VALUE"),
    ({"q": {"type": "choice", "instructions": "x", "criteria": [str(i) for i in range(21)]}}, "TOO_MANY_OPTIONS"),
    ({"q": {"type": "noul", "instructions": "x", "criteria": ["yes", "no"]}},                 "INVALID_NOUL_CRITERIA"),
    ({"q": {"type": "choice", "instructions": "x", "criteria": [["a"], "b"]}},                "UNSUPPORTED_CRITERIA_VALUE"),
    ({"q": {"type": "score", "instructions": "x", "criteria": {"low": 1, "high": 2}}},        "UNSUPPORTED_CRITERIA_VALUE"),
])
def test_systemone_400(questions, reason):
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": questions})
    assert_error(res, 400, reason)


@pytest.mark.parametrize("raw,reason", [
    (b'{"state": "x", "questions": {',                                                          "MALFORMED_JSON"),
    (b'[1, 2]',                                                                                  "BODY_NOT_OBJECT"),
    (b'{"state": NaN, "questions": {"q": {"type": "noul", "instructions": "x"}}}',               "UNSUPPORTED_NUMBER"),
    (b'{"state": 1e999, "questions": {"q": {"type": "noul", "instructions": "x"}}}',             "UNSUPPORTED_NUMBER"),
    (b'{"state": 99999999999999999999999, "questions": {"q": {"type": "noul", "instructions": "x"}}}', "UNSUPPORTED_NUMBER"),
    (b'{"state": "x"}',                                                                          "INVALID_REQUEST"),
])
def test_systemone_400_raw(raw, reason):
    server.start()
    res = post_raw("/v1/systemone", raw)
    assert res.status_code == 400
    assert res.json()["error"]["reason"] == reason


def test_systemone_too_many_questions():
    server.decision_max_items = 2
    server.start()
    q = {"type": "noul", "instructions": "x"}
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {"a": q, "b": q, "c": q}})
    assert_error(res, 400, "TOO_MANY_QUESTIONS")
    assert server.make_request("GET", "/props").body["decision"]["limits"]["max_questions"] == 2


def test_body_too_large():
    server.start()
    body = json.dumps({"state": "x" * (1024 * 1024), "questions": {"q": {"type": "noul", "instructions": "x"}}}).encode()
    res = post_raw("/v1/systemone", body)
    assert res.status_code == 413
    assert res.json()["error"]["reason"] == "BODY_TOO_LARGE"


def test_truncation():
    server.start()
    body = {"state": "word " * 400, "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "yes", "false": "no"}}}}
    res = server.make_request("POST", "/v1/systemone", data=body)
    assert res.status_code == 200
    assert "state_truncated" in res.body["warnings"]

    res = server.make_request("POST", "/v1/systemone", data={**body, "truncation": "error"})
    assert_error(res, 422, "STATE_TRUNCATED")

    body = {"state": "x", "truncation": "error", "questions": {"q": {"type": "noul", "instructions": "x", "criteria": {"true": "yes " * 40, "false": "no"}}}}
    res = server.make_request("POST", "/v1/systemone", data=body)
    assert_error(res, 422, "OPTIONS_TRUNCATED")

    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "truncation": "sometimes", "questions": QUESTIONS})
    assert_error(res, 400, "INVALID_REQUEST")


def test_instructions_truncation():
    server.start()
    body = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "word " * 100}}}
    res = server.make_request("POST", "/v1/systemone", data=body)
    assert res.status_code == 200, res.body
    assert "instructions_truncated" in res.body["warnings"]

    res = server.make_request("POST", "/v1/systemone", data={**body, "truncation": "error"})
    assert_error(res, 422, "PROMPT_TOO_LONG")
    assert res.body["error"]["param"] == "questions.q.instructions"


def test_packed_plan():
    server.decision_plan = "packed"
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": QUESTIONS})
    assert res.status_code == 200
    assert res.body["runtime"]["plan"] == "packed"
    assert server.make_request("GET", "/props").body["decision"]["plan"]["router"] == "sequential"


def test_packed_plan_token_budget():
    # 16 full-length items are packed in groups under the token budget, not in one graph
    server.decision_plan = "packed"
    server.start()
    questions = {f"q{i}": {"type": "noul", "instructions": f"Question {i}?"} for i in range(16)}
    res = server.make_request("POST", "/v1/systemone", data={"state": "long state " * 100, "questions": questions})
    assert res.status_code == 200, res.body
    assert len(res.body["answers"]) == 16
    assert res.body["usage"]["input_tokens"] > 2048


#
# router
#

def router_body(*ids, task="Sum 2 and 2", criterion="The answer is 4", **extra):
    return {"task": task, "criterion": criterion, "candidates": [{"id": i, "card": CARD} for i in ids], **extra}


def test_router_not_calibrated():
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=router_body("a"))
    assert_error(res, 501, "ROUTER_NOT_CALIBRATED")


def test_router_uncalibrated_scores():
    server.decision_allow_uncalibrated = True
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=router_body("b", "a", "c"))
    assert res.status_code == 200, res.body
    assert res.body["object"] == "router.scores"
    assert [s["id"] for s in res.body["scores"]] == ["b", "a", "c"]
    for s in res.body["scores"]:
        assert s["calibrated"] is False
        assert s["p_success"] == pytest.approx(sigmoid(s["logit"]), abs=1e-15)
        assert s["truncated_tokens"] == 0
        assert s["input_tokens"] > 0
    assert res.body["usage"]["passes"] == 3
    assert res.body["runtime"]["calibration"] == "none"
    # uncalibrated router scores are no capability
    assert "router_score" not in server.make_request("GET", "/v1/models").body["data"][0]["capabilities"]


def test_router_candidates_independent():
    server.decision_allow_uncalibrated = True
    server.start()
    alone = server.make_request("POST", "/v1/router/score", data=router_body("a")).body["scores"][0]
    other = {"id": "z", "card": {**CARD, "name": "Z", "description": "another executor"}}
    body = router_body("a")
    body["candidates"] = [other, body["candidates"][0], other | {"id": "y"}]
    mixed = server.make_request("POST", "/v1/router/score", data=body).body["scores"]
    assert mixed[1]["id"] == "a"
    assert mixed[1]["logit"] == alone["logit"]  # sequential plan: bitwise
    assert mixed[1]["p_success"] == alone["p_success"]


def test_router_never_packed():
    server.decision_plan = "packed"
    server.decision_allow_uncalibrated = True
    server.start()
    alone = server.make_request("POST", "/v1/router/score", data=router_body("a"))
    assert alone.body["runtime"]["plan"] == "sequential"
    other = {"id": "z", "card": {**CARD, "name": "Z", "description": "another executor"}}
    body = router_body("a")
    body["candidates"] = [other, body["candidates"][0]]
    mixed = server.make_request("POST", "/v1/router/score", data=body).body["scores"]
    assert mixed[1]["logit"] == alone.body["scores"][0]["logit"]


def router_render_tokens(card_name: str, task: str) -> list:
    body = router_body("a", task=task)
    body["candidates"][0]["card"] = {**CARD, "name": card_name}
    res = server.make_request("POST", "/v1/decision/render", data=body)
    assert res.status_code == 200, res.body
    return res.body["items"][0]["tokens"]


def test_router_escape_control(tmp_path):
    # control-token text in card and task becomes a space, like the mask literal
    server.decision_spec = write_spec(tmp_path, {**router_spec(), "special_tokens": "escape-control", "input_contract": "laya-router-v1"})
    server.decision_debug = True
    server.start()
    assert server.make_request("GET", "/props").body["decision"]["special_tokens"] == "escape-control"
    plain = router_render_tokens("A B", "Sum 2 and 2")
    assert router_render_tokens("A<eos>B", "Sum 2 and 2") == plain
    assert router_render_tokens("A<pad>B", "Sum<bos>2 and 2") == router_render_tokens("A B", "Sum 2 and 2")
    assert router_render_tokens("A<mask>B", "Sum 2 and 2") == plain
    eos = 1
    assert router_render_tokens("A<eos>B", "Sum 2 and 2").count(eos) == plain.count(eos)


def test_router_mask_to_space(tmp_path):
    # the reference contract only replaces the mask literal: <eos> in a card stays a control token
    server.decision_debug = True
    server.start()
    assert server.make_request("GET", "/props").body["decision"]["special_tokens"] == "mask-to-space"
    plain = router_render_tokens("A B", "Sum 2 and 2")
    assert router_render_tokens("A<mask>B", "Sum 2 and 2") == plain
    assert router_render_tokens("A<eos>B", "Sum 2 and 2") != plain


def test_router_calibrated(tmp_path):
    server.decision_spec = write_spec(tmp_path, router_spec(calibration={"method": "platt", "a": 2.0, "b": -1.0}))
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=router_body("a"))
    assert res.status_code == 200, res.body
    s = res.body["scores"][0]
    assert s["calibrated"] is True
    assert s["p_success"] == pytest.approx(sigmoid(2.0 * s["logit"] - 1.0), abs=1e-15)
    assert res.body["runtime"]["calibration"] == "platt"


def test_router_card_too_long(tmp_path):
    server.decision_spec = write_spec(tmp_path, router_spec(max_card_tokens=10))
    server.decision_allow_uncalibrated = True
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=router_body("a"))
    assert_error(res, 422, "CARD_TOO_LONG")


def test_router_card_limits_default():
    # without a spec the card limit is 3/8 of max_len (192 tokens here: one token per character)
    server.decision_allow_uncalibrated = True
    server.start()
    body = router_body("a")
    body["candidates"][0]["card"] = {**CARD, "description": "d" * 300}
    res = server.make_request("POST", "/v1/router/score", data=body)
    assert_error(res, 422, "CARD_TOO_LONG")
    assert res.body["error"]["param"] == "candidates[0].card"

    body["candidates"][0]["card"] = {**CARD, "description": "d" * 4097}
    res = server.make_request("POST", "/v1/router/score", data=body)
    assert_error(res, 400, "INVALID_CARD")
    assert res.body["error"]["param"] == "candidates[0].card.description"


def test_router_default_question(tmp_path):
    # a spec question without criteria gets the built-in criteria, and no question is the built-in question
    global server
    scores = {}
    for name, router in [
        ("builtin", {}),
        ("explicit", {"question": {"type": "noul", "instructions": "Will the executor meet the success criterion on this task?",
                                   "criteria": {"true": "meets the criterion", "false": "does not meet the criterion"}}}),
        ("no_criteria", {"question": {"type": "noul", "instructions": "Will the executor meet the success criterion on this task?"}}),
    ]:
        server = tiny_laya_decision_server()
        server.decision_spec = write_spec(tmp_path, {"spec_version": 1, "layout": "laya", "router": {**router, "calibration": {"method": "platt", "a": 1.0, "b": 0.0}}})
        server.start()
        res = server.make_request("POST", "/v1/router/score", data=router_body("a"))
        assert res.status_code == 200, res.body
        scores[name] = res.body["scores"][0]["logit"]
        server.stop()
    assert scores["builtin"] == scores["explicit"] == scores["no_criteria"]


def test_router_truncation():
    server.decision_allow_uncalibrated = True
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=router_body("a", task="long task " * 60))
    assert res.status_code == 200
    assert res.body["scores"][0]["truncated_tokens"] > 0
    assert "task_truncated" in res.body["warnings"]

    res = server.make_request("POST", "/v1/router/score", data=router_body("a", task="long task " * 60, truncation="error"))
    assert_error(res, 422, "STATE_TRUNCATED")

    res = server.make_request("POST", "/v1/router/score", data=router_body("a", criterion="must hold " * 60))
    assert_error(res, 422, "CRITERION_TOO_LONG")


@pytest.mark.parametrize("candidates,reason", [
    ([{"id": "a", "card": CARD}, {"id": "a", "card": CARD}],                                   "DUPLICATE_CANDIDATE_ID"),
    ([{"id": "bad id!", "card": CARD}],                                                        "INVALID_CANDIDATE_ID"),
    ([{"id": "x" * 129, "card": CARD}],                                                        "INVALID_CANDIDATE_ID"),
    ([{"id": f"c{i}", "card": CARD} for i in range(17)],                                        "TOO_MANY_CANDIDATES"),
    ([{"id": "a"}],                                                                            "INVALID_CARD"),
    ([{"id": "a", "card": {**CARD, "schema": "other/1"}}],                                     "INVALID_CARD"),
    ([{"id": "a", "card": {**CARD, "checks": [{"skill": "s", "status": "measured", "passed": 1.5, "total": 2, "source": "x", "version": "1"}]}}], "INVALID_CARD"),
    ([{"id": "a", "card": {**CARD, "checks": [{"skill": "s", "status": "measured", "passed": 3, "total": 2, "source": "x", "version": "1"}]}}],   "INVALID_CARD"),
])
def test_router_400(candidates, reason):
    server.decision_allow_uncalibrated = True
    server.start()
    res = server.make_request("POST", "/v1/router/score", data={"task": "t", "criterion": "c", "candidates": candidates})
    assert_error(res, 400, reason)


def test_router_unknown_fields_warn():
    server.decision_allow_uncalibrated = True
    server.start()
    body = router_body("a")
    body["candidates"][0]["card"] = {**CARD, "color": "red"}
    res = server.make_request("POST", "/v1/router/score", data=body)
    assert res.status_code == 200
    assert any("color" in w for w in res.body["warnings"])


#
# server behaviour
#

def test_unknown_routes_404():
    server.start()
    for path in ["/v1/chat/completions", "/completion", "/v1/embeddings", "/v1/decision/render"]:
        res = server.make_request("POST", path, data={})
        assert res.status_code == 404, path
    assert server.make_request("GET", "/metrics").status_code == 404
    assert server.make_request("GET", "/").status_code == 404


def test_api_key():
    server.api_key = "sk-decision"
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": QUESTIONS})
    assert res.status_code == 401
    assert res.body["error"]["type"] == "authentication_error"
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": QUESTIONS},
                              headers={"Authorization": "Bearer sk-decision"})
    assert res.status_code == 200
    assert server.make_request("GET", "/props").status_code == 401
    for path in ["/health", "/v1/health", "/models", "/v1/models"]:
        assert server.make_request("GET", path).status_code == 200, path


def metric(name: str) -> float:
    for line in requests.get(url("/metrics"), timeout=10).text.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise KeyError(name)


def wait_for(cond, timeout: float = 20.0):
    deadline = time.time() + timeout
    while not cond():
        assert time.time() < deadline, "timed out"
        time.sleep(0.02)


def test_overloaded_429():
    # the job delay keeps the first request running; /metrics shows when each request is in place
    server.decision_debug = True
    server.decision_queue = 1
    server.server_metrics = True
    server.extra_env = {"LLAMA_DECISION_DEBUG_JOB_DELAY_MS": "3000"}
    server.start()
    body = {"state": "x", "questions": {"q": QUESTIONS["refund"]}}
    results = []

    def call():
        results.append(server.make_request("POST", "/v1/systemone", data=body))

    threads = [threading.Thread(target=call) for _ in range(2)]
    threads[0].start()
    wait_for(lambda: metric("decision_requests_total") == 1 and metric("decision_queue_waiting") == 0)
    threads[1].start()
    wait_for(lambda: metric("decision_queue_waiting") == 1)
    call()
    call()
    for t in threads:
        t.join()
    codes = sorted(r.status_code for r in results)
    # one running, one waiting, the rest rejected
    assert codes == [200, 200, 429, 429], codes
    rejected = [r for r in results if r.status_code == 429]
    for r in rejected:
        assert r.body["error"]["reason"] == "OVERLOADED"
        assert r.headers["Retry-After"] == "1"


def test_queue_zero_cancelled_job_frees_worker():
    # --decision-queue 0: a running request whose client left does not make the next one a 429
    server.decision_debug = True
    server.decision_queue = 0
    server.server_metrics = True
    server.extra_env = {"LLAMA_DECISION_DEBUG_JOB_DELAY_MS": "1500"}
    server.start()
    body = {"state": "x", "questions": {"q": QUESTIONS["refund"]}}
    with pytest.raises(requests.exceptions.ReadTimeout):
        requests.post(url("/v1/systemone"), json=body, timeout=0.5)
    # the server notices the disconnect within its 50 ms poll
    time.sleep(0.3)
    res = server.make_request("POST", "/v1/systemone", data=body)
    assert res.status_code == 200, res.body


@pytest.mark.skipif(os.name == "nt", reason="needs SIGINT")
def test_shutdown_cancels_jobs():
    # one running and three waiting requests; SIGINT must not wait for the three delayed jobs
    server.decision_debug = True
    server.server_metrics = True
    server.extra_env = {"LLAMA_DECISION_DEBUG_JOB_DELAY_MS": "2000"}
    server.start()
    body = {"state": "x", "questions": {"q": QUESTIONS["refund"]}}
    threads = [threading.Thread(target=lambda: server.make_request("POST", "/v1/systemone", data=body)) for _ in range(4)]
    for t in threads:
        t.start()
    wait_for(lambda: metric("decision_queue_waiting") == 3)
    t0 = time.time()
    server.process.send_signal(signal.SIGINT)
    server.process.wait(timeout=20)
    # the running job finishes its 2 s delay; without cancel the queue would add 6 s
    assert time.time() - t0 < 5.0
    for t in threads:
        t.join(timeout=10)


def test_loading_503():
    server.decision_debug = True
    server.extra_env = {"LLAMA_DECISION_DEBUG_LOAD_DELAY_MS": "3000"}
    starter = threading.Thread(target=server.start)
    starter.start()
    seen = None
    deadline = time.time() + 20
    while time.time() < deadline and seen is None:
        try:
            r = requests.get(url("/health"), timeout=1)
            if r.status_code == 503:
                seen = r
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.05)
    starter.join()
    assert seen is not None
    assert seen.json()["error"]["code"] == 503
    assert server.make_request("GET", "/health").status_code == 200


def test_debug_render():
    server.decision_debug = True
    server.start()
    res = server.make_request("POST", "/v1/decision/render", data={"state": "x", "questions": QUESTIONS})
    assert res.status_code == 200
    items = res.body["items"]
    assert [i["id"] for i in items] == ["refund", "cat", "sev"]
    assert items[1]["keys"] == ["billing", "tech", "other"]
    assert items[0]["n_tokens"] == len(items[0]["tokens"])
    res = server.make_request("POST", "/v1/decision/render", data=router_body("a"))
    assert res.status_code == 200
    assert res.body["items"][0]["state"].startswith("executor: A\nkind: local\n")


def test_router_tokens_match_whole_state():
    # with or without splitting, a router state tokenizes like the same text as a systemone state
    server.decision_debug = True
    server.start()
    res = server.make_request("POST", "/v1/decision/render", data=router_body("a", task="Sum <mask> 2 and 2\n\nplease"))
    assert res.status_code == 200, res.body
    item = res.body["items"][0]
    question = {"type": "noul", "instructions": "Will the executor meet the success criterion on this task?",
                "criteria": {"true": "meets the criterion", "false": "does not meet the criterion"}}
    assert render_tokens({"q": question}, state=item["state"]) == [item["tokens"]]


def test_metrics():
    server.server_metrics = True
    server.start()
    server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": QUESTIONS})
    res = requests.get(url("/metrics"), timeout=10)
    assert res.status_code == 200
    assert "llamacpp:decision_requests_total 1" in res.text
    assert "llamacpp:decision_items_total 3" in res.text


#
# start-up checks
#

def test_laya_without_decision_fails():
    server.decision = False
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)


def test_decision_in_router_child_fails():
    server.extra_env = {"LLAMA_SERVER_ROUTER_PORT": "1"}
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)


def test_unknown_plan_fails():
    server.decision_plan = "server-split"
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)


@pytest.mark.parametrize("spec", [
    {"spec_version": 1, "layout": "semif-letters"},
    {"spec_version": 1, "layout": "laya", "special_tokens": "parse"},
    {"spec_version": 1, "layout": "laya", "input_contract": "laya-v2"},
    {"spec_version": 1, "layout": "laya", "input_contract": "laya-router-v1"},
    {"spec_version": 1, "layout": "laya", "calibration": {"method": "temperature", "required": True, "temperature": {"choice": {"2": 1.2}}}},
    {"spec_version": 1, "layout": "laya", "router": {"question": {"type": "noul", "instructions": " "}}},
])
def test_bad_spec_fails(tmp_path, spec):
    server.decision_spec = write_spec(tmp_path, spec)
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)


#
# gguf_decision_spec.py applies the loader rules (the same cases run in test-decision-calib)
#

def load_spec_tool():
    path = os.path.join(REPO, "gguf-py/gguf/scripts/gguf_decision_spec.py")
    mod_spec = importlib.util.spec_from_file_location("gguf_decision_spec", path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


with open(os.path.join(REPO, "tests/decision/spec_cases.json"), encoding="utf-8") as f:
    SPEC_CASES = json.load(f)


@pytest.mark.parametrize("case", SPEC_CASES, ids=[c["name"] for c in SPEC_CASES])
def test_spec_tool_matches_loader(case):
    tool = load_spec_tool()
    if case["ok"]:
        tool.validate_spec(case["spec"])
    else:
        with pytest.raises(ValueError):
            tool.validate_spec(case["spec"])
