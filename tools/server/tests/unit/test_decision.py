import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import signal
import subprocess
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
    with open(path, "w", encoding="utf-8") as f:
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

def systemone_answers(srv: ServerProcess) -> dict:
    res = srv.make_request("POST", "/v1/systemone", data={"state": "Billed twice, please refund", "questions": QUESTIONS})
    assert res.status_code == 200, res.body
    return res.body["answers"]


def test_props_kernels_default():
    # auto: the BLAS backend when it is Accelerate (macOS builds), else the ggml CPU kernels
    server.start()
    assert server.make_request("GET", "/props").body["decision"]["plan"]["kernels"] in ("cpu", "cpu+blas")
    server.stop()
    server.decision_kernels = "default"
    server.start()
    assert server.make_request("GET", "/props").body["decision"]["plan"]["kernels"] == "cpu"


@pytest.mark.parametrize("kernels", ["default", "auto"])
def test_default_threads(kernels):
    # without -t: the performance cores (the same count for every kernels choice); BLAS: min(n, 8)
    server.n_threads = None
    server.decision_kernels = kernels
    server.start()
    plan = server.make_request("GET", "/props").body["decision"]["plan"]
    assert plan["n_threads"] >= 1
    if plan["kernels"].endswith("blas"):
        assert plan["n_threads_blas"] == min(plan["n_threads"], 8), plan
    else:
        assert plan["n_threads_blas"] == 0, plan


def weights_memory(srv: ServerProcess) -> dict:
    return srv.make_request("GET", "/props").body["decision"]["memory"]


# the token_embd mapping, --mlock (a failed lock only warns) and the warm-up pass never change a number
@pytest.mark.parametrize("load_mode,no_warmup", [("none", False), ("mmap+mlock", False), ("mlock", False), ("mmap", True)])
def test_load_modes_same_answers(load_mode, no_warmup):
    server.start()
    ref = systemone_answers(server)
    # default load mode: token_embd (518 x 64 F16 in the tiny model) is used from the file mapping
    mem = weights_memory(server)
    assert mem["weights_mapped_bytes"] == 518 * 64 * 2 and mem["weights_loaded_bytes"] > 0
    server.stop()
    other = tiny_laya_decision_server()
    other.load_mode = load_mode
    other.no_warmup = no_warmup
    other.start()
    # no silent fallback: a mapping mode maps token_embd, the others load it
    mem = weights_memory(other)
    assert mem["weights_mapped_bytes"] == (518 * 64 * 2 if load_mode.startswith("mmap") else 0), mem
    assert systemone_answers(other) == ref
    other.stop()


def apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def start_or_no_blas(srv: ServerProcess, tmp_path) -> bool:
    """ start srv; False when the build has no BLAS backend and srv asked for BLAS kernels explicitly
        (the load fails instead of computing with other kernels) """
    srv.log_path = os.path.join(tmp_path, "server.log")
    try:
        srv.start()
        return True
    except RuntimeError:
        with open(srv.log_path, encoding="utf-8", errors="replace") as f:
            log = f.read()
        assert "blas" in (srv.decision_kernels or "") and "no BLAS backend" in log, log[-2000:]
        assert not apple_silicon(), "macOS builds have Accelerate"
        return False


def assert_logits_close(ref: dict, got: dict, tol: float):
    for qid, a in ref.items():
        za, zb = a["debug"]["logits"], got[qid]["debug"]["logits"]
        assert len(za) == len(zb)
        assert max(abs(x - y) for x, y in zip(za, zb)) < tol, (qid, za, zb)


# other matmul kernels: same answers up to float rounding, and /props says which ran
@pytest.mark.parametrize("kernels", ["auto", "repack", "blas", "repack+blas"])
def test_kernels(kernels, tmp_path):
    server.decision_debug = True
    server.decision_kernels = "default"
    server.start()
    ref = systemone_answers(server)
    server.stop()
    other = tiny_laya_decision_server()
    other.decision_debug = True
    other.decision_kernels = kernels
    if not start_or_no_blas(other, tmp_path):
        return
    plan = other.make_request("GET", "/props").body["decision"]["plan"]
    # the tiny model is F32/F16: nothing to repack; auto is blas with Accelerate (macOS builds)
    if kernels == "auto":
        expected = "cpu+blas" if sys.platform == "darwin" else "cpu"
    else:
        expected = "cpu+blas" if "blas" in kernels else "cpu"
    assert plan["kernels"] == expected, plan
    # BLAS threads: min(threads, 8) (2 here); 0 without BLAS
    assert plan["n_threads_blas"] == (2 if plan["kernels"].endswith("blas") else 0), plan
    assert_logits_close(ref, systemone_answers(other), 1e-3)
    other.stop()


# Q8_0 weights: repack converts them (NEON dotprod / i8mm on Apple Silicon); same answers up to
# rounding; with repack+blas the repacked weights are not host memory, so BLAS takes none of them
@pytest.mark.parametrize("kernels", ["repack", "repack+blas"])
def test_kernels_q8_repack(kernels, tmp_path):
    server = tiny_laya_decision_server(q8=True)
    server.decision_debug = True
    server.decision_kernels = "default"
    server.start()
    ref = systemone_answers(server)
    assert weights_memory(server)["weights_repacked_bytes"] == 0
    server.stop()
    other = tiny_laya_decision_server(q8=True)
    other.decision_debug = True
    other.decision_kernels = kernels
    if not start_or_no_blas(other, tmp_path):
        return
    plan = other.make_request("GET", "/props").body["decision"]["plan"]
    mem = weights_memory(other)
    if apple_silicon():
        assert "+repack" in plan["kernels"] and mem["weights_repacked_bytes"] > 0, (plan, mem)
    assert ("+repack" in plan["kernels"]) == (mem["weights_repacked_bytes"] > 0), (plan, mem)
    assert_logits_close(ref, systemone_answers(other), 2e-2)
    other.stop()


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


#
# English checkpoints (laya, laya-typed-decisions): bytelevel-bpe tokenizer and temperature_by_options
#

def english_server() -> ServerProcess:
    global server
    server = tiny_laya_decision_server(english=True)
    server.decision_debug = True
    return server


def test_english_tokenizer():
    english_server().start()
    noul = {"type": "noul", "instructions": "x"}
    # NFC: a decomposed accent tokenizes like the composed one (the tiny vocab has no merges)
    assert render_tokens({"q": noul}, state="cafe\u0301 au lait") == render_tokens({"q": noul}, state="caf\u00e9 au lait")
    # [CLS] noul question: x [SEP] [MASK] ... [MASK] ... [SEP] state [SEP] with the ModernBERT special ids
    toks = render_tokens({"q": noul}, state="ab")[0]
    cls, sep, mask = 259, 260, 262
    assert toks[0] == cls and toks.count(mask) == 2 and toks.count(sep) == 3 and toks[-1] == sep
    # the mask literal in user text becomes a space; a space run is the normalized added token "   "
    assert render_tokens({"q": noul}, state="a[MASK]b") == render_tokens({"q": noul}, state="a b")
    assert 256 in render_tokens({"q": noul}, state="a    b")[0]


@pytest.mark.parametrize("question,temperature", [
    ({"type": "noul", "instructions": "x"}, 1.8125),  # bucket noul:2 (the base noul temperature is 1.9834)
    ({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}, 1.9063563346862793),
    ({"type": "choice", "instructions": "x", "criteria": ["a", "b", "c", "d"]}, 1.7601518630981445),
    ({"type": "choice", "instructions": "x", "criteria": [str(i) for i in range(7)]}, 1.0000158548355103),
    ({"type": "choice", "instructions": "x", "criteria": [str(i) for i in range(12)]}, 0.5),  # 0.1006 clamped to 0.5
    ({"type": "score", "instructions": "x", "criteria": ["a", "b", "c"]}, 1.375),  # bucket score:3-5
    ({"type": "score", "instructions": "x", "criteria": ["a", "b"]}, 1.2514300346374512),  # no score:2: base
])
def test_english_temperature_buckets(question, temperature):
    # laya.agent: temperature_by_options[temp_bucket(type, k)], else temperature[type]; clamped to [0.5, 5]
    english_server().start()
    res = server.make_request("POST", "/v1/systemone", data={"state": "x", "questions": {"q": question}})
    assert res.status_code == 200, res.body
    assert res.body["answers"]["q"]["debug"]["temperature"] == pytest.approx(temperature, rel=1e-6)
    assert server.make_request("GET", "/props").body["decision"]["calibration"]["calibrated"] is True


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


def test_unknown_kernels_fails():
    server.decision_kernels = "fast"
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=10)


@pytest.mark.parametrize("english", [False, True])
def test_marker_mismatch_fails(english):
    # build_sequence writes the mask id where the head looks for laya.marker_token_id
    server.model_file = tiny_laya_gguf(english, marker_mismatch=True)
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
    {"spec_version": 1, "layout": "laya", "plan": {"name": "sequential", "kernels": "fast"}},
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


#
# training and calibration tools: tools/decision/reference.py renders what the server renders;
# scripts/fit-router-calibration.py collects server logits and its spec loads (DECISION.md,
# "Training and calibration tools")
#

sys.path.insert(0, os.path.join(REPO, "tools/decision"))
import reference as dref  # noqa: E402

ROUTER_Q = {"type": "noul", "instructions": "Does the executor pass?", "criteria": {"TRUE": "it passes", "false": ""},
            "labels": {"false": " fail ", "true": "pass"}}


def golden_router_cases() -> list:
    with open(os.path.join(REPO, "tests/decision/golden/router_cases.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f.read().split("\n")[1:] if line]


def escape_router_spec() -> dict:
    return {**router_spec(question=ROUTER_Q), "special_tokens": "escape-control", "input_contract": "laya-router-v1"}


def check_render_matches_reference(english: bool, stride: int, tmp_path):
    spec = escape_router_spec()
    esc = dref.control_strings_from_gguf(tiny_laya_gguf(english=english))
    srv = tiny_laya_decision_server(english=english)
    srv.decision_spec = write_spec(tmp_path, spec)
    srv.decision_debug = True
    srv.start()
    base = f"http://{srv.server_host}:{srv.server_port}"
    n_cases = n_states = 0
    for case in golden_router_cases()[::stride]:
        body = case["body_text"].encode("utf-8")
        res = requests.post(base + "/v1/decision/render", data=body, headers={"Content-Type": "application/json"}, timeout=60)
        n_cases += 1
        try:
            want = dref.render_router(body, spec, esc)
        except dref.DecisionError as e:
            assert res.status_code == 400, (case["name"], res.text)
            err = res.json()["error"]
            assert (err["reason"], err.get("param", "")) == (e.reason, e.param), case["name"]
            continue
        assert res.status_code == 200, (case["name"], res.text)
        items = res.json()["items"]
        assert [it["id"] for it in items] == [w["id"] for w in want["items"]], case["name"]
        for it, w in zip(items, want["items"]):
            assert it["state"] == w["state"], case["name"]
            assert it["instructions"] == ROUTER_Q["instructions"] and it["keys"] == ["false", "true"]
            # escape-control: the reference's model text, rendered as a plain systemone state with the
            # same question, gives the router item's tokens
            so = requests.post(base + "/v1/decision/render", json={"state": w["model_text"], "questions": {"q": ROUTER_Q}}, timeout=60)
            assert so.status_code == 200, so.text
            assert so.json()["items"][0]["tokens"] == it["tokens"], (case["name"], it["id"])
            n_states += 1
    srv.stop()
    return n_cases, n_states


def test_router_render_matches_reference(tmp_path):
    n_cases, n_states = check_render_matches_reference(False, 1, tmp_path)
    assert n_cases >= 500 and n_states >= 500


def test_router_render_matches_reference_english(tmp_path):
    # bytelevel vocabulary: [unused0] is a USER_DEFINED escape string, the "   " added token is not
    esc = dref.control_strings_from_gguf(tiny_laya_gguf(english=True))
    assert esc.mask == "[MASK]" and "[unused0]" in esc.control and "   " not in esc.control
    n_cases, n_states = check_render_matches_reference(True, 5, tmp_path)
    assert n_cases >= 100 and n_states >= 100


def test_escape_set_of_tiny_model():
    esc = dref.control_strings_from_gguf(tiny_laya_gguf())
    assert esc.as_dict() == {"mask": "<mask>", "control": ["<bos>", "<eos>", "<mask>", "<pad>", "<unk>"]}


def test_router_spec_stamped_into_gguf(tmp_path):
    # the path fit-router-calibration.py prints: gguf_decision_spec.py set writes the spec into a
    # GGUF copy, which then loads without a sidecar; collect --spec takes the output of
    # "gguf_decision_spec.py get model.gguf > spec.json" (print adds a newline)
    pytest.importorskip("tqdm")  # gguf_new_metadata, used by set
    spec = escape_router_spec()
    spec["router"]["calibration"] = {"method": "platt", "a": 0.75, "b": -0.25}
    spec["plan"] = {"name": "sequential", "kernels": "default"}
    spec_path = os.path.join(tmp_path, "spec-router.json")
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=1)
        f.write("\n")
    spec_tool = "gguf-py/gguf/scripts/gguf_decision_spec.py"
    stamped = os.path.join(tmp_path, "tiny-router.gguf")
    res = run_py(spec_tool, "set", tiny_laya_gguf(), spec_path, "-o", stamped)
    assert res.returncode == 0, res.stdout + res.stderr
    res = run_py(spec_tool, "verify", stamped, spec_path)
    assert res.returncode == 0 and "OK" in res.stdout, res.stdout + res.stderr
    res = run_py(spec_tool, "get", stamped, text=False)
    with open(spec_path, "rb") as f:
        spec_bytes = f.read()
    assert res.returncode == 0 and res.stdout.rstrip(b"\r\n") == spec_bytes.rstrip(b"\n"), res.stderr
    spec_get = os.path.join(tmp_path, "spec-get.json")
    with open(spec_get, "wb") as f:
        f.write(res.stdout)
    assert res.stdout != spec_bytes.rstrip(b"\n")  # the newline get adds: collect must accept it

    server.model_file = stamped
    server.start()
    d = server.make_request("GET", "/props").body["decision"]
    assert d["spec_sha256"] == hashlib.sha256(spec_bytes).hexdigest() and d["plan"]["kernels"] == "cpu"
    rec = {"id": "r0", "task": "Sum 2 and 3 <eos>", "criterion": "exact",
           "candidates": [{"id": "a", "card": CARD, "outcome": 1}, {"id": "b", "card": {**CARD, "name": "B"}, "outcome": 0}]}
    scores = server.make_request("POST", "/v1/router/score", data={k: rec[k] for k in ("task", "criterion")} | {
        "candidates": [{"id": c["id"], "card": c["card"]} for c in rec["candidates"]]}).body["scores"]
    for s in scores:
        assert s["calibrated"] is True
        assert s["p_success"] == pytest.approx(sigmoid(0.75 * s["logit"] - 0.25), abs=1e-15)
    data = os.path.join(tmp_path, "one.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    url_ = f"http://{server.server_host}:{server.server_port}"
    out = os.path.join(tmp_path, "one-logits.jsonl")
    res = run_fit_tool("collect", data, out, "--url", url_, "--spec", spec_get)
    assert res.returncode == 0, res.stdout + res.stderr
    with open(out, encoding="utf-8") as f:
        got = json.loads(f.readline())
    assert [c["logit"] for c in got["candidates"]] == [s["logit"] for s in scores]
    other = os.path.join(tmp_path, "other.json")
    with open(other, "wb") as f:
        f.write(spec_bytes.replace(b"-0.25", b"-0.5"))
    res = run_fit_tool("collect", data, out, "--url", url_, "--spec", other)
    assert res.returncode == 2 and "differs from the server's spec_sha256" in res.stdout


def run_py(script: str, *args, text: bool = True) -> subprocess.CompletedProcess:
    # UTF-8 both ways, also under a Windows ANSI code page
    return subprocess.run([sys.executable, os.path.join(REPO, script), *map(str, args)], capture_output=True, timeout=600,
                          env=dict(os.environ, PYTHONIOENCODING="utf-8"), **({"text": True, "encoding": "utf-8"} if text else {}))


def run_fit_tool(*args) -> subprocess.CompletedProcess:
    return run_py("scripts/fit-router-calibration.py", *args)


def test_router_collect_fit_and_stamp(tmp_path):
    spec_path = write_spec(tmp_path, escape_router_spec())
    server.decision_spec = spec_path
    server.decision_allow_uncalibrated = True
    server.start()
    # short records for the tiny model (one token per character, max_len 512)
    rng = random.Random(3)
    recs = []
    for i in range(150):
        cands = []
        for j in range(rng.randint(1, 3)):
            total = rng.choice([10, 200])
            card = {**CARD, "name": f"exec {j}", "checks": [{"skill": "sum", "status": "measured", "passed": rng.randint(0, total), "total": total}]}
            cands.append({"id": f"c{j}", "card": card, "outcome": rng.randint(0, 1)})
        recs.append({"id": f"r{i}", "task": f"Sum {i} and {rng.randint(0, 99)} <eos>", "criterion": "exact", "candidates": cands})
    long_card = {**CARD, "description": "x" * 400}  # CARD_TOO_LONG: collect reports it and goes on
    data = os.path.join(tmp_path, "data.jsonl")
    with open(data, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        f.write(json.dumps({"id": "long", "task": "t", "criterion": "c", "candidates": [{"id": "a", "card": long_card, "outcome": 1}]}) + "\n")
    logits = os.path.join(tmp_path, "logits.jsonl")
    url = f"http://{server.server_host}:{server.server_port}"
    res = run_fit_tool("collect", data, logits, "--url", url, "--spec", spec_path)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "record long: 422 CARD_TOO_LONG" in res.stdout and "1 refused" in res.stdout, res.stdout
    with open(logits, encoding="utf-8") as f:
        got = [json.loads(line) for line in f]
    assert len(recs) == 150 and len(got) == len(recs)
    assert got[0]["engine"]["spec_sha256"] == server.make_request("GET", "/props").body["decision"]["spec_sha256"]
    direct = server.make_request("POST", "/v1/router/score", data={"task": recs[0]["task"], "criterion": recs[0]["criterion"],
                                 "candidates": [{"id": c["id"], "card": c["card"]} for c in recs[0]["candidates"]]}).body["scores"]
    assert [c["logit"] for c in got[0]["candidates"]] == [s["logit"] for s in direct]

    spec_out = os.path.join(tmp_path, "spec-router.json")
    res = run_fit_tool("fit", logits, "--spec", spec_path, "--spec-out", spec_out, "--min-examples", 100, "--min-per-class", 20,
                       "--min-eval", 20, "--resamples", 200)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "gguf_decision_spec.py set MODEL.gguf" in res.stdout
    with open(spec_out, encoding="utf-8") as f:
        cal = json.load(f)["router"]["calibration"]
    res = run_fit_tool("fit", logits, "--spec", spec_path)  # default --min-examples 1000
    assert res.returncode == 2 and "refusing to fit" in res.stdout
    server.stop()

    # the stamped spec loads: calibrated scores, same raw logits
    server.decision_spec = spec_out
    server.decision_allow_uncalibrated = False
    server.start()
    scores = server.make_request("POST", "/v1/router/score", data={"task": recs[0]["task"], "criterion": recs[0]["criterion"],
                                 "candidates": [{"id": c["id"], "card": c["card"]} for c in recs[0]["candidates"]]}).body["scores"]
    for s, d in zip(scores, direct):
        assert s["calibrated"] is True and s["logit"] == d["logit"]
        assert s["p_success"] == pytest.approx(sigmoid(cal["a"] * s["logit"] + cal["b"]), abs=1e-15)
    # fit pinned the kernels the logits came from (auto differs between macOS and the rest)
    kernels = {"cpu": "default", "cpu+repack": "repack", "cpu+blas": "blas", "cpu+repack+blas": "repack+blas"}
    with open(spec_out, encoding="utf-8") as f:
        assert json.load(f)["plan"]["kernels"] == kernels[got[0]["engine"]["plan"]["kernels"]]
