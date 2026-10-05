import os

import pytest
from utils import *

# llama-server --decision on a random tiny Clef GGUF (generated, no network); see DECISION.md "Clef"

server: ServerProcess


# overrides the conftest fixture: the other presets download models, this module stays offline
@pytest.fixture(scope="module", autouse=True)
def do_something():
    yield


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = tiny_clef_decision_server()


QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Asks for money back?"},
    "cat": {"type": "choice", "instructions": "Category?", "criteria": {"tech": None, "billing": "about money", "other": "anything else"}},
    "sev": {"type": "score", "criteria": ["low", "mid", "high"]},
}


def systemone(body: dict) -> ServerResponse:
    return server.make_request("POST", "/v1/systemone", data=body)


def assert_error(res: ServerResponse, status: int, reason: str):
    assert res.status_code == status, res.body
    assert res.body["error"]["reason"] == reason


def test_props():
    server.start()
    dec = server.make_request("GET", "/props").body["decision"]
    assert dec["layout"] == "clef" and dec["format"] == "clef-v1"
    assert dec["spec_source"] == "default" and dec["confidence"] == "max_p"
    assert dec["plan"]["name"] == "joint"
    assert dec["limits"]["max_tokens"] == 2048
    assert dec["limits"]["max_options"] == 255
    assert server.make_request("GET", "/health").body["layout"] == "clef"


def test_answers():
    server.start()
    res = systemone({"state": "Billed twice, please refund", "questions": QUESTIONS})
    assert res.status_code == 200, res.body
    a = res.body["answers"]
    assert list(a) == ["refund", "cat", "sev"]
    assert 0.0 < a["refund"]["noul"] < 1.0
    # probabilities in criteria order, the choice is the most probable option, confidence = its probability
    assert list(a["cat"]["probabilities"]) == ["tech", "billing", "other"]
    assert abs(sum(a["cat"]["probabilities"].values()) - 1.0) < 1e-9
    assert a["cat"]["choice"] == max(a["cat"]["probabilities"], key=a["cat"]["probabilities"].get)
    assert a["cat"]["confidence"] == a["cat"]["probabilities"][a["cat"]["choice"]]
    probs = a["sev"]["probabilities"]
    assert abs(a["sev"]["score"] - sum(i * probs[str(i)] for i in range(3))) < 1e-9
    assert a["sev"]["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert res.body["runtime"]["layout"] == "clef" and res.body["runtime"]["plan"] == "joint"
    # one prompt for all questions
    assert res.body["usage"]["input_tokens"] == res.body["usage"]["evaluated_tokens"] > 0


def test_deterministic():
    server.start()
    body = {"state": {"b": 1.5, "a": ["x", None]}, "questions": QUESTIONS}
    assert systemone(body).body["answers"] == systemone(body).body["answers"]


def test_joint():
    # the head reads all questions together: another question changes the answer
    server.start()
    one = systemone({"state": "s", "questions": {"refund": QUESTIONS["refund"]}}).body["answers"]["refund"]["noul"]
    two = systemone({"state": "s", "questions": QUESTIONS}).body["answers"]["refund"]["noul"]
    assert one != two


def test_instructions_optional():
    # joint_schema_model.py: missing, null or "" instructions are the question id
    server.start()
    base = systemone({"state": "s", "questions": {"refund": {"type": "noul", "instructions": "refund"}}}).body["answers"]
    for q in ({"type": "noul"}, {"type": "noul", "instructions": None}, {"type": "noul", "instructions": ""}):
        res = systemone({"state": "s", "questions": {"refund": q}})
        assert res.status_code == 200, res.body
        assert res.body["answers"] == base


def test_debug_tokens():
    server.decision_debug = True
    server.start()
    res = systemone({"state": "s", "questions": QUESTIONS})
    a = res.body["answers"]
    # the prompt tokens are on the first question only
    assert len(a["refund"]["debug"]["tokens"]) == res.body["usage"]["input_tokens"]
    assert a["cat"]["debug"]["tokens"] == [] and a["sev"]["debug"]["tokens"] == []
    assert len(a["cat"]["debug"]["logits"]) == 3


def test_request_errors():
    server.start()
    assert_error(systemone({"questions": QUESTIONS}), 400, "INVALID_REQUEST")
    assert systemone({"state": None, "questions": QUESTIONS}).status_code == 200
    assert_error(systemone({"state": "s", "questions": {}}), 400, "INVALID_REQUEST")
    assert_error(systemone({"state": "s", "questions": {"q": {"type": "rank"}}}), 400, "UNKNOWN_QUESTION_TYPE")
    assert_error(systemone({"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}), 400, "TOO_FEW_OPTIONS")
    assert_error(systemone({"state": "s", "questions": {"q": {"type": "noul", "criteria": [1]}}}), 400, "INVALID_NOUL_CRITERIA")


def test_truncation():
    # the state is cut from the end to fit -c; the questions alone must fit
    server.n_ctx = 1200
    server.start()
    long_state = "x" * 3000
    res = systemone({"state": long_state, "questions": QUESTIONS})
    assert res.status_code == 200, res.body
    assert res.body["warnings"] == ["state_truncated"]
    assert res.body["usage"]["input_tokens"] == 1200
    assert_error(systemone({"state": long_state, "questions": QUESTIONS, "truncation": "error"}), 422, "STATE_TRUNCATED")
    many = {f"q{i}": {"type": "noul", "instructions": "y" * 200} for i in range(8)}
    assert_error(systemone({"state": "s", "questions": many}), 422, "PROMPT_TOO_LONG")


def test_checkpoint_dir_hint(tmp_path):
    # -m <Clef HF repo>: refused with a pointer to the GGUF, nothing is converted
    ckpt = os.path.join(tmp_path, "clef-flash")
    os.makedirs(ckpt)
    for name in ("config.json", "joint_head_config.json"):
        with open(os.path.join(ckpt, name), "w") as f:
            f.write("{}")
    server.model_file = ckpt
    server.model_alias = None
    server.log_path = os.path.join(tmp_path, "log.txt")
    with pytest.raises(RuntimeError):
        server.start(timeout_seconds=20)
    with open(server.log_path, encoding="utf-8", errors="replace") as f:
        log = f.read()
    assert "is a Clef checkpoint" in log and "ggml-org/Clef-Flash-GGUF" in log, log[-2000:]
    assert sorted(os.listdir(ckpt)) == ["config.json", "joint_head_config.json"]


def test_router_uncalibrated():
    server.start()
    body = {"task": "t", "criterion": "c", "candidates": [{"id": "a", "card": {"schema": "atomic.executor-card/1", "name": "A", "kind": "local"}}]}
    assert_error(server.make_request("POST", "/v1/router/score", data=body), 501, "ROUTER_NOT_CALIBRATED")
    server.stop()
    server.decision_allow_uncalibrated = True
    server.start()
    res = server.make_request("POST", "/v1/router/score", data=body)
    assert res.status_code == 200, res.body
    assert 0.0 < res.body["scores"][0]["p_success"] < 1.0
