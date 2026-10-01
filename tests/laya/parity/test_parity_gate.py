"""Unit tests of the public-output gate (tests/laya/parity/parity_gate.py).

The first three cases are ported from laya.cpp tests/test_compare.py:11-23
(https://github.com/lkarlslund/laya.cpp), MIT License, Copyright (c) 2026 Lars Karlslund;
the license text is in parity_gate.py.
"""

import copy
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_gate as G  # noqa: E402


# ---- ported from laya.cpp tests/test_compare.py

def test_rejects_wrong_label_even_with_close_confidence():
    with pytest.raises(ValueError):
        G.compare_values({"choice": "billing", "confidence": .99}, {"choice": "sales", "confidence": .99}, .0001)


def test_rejects_nonfinite_and_missing_fields():
    for actual in ({"p": float("nan")}, {"p": float("inf")}, {}, {"p": .5, "extra": 0}):
        with pytest.raises(ValueError):
            G.compare_values({"p": .5}, actual, .0001)


def test_decimal_output_boundary():
    G.compare_values({"p": .8000}, {"p": .8001}, .0001)
    with pytest.raises(ValueError):
        G.compare_values({"p": .8000}, {"p": .8002}, .0001)


# ---- ours

def test_phase4_boundary_value_passes():
    # English F32 CPU against the 4-digit reference answers: the largest error is 1.0000000000000167e-4
    assert G.within(0.00010000000000001674, 1e-4)
    assert not G.within(0.000101, 1e-4)


def _q(logits, answer, ids=(2, 5, 4, 7, 4, 8, 1), act=None):
    q = {"input_ids": list(ids), "raw_logits": list(logits)}
    if act is not None:
        q["act_raw_logits"] = list(act)
    return q, answer


def _run(questions, extra=None):
    """questions: {qid: (per_question, answer)} -> {id: record}."""
    rec = {"id": "it_0", "per_question": {k: v[0] for k, v in questions.items()}, "answers": {k: v[1] for k, v in questions.items()}}
    if extra:
        rec.update(extra)
    return {"it_0": rec}


def _choice(p):
    return {"type": "choice", "choice": "a" if p[0] >= p[1] else "b", "probabilities": {"a": p[0], "b": p[1]}, "confidence": abs(p[0] - p[1])}


STRICT = {"public_atol": 1e-4, "act_max_rel_dlogit": 2e-4, "known_differences": ["server_answer_shape", "server_no_act_head", "null_state_refused"]}
F16 = {"flip_max_gap": 0.009, "max_flip_fraction": 1.0, "max_abs_dp": 0.5, "mean_abs_dp": 0.5, "act_max_rel_dlogit": 0.2, "known_differences": ["server_no_act_head"]}


def test_strict_pass_and_category_fail():
    base = _run({"q": _q([1.0, 0.0], _choice([0.7311, 0.2689]), act=[1000.0, -900.0])})
    cand = _run({"q": _q([1.00001, 0.0], _choice([0.73110001, 0.26889999]), act=[1000.05, -900.0])})
    r = G.gate_run(base, cand, "strict-f32", STRICT)
    assert r["verdict"] == "pass", r["failures"]
    assert r["act_checked"] == 1

    bad = copy.deepcopy(cand)
    bad["it_0"]["answers"]["q"]["choice"] = "b"
    assert G.gate_run(base, bad, "strict-f32", STRICT)["verdict"] == "fail"


def test_strict_rejects_argmax_flip_and_act_drift():
    base = _run({"q": _q([1.0, 0.99], _choice([0.5025, 0.4975]), act=[1000.0, -900.0])})
    flip = _run({"q": _q([0.99, 1.0], _choice([0.5025, 0.4975]), act=[1000.0, -900.0])})
    assert G.gate_run(base, flip, "strict-f32", STRICT)["verdict"] == "fail"
    drift = _run({"q": _q([1.0, 0.99], _choice([0.5025, 0.4975]), act=[1001.0, -900.0])})
    r = G.gate_run(base, drift, "strict-f32", STRICT)
    assert r["verdict"] == "fail" and r["act_max_rel_dlogit"] == pytest.approx(1.0 / 1001.0)


def test_input_mismatch_is_not_compared_and_fails():
    base = _run({"q": _q([1.0, 0.0], _choice([0.7311, 0.2689]))})
    cand = _run({"q": _q([1.0, 0.0], _choice([0.7311, 0.2689]), ids=(2, 5, 4, 7, 4, 9, 1))})
    r = G.gate_run(base, cand, "strict-f32", STRICT)
    assert r["compared"] == 0 and r["verdict"] == "fail" and sum(r["input_mismatches"].values()) == 1


def test_known_differences_are_counted_apart():
    ref_answer = dict(_choice([0.7311, 0.2689]), answer_confidence=0.7311, action={"act_probability": 1.0})
    base = _run({"q": _q([1.0, 0.0], ref_answer, act=[1000.0, -900.0])})
    srv = _run({"q": _q([1.0, 0.0], dict(_choice([0.7311, 0.2689]), debug={"tokens": [1]}))})
    r = G.gate_run(base, srv, "strict-f32", STRICT)
    assert r["verdict"] == "pass", r["failures"]
    assert r["known_differences"] == {"server_no_act_head": 1, "server_answer_shape": 1}
    # the same shapes are not allowed where the tier does not list them
    assert G.gate_run(base, srv, "strict-f32", dict(STRICT, known_differences=[]))["verdict"] == "fail"


def test_server_action_is_compared():
    # a server answer with action (no answer_confidence) against the reference shape: act_probability and
    # the act logits (debug.act_logits) are gated; only answer_confidence is a known difference
    ref_answer = dict(_choice([0.7311, 0.2689]), answer_confidence=0.7311, action={"act_probability": 1.0})
    base = _run({"q": _q([1.0, 0.0], ref_answer, act=[1000.0, -900.0])})
    srv = _run({"q": _q([1.0, 0.0], dict(_choice([0.7311, 0.2689]), action={"act_probability": 1.0}), act=[1000.05, -900.0])})
    r = G.gate_run(base, srv, "strict-f32", STRICT)
    assert r["verdict"] == "pass", r["failures"]
    assert r["act_checked"] == 1
    assert r["known_differences"] == {"server_answer_shape": 1}
    bad = copy.deepcopy(srv)
    bad["it_0"]["answers"]["q"]["action"]["act_probability"] = 0.9998
    assert G.gate_run(base, bad, "strict-f32", STRICT)["verdict"] == "fail"
    drift = copy.deepcopy(srv)
    drift["it_0"]["per_question"]["q"]["act_raw_logits"] = [1000.5, -900.0]  # relative 5e-4 > 2e-4
    assert G.gate_run(base, drift, "strict-f32", STRICT)["verdict"] == "fail"
    # f16-class: act_probability is a probability (max / mean |dp|), the act logits a relative dlogit
    cpu = _run({"q": _q([1.0, 0.0], dict(_choice([0.7311, 0.2689]), action={"act_probability": 1.0}), act=[1000.0, -900.0])})
    dev = _run({"q": _q([1.001, 0.0], dict(_choice([0.7313, 0.2687]), action={"act_probability": 1.0}), act=[1100.0, -900.0])})
    r = G.gate_run(cpu, dev, "f16-class", F16)
    assert r["verdict"] == "pass", r["failures"]
    assert r["act_max_rel_dlogit"] > 0.09
    dev["it_0"]["per_question"]["q"]["act_raw_logits"] = [1500.0, -900.0]
    assert G.gate_run(cpu, dev, "f16-class", F16)["verdict"] == "fail"


def test_null_state_refusal():
    base = {"it_0": {"id": "it_0", "state": None, "questions": {"n": {"type": "noul", "instructions": "x"}},
                     "per_question": {"n": {"input_ids": [1], "raw_logits": [0.0, 1.0]}}, "api": {}}}
    cand = {"it_0": {"id": "it_0", "error": "http 400: state is required"}}
    ok = _run({"q": _q([1.0, 0.0], _choice([0.7311, 0.2689]))})
    base.update({"it_1": dict(ok["it_0"], id="it_1")})
    cand.update({"it_1": dict(ok["it_0"], id="it_1")})
    r = G.gate_run(base, cand, "strict-f32", STRICT)
    assert r["verdict"] == "pass", r["failures"]
    assert r["known_differences"]["null_state_refused"] == 1


def test_f16_flip_allowed_only_below_gap():
    near = _run({"q": _q([1.0, 0.99], _choice([0.5025, 0.4975]))})       # gap 0.005
    near_flip = _run({"q": _q([0.99, 1.0], _choice([0.4975, 0.5025]))})
    r = G.gate_run(near, near_flip, "f16-class", F16)
    assert r["verdict"] == "pass", r["failures"]
    assert len(r["allowed_flips"]) == 1

    far = _run({"q": _q([1.0, 0.9], _choice([0.525, 0.475]))})           # gap 0.05
    far_flip = _run({"q": _q([0.9, 1.0], _choice([0.475, 0.525]))})
    assert G.gate_run(far, far_flip, "f16-class", F16)["verdict"] == "fail"


def test_f16_probability_bound_and_finite():
    base = _run({"q": _q([2.0, 0.0], _choice([0.8808, 0.1192]))})
    close = _run({"q": _q([2.01, 0.0], _choice([0.8818, 0.1182]))})
    assert G.gate_run(base, close, "f16-class", F16)["verdict"] == "pass"
    far = _run({"q": _q([2.0, 0.0], _choice([0.2, 0.8]))})
    assert G.gate_run(base, far, "f16-class", dict(F16, max_abs_dp=0.1))["verdict"] == "fail"
    nan = _run({"q": _q([2.0, 0.0], _choice([float("nan"), 0.1192]))})
    assert G.gate_run(base, nan, "f16-class", F16)["verdict"] == "fail"


def test_empty_gate_fails():
    base = _run({"q": _q([1.0, 0.0], _choice([0.7311, 0.2689]))})
    cand = copy.deepcopy(base)
    cand["it_0"]["answers"] = {}
    r = G.gate_run(base, cand, "strict-f32", STRICT)
    assert r["verdict"] == "fail" and r["public_checked"] == 0


def test_top2_gap_and_rel_dlogit():
    assert G.top2_gap([0.0, 0.0]) == 0.0
    assert G.top2_gap([5.0]) == 1.0
    assert G.rel_dlogit([4000.0, -3000.0], [4000.4, -3000.0]) == pytest.approx(0.4 / 4000.4)
    assert G.rel_dlogit([0.1, 0.2], [0.1, 0.3]) == pytest.approx(0.1)


def _ident(corpus="c1", subset="all", gguf="g1", weights="f32", revision="r1", backend="cpu", precision="default", kernels=None):
    if kernels is None:
        kernels = "cpu+blas" if backend == "cpu" else backend
    return {"corpus": {"sha256": corpus, "subset": subset}, "model": {"sha256": gguf, "weights": weights, "revision": revision},
            "runtime": {"backend": backend, "precision": precision, "kernels": kernels}}


def test_identity_rules():
    tiers = G.load_tiers()
    strict = G.tier_config(tiers, "strict-f32")
    ref = {"corpus": {"sha256": "c1", "subset": "all"}, "model": {"revision": "r1", "weights": "f32"}, "runtime": {"backend": "torch-cpu"}}
    assert G.check_identity(ref, _ident(), "strict-f32", strict) == []
    assert G.check_identity(ref, _ident(corpus="c2"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(subset="english"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(revision="r2"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(weights="f16"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(backend="metal"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(backend="cuda"), "strict-f32", strict)
    assert G.check_identity(ref, _ident(backend="cuda", precision="strict"), "strict-f32", strict) == []
    assert G.check_identity(None, _ident(), "strict-f32", strict)
    assert G.check_identity(_ident(), _ident(), "strict-f32", strict)  # a CPU run is no reference

    f16 = G.tier_config(tiers, "f16-class", "f16")
    cpu, metal = _ident(weights="f16"), _ident(weights="f16", backend="metal")
    assert G.check_identity(cpu, metal, "f16-class", f16) == []
    assert G.check_identity(cpu, _ident(weights="f16", backend="metal", gguf="g2"), "f16-class", f16)
    assert G.check_identity(metal, cpu, "f16-class", f16)
    assert G.check_identity(cpu, metal, "f16-class", G.tier_config(tiers, "f16-class", "q8_0"))


def test_tiers_file_is_complete():
    tiers = G.load_tiers()
    with pytest.raises(ValueError):
        G.tier_config(tiers, "f16-class")
    for w in ("f16", "f32", "q8_0"):
        cfg = G.tier_config(tiers, "f16-class", w)
        for k in ("flip_max_gap", "max_flip_fraction", "max_abs_dp", "mean_abs_dp", "act_max_rel_dlogit"):
            assert cfg[k] > 0
    cfg = G.tier_config(tiers, "strict-f32")
    assert cfg["public_atol"] == 1e-4 and cfg["act_max_rel_dlogit"] > 0


# ---- tiers v2

def _score(probs, score):
    return {"type": "score", "score": score, "probabilities": {str(i): p for i, p in enumerate(probs)}, "confidence": 0.1}


def test_strict_v2_rounds_like_the_reference_and_scales_score():
    # the reference rounds to 4 digits, the server does not: 0.5480797 rounds to 0.5481, one step from 0.5482
    base = _run({"q": _q([1.0, 0.0], dict(_choice([0.5482, 0.4518]), confidence=0.0964), act=[1000.0, -900.0])})
    srv = _run({"q": _q([1.0, 0.0], dict(_choice([0.5480797089, 0.4519202911]), confidence=0.0963501), act=[1000.0, -900.0])})
    strict = dict(STRICT, known_differences=[])
    r = G.gate_run(base, srv, "strict-f32", strict)
    assert r["verdict"] == "pass", r["failures"]
    far = _run({"q": _q([1.0, 0.0], _choice([0.54795, 0.45205]), act=[1000.0, -900.0])})  # rounds to 0.548: two steps
    assert G.gate_run(base, far, "strict-f32", strict)["verdict"] == "fail"
    # a score of K = 10 options: 4.9e-4 raw is 5.4e-5 per step of the range (K - 1), reported both ways
    p = [0.1] * 10
    base = _run({"s": _q([0.0] * 10, _score(p, 3.4511), act=[1000.0, -900.0])})
    cand = _run({"s": _q([0.0] * 10, _score(p, 3.451589), act=[1000.0, -900.0])})
    r = G.gate_run(base, cand, "strict-f32", strict)
    assert r["verdict"] == "pass", r["failures"]
    assert r["max_abs_error"]["score"] == pytest.approx(0.0005 / 9) and r["max_abs_error"]["score_raw"] == pytest.approx(0.0005)
    worse = _run({"s": _q([0.0] * 10, _score(p, 3.4531), act=[1000.0, -900.0])})  # 2e-3 / 9 > 1e-4
    assert G.gate_run(base, worse, "strict-f32", strict)["verdict"] == "fail"


def test_strict_v2_unrounded_reference_is_not_rounded():
    # a reference answer that is not 4-digit (another engine run) is compared as it is
    base = _run({"q": _q([1.0, 0.0], _choice([0.54807970, 0.45192030]), act=[1000.0, -900.0])})
    cand = _run({"q": _q([1.0, 0.0], _choice([0.54822, 0.45178]), act=[1000.0, -900.0])})
    assert G.gate_run(base, cand, "strict-f32", dict(STRICT, known_differences=[]))["verdict"] == "fail"


def test_v2_act_head_and_public_answers_must_be_there():
    a = _choice([0.7311, 0.2689])
    with_act = {"q1": _q([1.0, 0.0], a, act=[1000.0, -900.0]), "q2": _q([1.0, 0.0], a, act=[1000.0, -900.0])}
    base = _run(with_act)
    # some questions of a run that has act logits elsewhere: always a failure
    part = _run({"q1": _q([1.0, 0.0], a, act=[1000.0, -900.0]), "q2": _q([1.0, 0.0], a)})
    assert G.gate_run(base, part, "f16-class", F16)["verdict"] == "fail"
    # a whole run without them: a known difference of f16-class (old server builds), not of strict-f32 v2
    none = _run({"q1": _q([1.0, 0.0], a), "q2": _q([1.0, 0.0], a)})
    r = G.gate_run(base, none, "f16-class", F16)
    assert r["verdict"] == "pass" and r["known_differences"] == {"server_no_act_head": 2}, r["failures"]
    tiers = G.load_tiers()
    strict = G.tier_config(tiers, "strict-f32")
    assert "server_no_act_head" not in strict["known_differences"]
    assert G.gate_run(base, none, "strict-f32", strict)["verdict"] == "fail"
    # a missing public answer fails
    miss = copy.deepcopy(base)
    del miss["it_0"]["answers"]["q2"]
    r = G.gate_run(base, miss, "f16-class", F16)
    assert r["verdict"] == "fail" and r["public_missing"] == 1


def test_v2_activation_class_rules():
    tiers = G.load_tiers()
    assert tiers["version"] == 2
    f16, q8 = G.tier_config(tiers, "f16-class", "f16"), G.tier_config(tiers, "f16-class", "q8_0")
    q8_aq = G.tier_config(tiers, "f16-class", "q8_0", activation_quantized=True)
    assert q8["flip_max_gap"] == f16["flip_max_gap"] and q8_aq["flip_max_gap"] > q8["flip_max_gap"]
    blas_q8 = _ident(weights="q8_0", kernels="cpu+blas")
    default_q8 = _ident(weights="q8_0", kernels="cpu")
    cuda_q8 = _ident(weights="q8_0", backend="cuda")
    assert G.activation_class(default_q8) == "quantized" and G.activation_class(blas_q8) == "f32"
    assert G.activation_class(_ident(weights="q8_0", kernels="auto")) is None
    # a device Q8_0 run: F32-activation baseline, f16 numbers
    assert G.check_identity(blas_q8, cuda_q8, "f16-class", q8) == []
    assert G.check_identity(default_q8, cuda_q8, "f16-class", q8)
    # CPU default on Q8_0 quantizes activations: only against itself, with the activation_quantized numbers
    assert G.check_identity(blas_q8, default_q8, "f16-class", q8)
    assert G.check_identity(blas_q8, default_q8, "f16-class", q8_aq)
    assert G.check_identity(default_q8, default_q8, "f16-class", q8_aq) == []
    assert G.check_identity(default_q8, default_q8, "f16-class", q8)
    # a declared quantizing device needs the same CPU kernels as its baseline
    dev_aq = _ident(weights="q8_0", backend="cuda")
    dev_aq["runtime"]["activations"] = "quantized"
    assert G.check_identity(default_q8, dev_aq, "f16-class", q8_aq)  # kernels differ (cpu vs cuda)
    # F16 on the CPU default kernels rounds activations to F16: not a baseline; an F32 GGUF is one on any kernels
    assert G.check_identity(_ident(weights="f16", kernels="cpu"), _ident(weights="f16", backend="metal"), "f16-class", f16)
    f32 = G.tier_config(tiers, "f16-class", "f32")
    assert G.check_identity(_ident(weights="f32", kernels="cpu"), _ident(weights="f32", backend="vulkan"), "f16-class", f32) == []
