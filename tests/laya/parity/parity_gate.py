"""Public-output gate for laya parity runs (Phase 4b; DECISION.md "Backend parity tiers").

Used by `tests/laya/verify_reference.py compare --gate <tier>` and tests/laya/parity/derive_tiers.py.
A run file is JSONL, one record per item: {"id", "per_question": {qid: {"input_ids", "raw_logits",
["marker_pos"], ["act_raw_logits"]}}, "answers": {qid: public answer}} or {"id", "error"}; a
reference file (verify_reference.py ref) has "state", "questions" and the public answers of
agent.system_one under "api".

The verdict is taken on the public outputs only: the answer key sets must be equal, categories
(`choice`, argmax of the scorer logits) must agree, and every number must be finite and within
the tier's tolerance, where an error that equals the tolerance up to float noise passes (a
4-digit reference answer of .8000 against .8001 passes at 1e-4, .8002 does not). strict-f32
(tiers v2) first rounds a candidate number with Python round(x, 4) when its reference number is
4-digit (the reference rounds, a server answer does not) and divides a score error by K - 1.
Raw-logit statistics stay diagnostic. Only questions with identical input ids are compared;
input mismatches, refusals, missing public answers and missing act-head logits fail, except
the kinds the tier lists as known intentional differences. The tiers and their numbers live in
tiers.json and are pre-registered in DECISION.md before the gates that use them.

compare_values() and its boundary rule are ported from laya.cpp benchmarks/compare.py:9-20
(https://github.com/lkarlslund/laya.cpp), MIT License, Copyright (c) 2026 Lars Karlslund (full text
also in licenses/LICENSE-laya.cpp):

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software
    and associated documentation files (the "Software"), to deal in the Software without
    restriction, including without limitation the rights to use, copy, modify, merge, publish,
    distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions: The above copyright
    notice and this permission notice shall be included in all copies or substantial portions of
    the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
    PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
    LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS_PATH = os.path.join(HERE, "tiers.json")

KNOWN_DIFFERENCES = {
    "null_state_refused": "the server refuses a null state (400 INVALID_REQUEST, param state); an older reference accepted it",
    "server_answer_shape": "server answers have no answer_confidence (and server builds before action.act_probability no action); the reference shape has both",
    "server_no_act_head": "a whole run without act-head logits (a server build before debug.act_logits): its act head is not compared",
    "list_state_truncation": "the state is a list and the inputs differ only in the state part, at the same length (the reference cuts list states from the left)",
}


def within(error, tolerance):
    return error <= tolerance or math.isclose(error, tolerance, rel_tol=1e-9, abs_tol=1e-12)


def compare_values(a, b, tolerance, path="result"):
    """Raise ValueError at the first key-set, category or number difference (port, see module doc)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError(f"{path}: keys differ")
        for key in a:
            compare_values(a[key], b[key], tolerance, f"{path}.{key}")
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        error = abs(a - b)
        if not math.isfinite(a) or not math.isfinite(b) or not within(error, tolerance):
            raise ValueError(f"{path}: {a} != {b} (tolerance {tolerance})")
    elif a != b:
        raise ValueError(f"{path}: {a!r} != {b!r}")


def load_tiers(path=None):
    with open(path or TIERS_PATH, encoding="utf-8") as f:
        return json.load(f)


def tier_config(tiers, tier, weights=None, activation_quantized=False):
    """Flat threshold dict for a tier; f16-class needs the GGUF weight type, and takes the
    activation_quantized numbers for a candidate that quantizes its activations."""
    if tier not in tiers["tiers"]:
        raise ValueError("unknown tier %r (have %s)" % (tier, ", ".join(tiers["tiers"])))
    t = dict(tiers["tiers"][tier])
    t["tiers_version"] = tiers.get("version", 1)
    aq = t.pop("activation_quantized", {})
    if "weights" in t:
        by_w = t.pop("weights")
        if activation_quantized:
            if weights not in aq:
                raise ValueError("tier %s has no activation_quantized numbers for %s weights" % (tier, weights))
            by_w = aq
        if weights not in by_w:
            raise ValueError("tier %s needs --weights, one of %s" % (tier, ", ".join(by_w)))
        t.update(by_w[weights])
        t["weights"] = weights
        t["activation_quantized"] = bool(activation_quantized)
    return t


def load_run(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d.get("id", d.get("key"))] = d
    return out


def public(rec):
    return rec.get("answers") if rec.get("answers") is not None else rec.get("api")


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def argmax(z):
    return max(range(len(z)), key=z.__getitem__)


def top2_gap(z):
    """p1 - p2 of softmax(z) in double: how close the first two options are (T = 1)."""
    if len(z) < 2:
        return 1.0
    p = sorted(softmax(z), reverse=True)
    return p[0] - p[1]


def rel_dlogit(a, b):
    """Largest |a - b| relative to the logit scale (the act head saturates at |logit| in the thousands)."""
    scale = max([abs(x) for x in a] + [abs(x) for x in b] + [1.0])
    return max(abs(x - y) for x, y in zip(a, b)) / scale


def all_finite(z):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in z)


def normalize_answers(ra, ca):
    """Copies without debug; a key of the reference answer shape that only one side has is dropped on both
    sides (a server answer has action but no answer_confidence; older server builds had neither), so
    action.act_probability is compared whenever both runs have it."""
    ra = {k: v for k, v in ra.items() if k != "debug"}
    ca = {k: v for k, v in ca.items() if k != "debug"}
    note = None
    for k in ("answer_confidence", "action"):
        if (k in ra) != (k in ca):
            ra.pop(k, None)
            ca.pop(k, None)
            note = "server_answer_shape"
    return ra, ca, note


def field_class(path):
    last = path[-1]
    if "probabilities" in path or last in ("noul", "act_probability"):
        return "prob"
    if last in ("confidence", "answer_confidence"):
        return "confidence"
    if last == "score":
        return "score"
    return "other"


def walk_numbers(a, b, path, out):
    """(path, a, b) for every number pair at the same place; key or type differences raise ValueError."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise ValueError("%s: keys differ (%s vs %s)" % (".".join(path) or "answer", sorted(a), sorted(b)))
        for k in a:
            walk_numbers(a[k], b[k], path + (k,), out)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))


def compare_public_f16(ra, ca, cfg, flipped):
    """f16-class public compare: returns (failures, per-class max error, sum |dp|, n dp)."""
    pairs = []
    try:
        walk_numbers(ra, ca, (), pairs)
    except ValueError as e:
        return [str(e)], {}, 0.0, 0
    n_opt = len(ra.get("probabilities") or {}) or 2
    fails, mx, dp_sum, dp_n = [], {}, 0.0, 0
    for path, a, b in pairs:
        p = ".".join(path)
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))) or isinstance(a, bool) or isinstance(b, bool):
            if path == ("choice",) and flipped:
                continue
            fails.append("%s: %r != %r" % (p, a, b))
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            fails.append("%s: not finite (%r, %r)" % (p, a, b))
            continue
        cls = field_class(path)
        err = abs(a - b)
        if cls == "prob":
            tol = cfg["max_abs_dp"]
            dp_sum += err
            dp_n += 1
        elif cls in ("confidence", "score"):
            # computed on the host from the probabilities by the same code: diagnostic only
            if cls == "score":
                mx["score_raw"] = max(mx.get("score_raw", 0.0), err)
                err = err / max(1, n_opt - 1)
            tol = math.inf
        else:
            tol = 0.0
        mx[cls] = max(mx.get(cls, 0.0), err)
        if not within(err, tol):
            fails.append("%s: %r vs %r, error %.3g > %.3g (%s)" % (p, a, b, err, tol, cls))
    return fails, mx, dp_sum, dp_n


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def is_4digit(x):
    """A number the reference may have rounded to 4 digits (Python round(x, 4) leaves it as it is)."""
    return math.isfinite(x) and round(x, 4) == x


def compare_public_strict(ra, ca, cfg):
    """strict-f32 public compare (tiers v2): returns (failures, per-class max error, raw max score error).

    A candidate number whose reference number is 4-digit is rounded like the reference (Python
    round(x, 4)); a score error is divided by K - 1 (the score range); everything at public_atol."""
    pairs = []
    try:
        walk_numbers(ra, ca, (), pairs)
    except ValueError as e:
        return [str(e)], {}, 0.0
    n_opt = len(ra.get("probabilities") or {}) or 2
    tol = cfg["public_atol"]
    fails, mx, score_raw = [], {}, 0.0
    for path, a, b in pairs:
        p = ".".join(path)
        if not (is_number(a) and is_number(b)):
            fails.append("%s: %r != %r" % (p, a, b))
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            fails.append("%s: not finite (%r, %r)" % (p, a, b))
            continue
        cls = field_class(path)
        err = abs(a - round(b, 4)) if is_4digit(a) else abs(a - b)
        if cls == "score":
            score_raw = max(score_raw, err)
            err = err / max(1, n_opt - 1)
        mx[cls] = max(mx.get(cls, 0.0), err)
        if not within(err, tol):
            fails.append("%s: %r vs %r, error %.3g > %.3g (%s%s)" % (p, a, b, err, tol, cls, ", / (K - 1)" if cls == "score" else ""))
    return fails, mx, score_raw


def public_errors(ra, ca):
    """Per-class max |error| between two answers (diagnostic; None if the shapes differ); score / (K - 1),
    and score_raw unscaled."""
    pairs = []
    try:
        walk_numbers(ra, ca, (), pairs)
    except ValueError:
        return None
    n_opt = len(ra.get("probabilities") or {}) or 2
    mx = {}
    for path, a, b in pairs:
        if is_number(a) and is_number(b):
            cls = field_class(path)
            err = abs(a - b) / (max(1, n_opt - 1) if cls == "score" else 1)
            mx[cls] = max(mx.get(cls, 0.0), err)
            if cls == "score":
                mx["score_raw"] = max(mx.get("score_raw", 0.0), abs(a - b))
    return mx


def sum_abs_dp(ra, ca):
    pairs = []
    try:
        walk_numbers(ra, ca, (), pairs)
    except ValueError:
        return 0.0, 0
    d = [abs(a - b) for path, a, b in pairs if field_class(path) == "prob" and isinstance(a, (int, float)) and isinstance(b, (int, float))
         and not isinstance(a, bool) and not isinstance(b, bool)]
    return sum(d), len(d)


def classify_input_mismatch(state, bq, cq):
    a, b = bq["input_ids"], cq["input_ids"]
    first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    markers = bq.get("marker_pos") or cq.get("marker_pos")
    if markers and first > max(markers):
        if isinstance(state, list) and len(a) == len(b):
            return "list_state_truncation", first
        return "state_tokens", first
    if markers:
        return "head_tokens", first
    return "input_ids", first


def n_questions(b, c):
    for r in (b, c):
        if r is not None and isinstance(r.get("questions"), dict):
            return len(r["questions"])
        if r is not None and isinstance(r.get("per_question"), dict):
            return len(r["per_question"])
    return 1


def gate_run(base, cand, tier, cfg, max_list=50):
    """Gate one candidate run against a baseline run (dicts id -> record). Returns a JSON-able result."""
    strict = tier == "strict-f32"
    known_ok = set(cfg.get("known_differences", []))
    res = {"tier": tier, "thresholds": cfg, "items": 0, "questions": 0, "compared": 0, "public_checked": 0,
           "public_missing": 0, "both_refused": 0, "argmax_agree": 0, "flips": [], "allowed_flips": [],
           "input_mismatches": {}, "known_differences": {}, "failures": [], "n_failures": 0,
           "max_abs_error": {}, "mean_abs_dp": 0.0, "act_checked": 0, "act_max_rel_dlogit": 0.0,
           "act_missing": {"baseline": 0, "candidate": 0}, "tiers_version": cfg.get("tiers_version", 1),
           "diagnostic": {"max_abs_dlogit": 0.0, "max_abs_dlogit_at": None, "mean_abs_dlogit": 0.0, "max_tvd": 0.0, "mean_tvd": 0.0,
                          "max_abs_dp_softmax": 0.0, "mean_abs_dp_softmax": 0.0}}
    dp_sum, dp_n, dl_sum, dl_n, tvd_sum = 0.0, 0, 0.0, 0, 0.0
    sdp_sum, sdp_n = 0.0, 0
    act_seen = {"baseline": 0, "candidate": 0}  # compared questions with act logits, per side

    def fail(msg):
        res["n_failures"] += 1
        if len(res["failures"]) < max_list:
            res["failures"].append(msg)

    def known(kind, n=1):
        res["known_differences"][kind] = res["known_differences"].get(kind, 0) + n
        return kind in known_ok

    ids = list(base) + [i for i in cand if i not in base]
    for iid in ids:
        b, c = base.get(iid), cand.get(iid)
        res["items"] += 1
        nq = n_questions(b, c)
        res["questions"] += nq
        if b is None or c is None:
            fail("%s: item only in the %s run" % (iid, "candidate" if b is None else "baseline"))
            continue
        be, ce = "error" in b, "error" in c
        if be and ce:
            res["both_refused"] += nq
            continue
        if be or ce:
            if ce and "state" in b and b["state"] is None:
                if not known("null_state_refused", nq):
                    fail("%s: candidate refuses a null state: %s" % (iid, c["error"][:200]))
                continue
            fail("%s: %s refuses: %s" % (iid, "candidate" if ce else "baseline", (c if ce else b)["error"][:200]))
            continue
        qids = list(b["questions"]) if isinstance(b.get("questions"), dict) else list(b["per_question"])
        pb_all, pc_all = public(b) or {}, public(c) or {}
        for qid in qids:
            bq, cq = b["per_question"].get(qid), c.get("per_question", {}).get(qid)
            where = "%s/%s" % (iid, qid)
            if bq is None or cq is None:
                fail("%s: question missing in the %s run" % (where, "baseline" if bq is None else "candidate"))
                continue
            if bq["input_ids"] != cq["input_ids"] or ("marker_pos" in bq and "marker_pos" in cq and bq["marker_pos"] != cq["marker_pos"]):
                kind, first = classify_input_mismatch(b.get("state"), bq, cq)
                res["input_mismatches"][kind] = res["input_mismatches"].get(kind, 0) + 1
                if kind in known_ok:
                    known(kind)
                else:
                    fail("%s: input ids differ at %d (%s)" % (where, first, kind))
                continue
            za, zb = bq["raw_logits"], cq["raw_logits"]
            if len(za) != len(zb) or not all_finite(za) or not all_finite(zb):
                fail("%s: raw logits not finite or of different length" % where)
                continue
            res["compared"] += 1
            d = [abs(x - y) for x, y in zip(za, zb)]
            dl_sum += sum(d)
            dl_n += len(d)
            if max(d) > res["diagnostic"]["max_abs_dlogit"]:
                res["diagnostic"]["max_abs_dlogit"], res["diagnostic"]["max_abs_dlogit_at"] = max(d), where
            sd = [abs(x - y) for x, y in zip(softmax(za), softmax(zb))]
            tvd = 0.5 * sum(sd)
            tvd_sum += tvd
            sdp_sum += sum(sd)
            sdp_n += len(sd)
            res["diagnostic"]["max_abs_dp_softmax"] = max(res["diagnostic"]["max_abs_dp_softmax"], max(sd))
            res["diagnostic"]["max_tvd"] = max(res["diagnostic"]["max_tvd"], tvd)

            flipped = argmax(za) != argmax(zb)
            if not flipped:
                res["argmax_agree"] += 1
            else:
                gap = top2_gap(za)
                entry = {"q": where, "baseline_gap": gap, "max_abs_dlogit": max(d)}
                res["flips"].append(entry)
                if strict:
                    fail("%s: argmax differs (baseline top-2 gap %.3g)" % (where, gap))
                elif gap < cfg["flip_max_gap"]:
                    res["allowed_flips"].append(entry)
                else:
                    fail("%s: argmax differs at baseline top-2 gap %.3g >= %.3g" % (where, gap, cfg["flip_max_gap"]))

            aa, ab = bq.get("act_raw_logits"), cq.get("act_raw_logits")
            act_seen["baseline"] += aa is not None
            act_seen["candidate"] += ab is not None
            if aa is not None and ab is not None:
                if not all_finite(aa) or not all_finite(ab) or len(aa) != len(ab):
                    fail("%s: act logits not finite or of different length" % where)
                else:
                    r = rel_dlogit(aa, ab)
                    res["act_checked"] += 1
                    res["act_max_rel_dlogit"] = max(res["act_max_rel_dlogit"], r)
                    if not within(r, cfg["act_max_rel_dlogit"]):
                        fail("%s: act head relative dlogit %.3g > %.3g" % (where, r, cfg["act_max_rel_dlogit"]))
            else:
                # decided after the loop: a whole run without act logits may be a known difference
                res["act_missing"]["baseline"] += aa is None
                res["act_missing"]["candidate"] += ab is None

            ra, ca = pb_all.get(qid), pc_all.get(qid)
            if ra is None or ca is None:
                res["public_missing"] += 1
                if not cfg.get("allow_public_missing"):
                    fail("%s: public answer missing in the %s run" % (where, "baseline" if ra is None else "candidate"))
                continue
            ra, ca, note = normalize_answers(ra, ca)
            if note and not known(note):
                fail("%s: answer shapes differ" % where)
                continue
            res["public_checked"] += 1
            if strict:
                fails, mx, score_raw = compare_public_strict(ra, ca, cfg)
                for f_ in fails:
                    fail("%s: %s" % (where, f_))
                if score_raw:
                    mx["score_raw"] = score_raw
                s, n = sum_abs_dp(ra, ca)
                dp_sum += s
                dp_n += n
            else:
                fails, mx, s, n = compare_public_f16(ra, ca, cfg, flipped)
                dp_sum += s
                dp_n += n
                for f_ in fails:
                    fail("%s: %s" % (where, f_))
            for k, v in mx.items():
                res["max_abs_error"][k] = max(res["max_abs_error"].get(k, 0.0), v)

    for side in ("baseline", "candidate"):
        n_miss = res["act_missing"][side]
        if not n_miss:
            continue
        if act_seen[side]:
            fail("act head logits missing on %d questions of the %s run, which has them on %d others" % (n_miss, side, act_seen[side]))
        elif not known("server_no_act_head", n_miss):
            fail("the %s run has no act head logits (%d questions unchecked)" % (side, n_miss))
    res["mean_abs_dp"] = dp_sum / dp_n if dp_n else 0.0
    res["diagnostic"]["mean_abs_dlogit"] = dl_sum / dl_n if dl_n else 0.0
    res["diagnostic"]["mean_tvd"] = tvd_sum / res["compared"] if res["compared"] else 0.0
    res["diagnostic"]["mean_abs_dp_softmax"] = sdp_sum / sdp_n if sdp_n else 0.0
    if not strict:
        frac = len(res["flips"]) / res["compared"] if res["compared"] else 0.0
        res["flip_fraction"] = frac
        if not within(frac, cfg["max_flip_fraction"]):
            fail("flip fraction %.4g > %.4g" % (frac, cfg["max_flip_fraction"]))
        if not within(res["mean_abs_dp"], cfg["mean_abs_dp"]):
            fail("mean |dp| %.3g > %.3g" % (res["mean_abs_dp"], cfg["mean_abs_dp"]))
    if res["compared"] == 0 or res["public_checked"] == 0:
        fail("nothing to gate: %d questions compared, %d public answers checked" % (res["compared"], res["public_checked"]))
    res["flips"] = res["flips"][:max_list]
    res["allowed_flips"] = res["allowed_flips"][:max_list]
    res["verdict"] = "pass" if res["n_failures"] == 0 else "fail"
    return res


def _kernels(rt):
    return rt.get("kernels_resolved") or rt.get("kernels")


def activation_class(ident):
    """How a run treats the activations of its weight matmuls: "f32", "f16" or "quantized"; None when the
    identity does not say (CPU kernels "auto" that were not resolved).

    A device declares quantized activations with runtime.activations; none does (the laya graph dequantizes
    Q8_0 on CUDA, Metal and Vulkan keep F32 / F16 activations), so a device run is "f32". On the CPU, BLAS
    converts the weights to F32 (F32 activations); the ggml kernels multiply in the weight's vec_dot type:
    F32 for F32 weights, F16 for F16, an 8-bit block format for the quantized types (default and repack)."""
    rt, m = ident.get("runtime") or {}, ident.get("model") or {}
    if rt.get("activations"):
        return rt["activations"]
    backend = rt.get("backend", "cpu")
    if backend == "torch-cpu" or backend != "cpu":
        return "f32"
    k = _kernels(rt) or ""
    if "blas" in k:
        return "f32"
    if k not in ("cpu", "default", "cpu+repack", "repack"):
        return None
    w = m.get("weights")
    if w in ("f32", "f16", "bf16"):
        return w
    return "quantized" if w else None


def check_identity(base_id, cand_id, tier, cfg):
    """Reasons why two runs may not be gated against each other under this tier (empty list: fine)."""
    why = []
    if base_id is None or cand_id is None:
        return ["no identity record for the %s run (<run>.identity.json)" % ("baseline" if base_id is None else "candidate")]
    bc, cc = base_id.get("corpus") or {}, cand_id.get("corpus") or {}
    if not bc.get("sha256") or bc.get("sha256") != cc.get("sha256"):
        why.append("corpus sha256 differs: %s vs %s" % (bc.get("sha256"), cc.get("sha256")))
    if bc.get("subset", "all") != cc.get("subset", "all"):
        why.append("corpus subset differs: %s vs %s" % (bc.get("subset"), cc.get("subset")))
    bm, cm = base_id.get("model") or {}, cand_id.get("model") or {}
    br, cr = base_id.get("runtime") or {}, cand_id.get("runtime") or {}
    if tier == "strict-f32":
        if br.get("backend") != "torch-cpu":
            why.append("the baseline must be the PyTorch reference (verify_reference.py ref), has backend %s" % br.get("backend"))
        if bm.get("revision") and cm.get("revision") and bm["revision"] != cm["revision"]:
            why.append("checkpoint revision differs: %s vs %s" % (bm["revision"], cm["revision"]))
        if not (bm.get("revision") and cm.get("revision")):
            why.append("checkpoint revision unknown on one side")
        if cm.get("weights") not in cfg.get("candidate_weights", ["f32"]):
            why.append("candidate weights %s are not in %s" % (cm.get("weights"), cfg.get("candidate_weights")))
        backend = cr.get("backend", "cpu")
        need = cfg.get("candidate_backends", {}).get(backend)
        if need is None:
            why.append("backend %s is not a strict-f32 backend" % backend)
        elif need != "any" and cr.get("precision", "default") != need:
            why.append("backend %s needs precision %s (run has %s)" % (backend, need, cr.get("precision", "default")))
    else:
        if not bm.get("sha256") or bm.get("sha256") != cm.get("sha256"):
            why.append("GGUF differs: %s vs %s" % (bm.get("sha256"), cm.get("sha256")))
        if br.get("backend", "cpu") != "cpu":
            why.append("the baseline must be a CPU run (has %s)" % br.get("backend"))
        if cfg.get("weights") and cm.get("weights") and cm["weights"] != cfg["weights"]:
            why.append("GGUF weights are %s, the gate was asked for %s" % (cm["weights"], cfg["weights"]))
        ba, ca = activation_class(base_id), activation_class(cand_id)
        if ca == "quantized":
            if ba != "quantized" or _kernels(br) != _kernels(cr):
                why.append("the candidate quantizes activations (%s %s): the baseline must quantize them the same way, has %s %s" % (
                    cr.get("backend", "cpu"), _kernels(cr), br.get("backend", "cpu"), _kernels(br)))
            if not cfg.get("activation_quantized"):
                why.append("the candidate quantizes activations: gate it with the activation_quantized numbers")
        else:
            if ba != "f32":
                why.append("the baseline must have F32 activations (kernels blas, or an F32 GGUF), has kernels %s (%s activations)" % (
                    _kernels(br), ba or "unknown"))
            if cfg.get("activation_quantized"):
                why.append("activation_quantized numbers, but the candidate does not quantize activations")
    return why
