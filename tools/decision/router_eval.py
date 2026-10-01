"""Router dataset, metrics, Platt scaling and bootstrap intervals (numpy + tools/decision/reference.py).

Shared by scripts/fit-router-calibration.py and scripts/router-baselines.py, so both report the
same numbers for the same data.

Dataset: JSONL, one record per task (the /v1/router/score request plus outcomes):

    {"id": "t-0001", "task": "...", "criterion": "...", "split": "fit" | "eval" (optional),
     "candidates": [{"id": "local/qwen3.5-4b-q4_k_m", "card": {...atomic.executor-card/1...},
                     "outcome": 1, "logit": 2.78 (engine raw z, optional)}],
     "engine": {...} (optional: identity of the engine that produced the logits)}

outcome is 1 when the executor met the criterion on the task, else 0. Every record must be a
valid router request (reference.parse_router), so the data cannot hold a card the engine would
refuse. One record is one cluster for the bootstrap: candidates of one task share the task.

Split: when every record has "split", "fit"/"train" records fit and "eval"/"heldout"/"test"
records evaluate. Otherwise a record goes to eval when sha256("<seed>:<id>") falls below
heldout_frac, so the split is stable across runs and machines.

Metrics of p = p_success against the 0/1 outcome:
    ece    10 equal-width bins of p, sum_b (n_b / n) * |mean(y_b) - mean(p_b)| (positive-class ECE;
           JevBench's top-label ECE on a binary question bins max(p, 1-p) instead)
    brier  mean (p - y)^2 (one-class; the 2-class convention of JevBench is twice this)
    nll    mean -log p(y), p clipped to [1e-12, 1 - 1e-12]
    auc    ROC AUC (ties count 1/2); rank quality only, calibration does not change it
    acc    mean [p >= 0.5] == y
Intervals are 95 % percentile intervals of a cluster bootstrap over records (multinomial weights,
seeded). A paired delta uses the same resamples for both predictors; an interval that contains
zero means no convincing difference.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference  # noqa: E402

FIT_SPLITS = ("fit", "train")
EVAL_SPLITS = ("eval", "heldout", "test")
EPS = 1e-12


class DataError(ValueError):
    pass


#
# dataset
#

def request_body(rec: dict) -> dict:
    """The /v1/router/score body of a record: outcomes, logits and bookkeeping removed."""
    body = {"task": rec.get("task"), "criterion": rec.get("criterion"),
            "candidates": [{"id": c.get("id"), "card": c.get("card")} if isinstance(c, dict) else c
                           for c in rec.get("candidates", [])] if isinstance(rec.get("candidates"), list) else rec.get("candidates"),
            "truncation": "allow"}
    return body


def load_records(path: str, need_logits: bool = False) -> list:
    """Read and check a dataset file. Raises DataError naming the line."""
    recs = []
    seen = set()
    with open(path, "rb") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            where = "%s:%d" % (path, n)
            try:
                rec = reference.loads_strict(raw)
            except reference.DecisionError as e:
                raise DataError("%s: %s" % (where, e)) from None
            if not isinstance(rec, dict):
                raise DataError(where + ": a record must be an object")
            rid = rec.get("id")
            if not isinstance(rid, str) or not rid:
                raise DataError(where + ": record needs a string id")
            if rid in seen:
                raise DataError("%s: duplicate record id %r" % (where, rid))
            seen.add(rid)
            try:
                reference.parse_router(request_body(rec))
            except reference.DecisionError as e:
                raise DataError("%s: not a valid router request: %s %s (%s)" % (where, e.reason, e.param, e)) from None
            for i, c in enumerate(rec["candidates"]):
                y = c.get("outcome")
                if isinstance(y, bool) or y not in (0, 1):
                    raise DataError("%s: candidates[%d].outcome must be 0 or 1" % (where, i))
                z = c.get("logit")
                if z is not None and (isinstance(z, bool) or not isinstance(z, (int, float)) or not math.isfinite(z)):
                    raise DataError("%s: candidates[%d].logit must be a finite number" % (where, i))
                if need_logits and z is None:
                    raise DataError("%s: candidates[%d] has no logit (run fit-router-calibration.py collect first)" % (where, i))
            split = rec.get("split")
            if split is not None and split not in FIT_SPLITS + EVAL_SPLITS:
                raise DataError("%s: split must be one of %s" % (where, ", ".join(FIT_SPLITS + EVAL_SPLITS)))
            recs.append(rec)
    if not recs:
        raise DataError(path + ": no records")
    return recs


def split_records(recs: list, heldout_frac: float = 0.3, seed: str = "router") -> tuple:
    """(fit, eval) record lists, see the module docstring."""
    if all("split" in r for r in recs):
        return [r for r in recs if r["split"] in FIT_SPLITS], [r for r in recs if r["split"] in EVAL_SPLITS]
    if any("split" in r for r in recs):
        raise DataError("some records have a split and some do not: give it to all or none")
    fit, ev = [], []
    for r in recs:
        h = int.from_bytes(hashlib.sha256(("%s:%s" % (seed, r["id"])).encode("utf-8")).digest()[:8], "big") / 2.0**64
        (ev if h < heldout_frac else fit).append(r)
    return fit, ev


def examples(recs: list) -> list:
    """Flat candidate examples: {"rid", "cid", "card", "y", "z", "task", "criterion"}."""
    out = []
    for r in recs:
        for c in r["candidates"]:
            out.append({"rid": r["id"], "cid": c["id"], "card": c["card"], "y": int(c["outcome"]), "z": c.get("logit"),
                        "task": r["task"], "criterion": r["criterion"]})
    return out


def engine_identity(recs: list) -> list:
    """Distinct "engine" blocks of the records (collect writes one per record). A record without
    one adds None, its own identity: logits of unknown origin never pass as the engine's."""
    seen = []
    for r in recs:
        e = r.get("engine")
        if e not in seen:
            seen.append(e)
    return seen


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#
# metrics
#

def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def _bins(p, bins=10):
    return np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(np.int64), bins - 1)


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    b = _bins(p, bins)
    sp = np.bincount(b, weights=p, minlength=bins)
    sy = np.bincount(b, weights=y, minlength=bins)
    return float(np.abs(sy - sp).sum() / max(len(p), 1))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def nll(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def auc(p, y) -> float | None:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(sp):  # average ranks of ties
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def metrics(p, y) -> dict:
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return {"n": int(len(y)), "pos_rate": float(y.mean()), "mean_p": float(p.mean()), "ece": ece(p, y), "brier": brier(p, y),
            "nll": nll(p, y), "auc": auc(p, y), "acc": float(np.mean((p >= 0.5) == (y == 1)))}


#
# cluster bootstrap
#

BOOT_METRICS = ("ece", "brier", "nll", "acc")


class Bootstrap:
    """Cluster bootstrap over records with seeded multinomial weights; every predictor sees the
    same resamples (weights are drawn again per call, in chunks, from the same seeds)."""

    CHUNK = 256

    def __init__(self, clusters: list, resamples: int = 2000, seed: int = 0, bins: int = 10):
        names = sorted(set(clusters))
        index = {c: i for i, c in enumerate(names)}
        self.cid = np.array([index[c] for c in clusters], dtype=np.int64)
        self.n_clusters = len(names)
        self.resamples = resamples
        self.seed = seed
        self.bins = bins

    def _sums(self, per_item):
        # (resamples, k) weighted sums of per-item columns, through per-cluster sums
        m = np.zeros((self.n_clusters, per_item.shape[1]))
        np.add.at(m, self.cid, per_item)
        out = []
        for k, start in enumerate(range(0, self.resamples, self.CHUNK)):
            rng = np.random.default_rng([self.seed, k])
            w = rng.multinomial(self.n_clusters, np.full(self.n_clusters, 1.0 / self.n_clusters),
                                size=min(self.CHUNK, self.resamples - start)).astype(np.float64)
            out.append(w @ m)
        return np.vstack(out)

    def stats(self, p, y) -> dict:
        """{metric: (resamples,) array}"""
        p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
        pc = np.clip(p, EPS, 1 - EPS)
        b = _bins(p, self.bins)
        onehot = np.zeros((len(p), self.bins))
        onehot[np.arange(len(p)), b] = 1.0
        cols = np.column_stack([np.ones(len(p)), (p - y) ** 2, -(y * np.log(pc) + (1 - y) * np.log1p(-pc)),
                                ((p >= 0.5) == (y == 1)).astype(np.float64), onehot * p[:, None], onehot * y[:, None]])
        s = self._sums(cols)
        n = s[:, 0]
        sp, sy = s[:, 4:4 + self.bins], s[:, 4 + self.bins:4 + 2 * self.bins]
        return {"brier": s[:, 1] / n, "nll": s[:, 2] / n, "acc": s[:, 3] / n, "ece": np.abs(sy - sp).sum(axis=1) / n}


def interval(samples) -> dict:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def report(p, y, boot: Bootstrap) -> dict:
    """Point metrics and bootstrap intervals of one predictor."""
    out = metrics(p, y)
    st = boot.stats(p, y)
    out["ci"] = {k: interval(st[k]) for k in BOOT_METRICS}
    return out


def paired_delta(p_new, p_ref, y, boot: Bootstrap) -> dict:
    """new - ref per metric with a paired bootstrap interval (negative is better for ece, brier, nll)."""
    a, b = boot.stats(p_new, y), boot.stats(p_ref, y)
    mn, mr = metrics(p_new, y), metrics(p_ref, y)
    out = {}
    for k in BOOT_METRICS:
        ci = interval(a[k] - b[k])
        out[k] = {"delta": mn[k] - mr[k], **ci, "clear": bool(ci["lo"] > 0 or ci["hi"] < 0)}
    return out


#
# fitting
#

def platt_fit(z, y, prior_targets: bool = True, max_iter: int = 100) -> tuple:
    """Platt scaling p = sigmoid(a * z + b) by Newton's method with backtracking (Lin, Lin and Weng
    2007). prior_targets uses Platt's targets (N+ + 1) / (N+ + 2) and 1 / (N- + 2) instead of 1 / 0,
    which keeps a and b finite on separable data. Returns (a, b, iterations)."""
    z, y = np.asarray(z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    t = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2)) if prior_targets else y
    a, b = 1.0, float(logit((n_pos + 1) / (len(y) + 2)))

    def loss(a, b):
        f = a * z + b
        # sum t * log(1 + e^-f) + (1 - t) * log(1 + e^f), stable
        return float(np.sum(t * np.logaddexp(0, -f) + (1 - t) * np.logaddexp(0, f)))

    cur = loss(a, b)
    for it in range(1, max_iter + 1):
        p = sigmoid(a * z + b)
        g = np.array([np.sum((p - t) * z), np.sum(p - t)])
        w = p * (1 - p)
        h = np.array([[np.sum(w * z * z) + 1e-12, np.sum(w * z)], [np.sum(w * z), np.sum(w) + 1e-12]])
        if np.max(np.abs(g)) < 1e-9 * max(1.0, len(y)):
            return float(a), float(b), it
        step = np.linalg.solve(h, g)
        s = 1.0
        while s > 1e-10:
            na, nb = a - s * step[0], b - s * step[1]
            nl = loss(na, nb)
            if nl < cur + 1e-4 * s * float(g @ -step):
                break
            s /= 2
        if s <= 1e-10:
            return float(a), float(b), it
        a, b, cur = na, nb, nl
    return float(a), float(b), max_iter


def logreg_fit(x, y, l2: float = 1.0, max_iter: int = 100) -> np.ndarray:
    """L2-regularized logistic regression by Newton's method (IRLS). x has an intercept column
    first (not penalized). Returns the weights."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w = np.zeros(x.shape[1])
    pen = np.full(x.shape[1], l2)
    pen[0] = 0.0
    for _ in range(max_iter):
        p = sigmoid(x @ w)
        g = x.T @ (p - y) + pen * w
        h = (x * (p * (1 - p))[:, None]).T @ x + np.diag(pen + 1e-9)
        step = np.linalg.solve(h, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def check_enough(y, min_examples: int, min_per_class: int, what: str) -> None:
    """Raise DataError when a fit would rest on too few examples."""
    y = np.asarray(y)
    n, n_pos = len(y), int(y.sum())
    if n < min_examples:
        raise DataError("refusing to fit %s on %d examples (need >= %d; --min-examples)" % (what, n, min_examples))
    if min(n_pos, n - n_pos) < min_per_class:
        raise DataError("refusing to fit %s: %d positive / %d negative examples (need >= %d of each; --min-per-class)"
                        % (what, n_pos, n - n_pos, min_per_class))


def fmt_ci(m: dict, key: str, digits: int = 4) -> str:
    ci = m.get("ci", {}).get(key)
    v = m[key]
    if v is None:
        return "n/a"
    f = "%%.%df" % digits
    return (f % v) + (" [" + (f % ci["lo"]) + ", " + (f % ci["hi"]) + "]" if ci else "")


def fmt_delta(d: dict, key: str, digits: int = 4) -> str:
    f = "%%+.%df" % digits
    x = d[key]
    return (f % x["delta"]) + " [" + (f % x["lo"]) + ", " + (f % x["hi"]) + "]" + ("" if x["clear"] else " (contains 0)")


def dump_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
