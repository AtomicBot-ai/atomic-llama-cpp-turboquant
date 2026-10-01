#!/usr/bin/env python3
"""Simple router baselines on the router data format, with the metrics of fit-router-calibration.py.

Usage:
    python3 scripts/router-baselines.py data.jsonl [-o report.json] [--reference NAME] [--l2 1.0]
        [--min-examples 1000] [--min-per-class 50] [--heldout-frac 0.3] [--resamples 2000]

Data: tools/decision/router_eval.py (records with task, criterion and candidates carrying card and
0/1 outcome; "logit" per candidate when the file comes from fit-router-calibration.py collect).
Every predictor that has parameters fits on the fit split; all are scored on the eval split, the
same split fit-router-calibration.py uses for the same --heldout-frac and --split-seed.

Predictors:
    base-rate        constant p = positive rate of the fit split (the floor)
    pass-rate        pooled card pass rate (sum passed + 1) / (sum total + 2) over measured checks;
                     a card without measured checks gets the fit base rate. No fitting.
    pass-rate-platt  Platt scaling of logit(pass-rate) on the fit split
    logreg           L2 logistic regression on card metrics (below), standardized on the fit split
    engine-raw       sigmoid(z), the zero-shot router (only when every example has a logit, all from
                     one engine identity as in fit-router-calibration.py)
    engine-platt     Platt scaling of z on the fit split: what fit-router-calibration.py would stamp

logreg features (card only; the task text is not used): logit of the pooled pass rate, log1p of
the summed totals, the measured and missing check counts / 32, has-measured, logit of the lowest
and highest per-check pass rate, kind == "local", kind == "cloud".

The table has point values with cluster-bootstrap 95 % intervals over records, and deltas against
--reference (default engine-platt when logits are present, else pass-rate-platt) on the same
resamples; an interval that contains zero means no convincing difference.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "decision"))

import reference  # noqa: E402
import router_eval as E  # noqa: E402

FEATURES = ("pooled_logit", "log_total", "n_measured", "n_missing", "has_measured", "min_logit", "max_logit", "kind_local",
            "kind_cloud")


def smoothed(passed: int, total: int) -> float:
    return (passed + 1.0) / (total + 2.0)


def card_stats(card: dict) -> dict:
    measured = [c for c in card.get("checks", []) if c["status"] == "measured"]
    missing = [c for c in card.get("checks", []) if c["status"] == "missing"]
    sp = sum(c["passed"] for c in measured)
    st = sum(c["total"] for c in measured)
    rates = [smoothed(c["passed"], c["total"]) for c in measured]
    kind = reference.canon_line(card["kind"]).lower()
    return {"measured": len(measured), "missing": len(missing), "passed": sp, "total": st, "rates": rates, "kind": kind}


def lg(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def pass_rate(ex: list, base: float) -> np.ndarray:
    out = []
    for e in ex:
        s = card_stats(e["card"])
        out.append(smoothed(s["passed"], s["total"]) if s["measured"] else base)
    return np.array(out)


def features(ex: list, base: float) -> np.ndarray:
    rows = []
    for e in ex:
        s = card_stats(e["card"])
        pooled = smoothed(s["passed"], s["total"]) if s["measured"] else base
        rows.append([lg(pooled), math.log1p(s["total"]), s["measured"] / 32.0, s["missing"] / 32.0, 1.0 if s["measured"] else 0.0,
                     lg(min(s["rates"])) if s["rates"] else lg(base), lg(max(s["rates"])) if s["rates"] else lg(base),
                     1.0 if s["kind"] == "local" else 0.0, 1.0 if s["kind"] == "cloud" else 0.0])
    return np.array(rows, dtype=np.float64)


def standardize(x_fit: np.ndarray, x: np.ndarray) -> np.ndarray:
    mu, sd = x_fit.mean(axis=0), x_fit.std(axis=0)
    sd[sd == 0] = 1.0
    z = (x - mu) / sd
    return np.column_stack([np.ones(len(z)), z])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("data")
    ap.add_argument("-o", "--output", help="report JSON")
    ap.add_argument("--reference", help="predictor the deltas are taken against")
    ap.add_argument("--l2", type=float, default=1.0, help="L2 penalty of logreg on standardized features (default 1.0)")
    ap.add_argument("--min-examples", type=int, default=1000)
    ap.add_argument("--min-per-class", type=int, default=50)
    ap.add_argument("--min-eval", type=int, default=200)
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--split-seed", default="router")
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    try:
        recs = E.load_records(a.data)
        fit_recs, eval_recs = E.split_records(recs, a.heldout_frac, a.split_seed)
        fit_ex, eval_ex = E.examples(fit_recs), E.examples(eval_recs)
        y_fit = np.array([e["y"] for e in fit_ex], dtype=np.float64)
        E.check_enough(y_fit, a.min_examples, a.min_per_class, "the baselines")
        if len(eval_ex) < a.min_eval:
            raise E.DataError("refusing to report on %d eval examples (need >= %d; --min-eval, --heldout-frac)" % (len(eval_ex), a.min_eval))
    except E.DataError as e:
        print(str(e), flush=True)
        return 2

    y = np.array([e["y"] for e in eval_ex], dtype=np.float64)
    base = float(y_fit.mean())
    preds = {}
    params = {}
    preds["base-rate"] = np.full(len(eval_ex), base)
    preds["pass-rate"] = pass_rate(eval_ex, base)
    pa, pb, _ = E.platt_fit(E.logit(pass_rate(fit_ex, base)), y_fit)
    preds["pass-rate-platt"] = E.sigmoid(pa * E.logit(preds["pass-rate"]) + pb)
    params["pass-rate-platt"] = {"a": pa, "b": pb}
    xf = features(fit_ex, base)
    w = E.logreg_fit(standardize(xf, xf), y_fit, a.l2)
    preds["logreg"] = E.sigmoid(standardize(xf, features(eval_ex, base)) @ w)
    params["logreg"] = {"l2": a.l2, "intercept": float(w[0]), "weights": dict(zip(FEATURES, (float(v) for v in w[1:])))}
    ids = E.engine_identity(recs)
    has_z = all(e["z"] is not None for e in fit_ex + eval_ex)
    if has_z and len(ids) > 1:
        print("logits come from %d engine identities (a record without an engine block counts as its own): "
              "engine-raw and engine-platt left out" % len(ids), flush=True)
    if has_z and len(ids) == 1:
        zf = np.array([e["z"] for e in fit_ex], dtype=np.float64)
        ze = np.array([e["z"] for e in eval_ex], dtype=np.float64)
        preds["engine-raw"] = E.sigmoid(ze)
        ea, eb, _ = E.platt_fit(zf, y_fit)
        preds["engine-platt"] = E.sigmoid(ea * ze + eb)
        params["engine-platt"] = {"a": ea, "b": eb}

    ref_name = a.reference or ("engine-platt" if "engine-platt" in preds else "pass-rate-platt")
    if ref_name not in preds:
        print("--reference %s is not one of %s" % (ref_name, ", ".join(preds)), flush=True)
        return 2
    boot = E.Bootstrap([e["rid"] for e in eval_ex], a.resamples, a.seed)
    results = {name: E.report(p, y, boot) for name, p in preds.items()}
    deltas = {name: E.paired_delta(p, preds[ref_name], y, boot) for name, p in preds.items() if name != ref_name}

    print("fit: %d records / %d examples (%.3f positive); eval: %d records / %d examples (%.3f positive)"
          % (len(fit_recs), len(fit_ex), base, len(eval_recs), len(eval_ex), float(y.mean())), flush=True)
    print("\n%-16s %-25s %-25s %-25s %-8s" % ("predictor", "ece", "brier", "nll", "auc"), flush=True)
    for name, r in results.items():
        auc = "n/a" if r["auc"] is None else "%.4f" % r["auc"]
        print("%-16s %-25s %-25s %-25s %-8s" % (name, E.fmt_ci(r, "ece"), E.fmt_ci(r, "brier"), E.fmt_ci(r, "nll"), auc), flush=True)
    print("\ndelta vs %s (negative is better)" % ref_name, flush=True)
    for name, d in deltas.items():
        print("%-16s ece %s  brier %s  nll %s" % (name, E.fmt_delta(d, "ece"), E.fmt_delta(d, "brier"), E.fmt_delta(d, "nll")), flush=True)

    if a.output:
        E.dump_json({
            "tool": "scripts/router-baselines.py",
            "data": os.path.abspath(a.data), "data_sha256": E.file_sha256(a.data),
            "engine": ids,
            "split": {"heldout_frac": a.heldout_frac, "seed": a.split_seed, "fit_records": len(fit_recs), "eval_records": len(eval_recs),
                      "fit_examples": len(fit_ex), "eval_examples": len(eval_ex)},
            "bootstrap": {"resamples": a.resamples, "seed": a.seed, "cluster": "record"},
            "reference": ref_name, "params": params, "eval": results, "delta_vs_reference": deltas,
        }, a.output)
        print("\nreport: %s" % a.output, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
