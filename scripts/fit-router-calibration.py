#!/usr/bin/env python3
"""Fit the router calibration (Platt a, b) on raw router logits of the engine.

Usage:
    # 1. raw logits of this engine, GGUF, plan and kernels (a server started as it will run, plus
    #    --decision-allow-uncalibrated when the GGUF has no router calibration yet)
    python3 scripts/fit-router-calibration.py collect data.jsonl logits.jsonl --url http://127.0.0.1:8090 [--spec spec.json]
    # 2. fit on the fit split, report ECE / Brier / NLL before and after on the eval split
    python3 scripts/fit-router-calibration.py fit logits.jsonl [--spec spec.json --spec-out spec-router.json] [-o report.json]
        [--min-examples 1000] [--min-per-class 50] [--heldout-frac 0.3] [--resamples 2000] [--before A,B]

Data format: tools/decision/router_eval.py (one record per task: task, criterion, candidates with
card and 0/1 outcome). collect sends each record to POST /v1/router/score and writes the record
back with every candidate's raw "logit" (z = s_true - s_false; the server reports the raw z with
or without a calibration) and an "engine" block from /props (model, spec_sha256, plan, kernels).

Why engine logits: the calibration maps this engine's z for this GGUF, recipe, plan and kernels
(DECISION.md, "Calibration rules"); logits from PyTorch or from another quantization would fit a
different function. fit refuses logits from more than one engine identity (a record without an
"engine" block counts as its own), and refuses to fit on fewer than --min-examples fit examples or
--min-per-class of either outcome. collect --spec accepts the file the server runs with
(--decision-spec) and the output of "gguf_decision_spec.py get model.gguf > spec.json" (the GGUF text
plus the one newline get adds).

"before" is what ships now: --before A,B, else the router.calibration of --spec, else the
uncalibrated zero-shot baseline p = sigmoid(z); "after" is sigmoid(a z + b). Both are scored on the
eval split, never on the data the fit saw, with cluster-bootstrap intervals over records
(router_eval.py); the delta row is paired. The fit-split numbers are printed too, marked in-sample.

Output: the spec block {"method": "platt", "a": ..., "b": ...} on stdout. GGUFs are never
changed: with --spec (the spec the model has now: gguf_decision_spec.py get model.gguf > spec.json)
and --spec-out, fit writes that spec with router.calibration replaced, checks it with the loader
rules of gguf_decision_spec.py, and prints the command that stamps it. It also sets plan.kernels
to the kernels the logits came from (/props plan.kernels cpu, cpu+blas, cpu+repack, cpu+repack+blas
-> default, blas, repack, repack+blas): "auto" is blas on macOS and default elsewhere, and the two
give different z, so a calibration fitted on one must not follow "auto" onto the other. The
criteria of the router question are part of the prompt, so the stamped calibration is valid only
for the question in the spec the logits were collected with.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "decision"))
sys.path.insert(0, os.path.join(ROOT, "gguf-py", "gguf", "scripts"))

import numpy as np  # noqa: E402

import router_eval as E  # noqa: E402

IDENTITY_KEYS = ("model_id", "model_version", "layout", "format", "spec_sha256", "device")
PLAN_KEYS = ("router", "kernels", "recipe")


def http_json(url: str, body: dict | None, api_key: str | None, timeout: float):
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            return res.status, json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, {"error": {"message": raw}}


def engine_block(props: dict) -> dict:
    d = props.get("decision") or {}
    e = {k: d.get(k) for k in IDENTITY_KEYS}
    plan = d.get("plan") or {}
    e["plan"] = {k: plan.get(k) for k in PLAN_KEYS}
    return e


def cmd_collect(a) -> int:
    recs = E.load_records(a.data)
    url = a.url.rstrip("/")
    st, props = http_json(url + "/props", None, a.api_key, a.timeout)
    if st != 200 or "decision" not in props:
        print("%s/props: status %d, no decision block: is this llama-server --decision?" % (url, st), flush=True)
        return 2
    engine = engine_block(props)
    if a.spec:
        with open(a.spec, "rb") as f:
            raw = f.read()
        # the sidecar's bytes, or the GGUF text that "gguf_decision_spec.py get > spec.json" wrote with one newline
        shas = {hashlib.sha256(raw).hexdigest()}
        for end in (b"\r\n", b"\n"):
            if raw.endswith(end):
                shas.add(hashlib.sha256(raw[:-len(end)]).hexdigest())
                break
        if engine["spec_sha256"] not in shas:
            print("--spec sha256 %s differs from the server's spec_sha256 %s" % (hashlib.sha256(raw).hexdigest(), engine["spec_sha256"]),
                  flush=True)
            return 2
    print("engine: %s" % json.dumps(engine, ensure_ascii=False), flush=True)
    n_ok = n_err = 0
    with open(a.out, "w", encoding="utf-8") as out:
        for i, rec in enumerate(recs):
            st, res = http_json(url + "/v1/router/score", E.request_body(rec), a.api_key, a.timeout)
            if st == 501:
                print("the server has no router calibration: restart it with --decision-allow-uncalibrated", flush=True)
                return 2
            if st != 200:
                err = res.get("error", {})
                print("record %s: %d %s %s" % (rec["id"], st, err.get("reason"), err.get("message")), flush=True)
                n_err += 1
                continue
            scores = {s["id"]: s for s in res["scores"]}
            rec = dict(rec)
            rec["candidates"] = [dict(c, logit=scores[c["id"]]["logit"], truncated_tokens=scores[c["id"]]["truncated_tokens"])
                                 for c in rec["candidates"]]
            rec["engine"] = engine
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            if (i + 1) % 500 == 0:
                print("  %d / %d records" % (i + 1, len(recs)), flush=True)
    print("wrote %s: %d records, %d refused by the server" % (a.out, n_ok, n_err), flush=True)
    return 0 if n_ok else 1


# /props.decision.plan.kernels (what a context computes with) -> spec plan.kernels (what to load)
PROPS_TO_SPEC_KERNELS = {"cpu": "default", "cpu+repack": "repack", "cpu+blas": "blas", "cpu+repack+blas": "repack+blas"}


def fitted_kernels(ids: list):
    """spec plan.kernels of the one engine identity the logits came from, else None."""
    if len(ids) != 1 or not isinstance(ids[0], dict):
        return None
    return PROPS_TO_SPEC_KERNELS.get((ids[0].get("plan") or {}).get("kernels"))


def load_spec_validator():
    try:
        from gguf_decision_spec import validate_spec  # stdlib-only part of the stamping tool
    except ImportError as e:
        raise SystemExit("cannot import gguf-py/gguf/scripts/gguf_decision_spec.py: %s" % e)
    return validate_spec


def cmd_fit(a) -> int:
    if a.spec_out and not a.spec:
        print("--spec-out needs --spec (the model's current spec: gguf_decision_spec.py get model.gguf > spec.json)", flush=True)
        return 2
    try:
        recs = E.load_records(a.data, need_logits=True)
        ids = E.engine_identity(recs)
        if len(ids) > 1 and not a.allow_mixed:
            raise E.DataError("logits come from %d engine identities (collect writes one per server; a record without an "
                              "engine block counts as its own); fit one at a time or pass --allow-mixed:\n  %s"
                              % (len(ids), "\n  ".join("(no engine block)" if i is None else json.dumps(i) for i in ids)))
        fit_recs, eval_recs = E.split_records(recs, a.heldout_frac, a.split_seed)
        fit_ex, eval_ex = E.examples(fit_recs), E.examples(eval_recs)
        y_fit = [e["y"] for e in fit_ex]
        E.check_enough(y_fit, a.min_examples, a.min_per_class, "the router calibration")
        if len(eval_ex) < a.min_eval:
            raise E.DataError("refusing to report on %d eval examples (need >= %d; --min-eval, --heldout-frac)" % (len(eval_ex), a.min_eval))
    except E.DataError as e:
        print(str(e), flush=True)
        return 2

    spec = None
    if a.spec:
        with open(a.spec, encoding="utf-8") as f:
            spec = json.load(f)
        if not isinstance(spec.get("router"), dict):
            print("%s has no router block: add one (card_schema, card_renderer, question) first" % a.spec, flush=True)
            return 2

    z_fit = np.array([e["z"] for e in fit_ex], dtype=np.float64)
    ca, cb, iters = E.platt_fit(z_fit, y_fit, prior_targets=not a.raw_targets)
    # "before" is what ships now: --before, else the spec's router.calibration, else sigmoid(z)
    deployed = spec["router"].get("calibration") if spec else None
    if a.before:
        before_a, before_b, before_src = a.before[0], a.before[1], "--before"
    elif isinstance(deployed, dict) and deployed.get("method") == "platt":
        before_a, before_b, before_src = float(deployed["a"]), float(deployed["b"]), "spec router.calibration"
    else:
        before_a, before_b, before_src = 1.0, 0.0, "uncalibrated sigmoid(z)"

    def pair(ex):
        z = np.array([e["z"] for e in ex], dtype=np.float64)
        y = np.array([e["y"] for e in ex], dtype=np.float64)
        boot = E.Bootstrap([e["rid"] for e in ex], a.resamples, a.seed)
        p0, p1 = E.sigmoid(before_a * z + before_b), E.sigmoid(ca * z + cb)
        return {"before": E.report(p0, y, boot), "after": E.report(p1, y, boot), "delta": E.paired_delta(p1, p0, y, boot)}

    res_eval, res_fit = pair(eval_ex), pair(fit_ex)
    block = {"method": "platt", "a": ca, "b": cb}
    report = {
        "tool": "scripts/fit-router-calibration.py",
        "data": os.path.abspath(a.data), "data_sha256": E.file_sha256(a.data),
        "engine": ids[0] if len(ids) == 1 else ids,
        "split": {"heldout_frac": a.heldout_frac, "seed": a.split_seed, "fit_records": len(fit_recs), "eval_records": len(eval_recs),
                  "fit_examples": len(fit_ex), "eval_examples": len(eval_ex)},
        "fit": {"prior_targets": not a.raw_targets, "iterations": iters, "min_examples": a.min_examples, "min_per_class": a.min_per_class},
        "before": {"a": before_a, "b": before_b, "source": before_src},
        "calibration": block,
        "bootstrap": {"resamples": a.resamples, "seed": a.seed, "cluster": "record"},
        "eval": res_eval,
        "fit_in_sample": res_fit,
    }

    print("fit: %d records / %d examples (%.3f positive); eval: %d records / %d examples"
          % (len(fit_recs), len(fit_ex), float(np.mean(y_fit)), len(eval_recs), len(eval_ex)), flush=True)
    print("platt: a = %.6f, b = %.6f (%d Newton steps); before = %s (a = %.6f, b = %.6f)"
          % (ca, cb, iters, before_src, before_a, before_b), flush=True)
    for name, r in (("eval", res_eval), ("fit (in-sample)", res_fit)):
        print("\n%s            before                    after                     delta (after - before)" % name, flush=True)
        for k in ("ece", "brier", "nll", "acc"):
            print("  %-6s %-25s %-25s %s" % (k, E.fmt_ci(r["before"], k), E.fmt_ci(r["after"], k), E.fmt_delta(r["delta"], k)), flush=True)
        print("  %-6s %-25s %-25s" % ("auc", E.fmt_ci(r["before"], "auc"), E.fmt_ci(r["after"], "auc")), flush=True)

    if spec is not None:
        validate_spec = load_spec_validator()
        spec["router"]["calibration"] = block
        # the calibration maps the z of these kernels (auto is blas on macOS, default elsewhere): pin them
        kernels = fitted_kernels(ids)
        if kernels is None:
            print("\nwarning: the logits do not name one kernels choice (no engine block from collect, or several): "
                  "plan.kernels left as it is (%s)" % json.dumps((spec.get("plan") or {}).get("kernels", "unset")), flush=True)
        else:
            if not isinstance(spec.get("plan"), dict):
                spec["plan"] = {}
            if spec["plan"].get("kernels") != kernels:
                print("\nplan.kernels: %s -> %s (the kernels the logits came from)"
                      % (json.dumps(spec["plan"].get("kernels", "unset")), kernels), flush=True)
            spec["plan"]["kernels"] = kernels
        report["plan_kernels"] = (spec.get("plan") or {}).get("kernels")
        try:
            validate_spec(spec)
        except ValueError as e:
            print("\nthe new spec fails the loader rules: %s" % e, flush=True)
            return 2
        report["spec_in"] = os.path.abspath(a.spec)
        if a.spec_out:
            # LF on every OS: gguf_decision_spec.py set stores these bytes, and spec_sha256 is their hash
            with open(a.spec_out, "w", encoding="utf-8", newline="\n") as f:
                json.dump(spec, f, ensure_ascii=False, indent=1)
                f.write("\n")
            report["spec_out"] = os.path.abspath(a.spec_out)
            report["stamp_command"] = ("python3 gguf-py/gguf/scripts/gguf_decision_spec.py set MODEL.gguf %s -o MODEL-router.gguf"
                                       % a.spec_out)
    if a.output:
        E.dump_json(report, a.output)
        print("\nreport: %s" % a.output, flush=True)

    print("\nrouter.calibration:", flush=True)
    print(json.dumps(block), flush=True)
    if a.spec_out:
        print("\nwrote %s (spec with this router.calibration). Stamp a copy of the GGUF with:" % a.spec_out, flush=True)
        print("  " + report["stamp_command"], flush=True)
        print("  python3 gguf-py/gguf/scripts/gguf_decision_spec.py verify MODEL-router.gguf %s" % a.spec_out, flush=True)
    else:
        print("\nTo stamp it: gguf_decision_spec.py get MODEL.gguf > spec.json, then rerun fit with --spec spec.json "
              "--spec-out spec-router.json (this tool never writes GGUFs).", flush=True)
    return 0


def parse_pair(s: str):
    try:
        a, b = (float(x) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected A,B") from None
    return a, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="raw router logits from a running llama-server --decision")
    c.add_argument("data")
    c.add_argument("out")
    c.add_argument("--url", default="http://127.0.0.1:8090")
    c.add_argument("--api-key", default=os.environ.get("LLAMA_API_KEY"))
    c.add_argument("--spec", help="the spec the server runs with: its sha256 must match /props")
    c.add_argument("--timeout", type=float, default=120.0)
    f = sub.add_parser("fit", help="fit Platt a, b and report calibration metrics")
    f.add_argument("data")
    f.add_argument("--spec", help="current spec of the model (JSON)")
    f.add_argument("--spec-out", help="write --spec with the new router.calibration here")
    f.add_argument("-o", "--output", help="report JSON")
    f.add_argument("--min-examples", type=int, default=1000, help="fit examples required (default 1000)")
    f.add_argument("--min-per-class", type=int, default=50, help="fit examples of each outcome required (default 50)")
    f.add_argument("--min-eval", type=int, default=200, help="eval examples required (default 200)")
    f.add_argument("--heldout-frac", type=float, default=0.3)
    f.add_argument("--split-seed", default="router")
    f.add_argument("--resamples", type=int, default=2000)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--before", type=parse_pair, help="calibration A,B to compare against (default: router.calibration of --spec, else 1,0: sigmoid(z))")
    f.add_argument("--raw-targets", action="store_true", help="fit on 0/1 targets instead of Platt's prior-corrected targets")
    f.add_argument("--allow-mixed", action="store_true", help="accept logits from more than one engine identity")
    a = ap.parse_args(argv)
    try:
        return cmd_collect(a) if a.cmd == "collect" else cmd_fit(a)
    except E.DataError as e:
        print(str(e), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
