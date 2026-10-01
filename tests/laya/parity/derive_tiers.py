#!/usr/bin/env python3
"""Derive the backend parity thresholds from CPU runs that exist before any GPU run.

    python3 tests/laya/parity/derive_tiers.py --p4 <dir> -o tiers-derivation.json [--check tests/laya/parity/tiers.json]
    python3 tests/laya/parity/derive_tiers.py --pair f16 NAME BASE.jsonl CAND.jsonl ... -o out.json      # other pairs

<dir> holds the Phase 4 parity outputs (the build-p4/parity tree: ml-ref.jsonl, en/ref.jsonl, td/ref.jsonl,
final/*.jsonl). Every pair is (baseline, candidate) on the same questions; the baseline is the more exact
run. For each pair the tool records, on the questions with identical inputs: the argmax flips and the
baseline top-2 probability gap (softmax of the raw scorer logits, T = 1) of each flip, the flip fraction,
the largest public-output error per field class (probabilities and noul; confidence; score / (K - 1)),
the mean |dp| over probability fields, and the act-head relative dlogit.

Pairs per class:
  f16     CPU F16 GGUF against CPU F32 (the F32 GGUF, or the PyTorch fp32 reference for laya-multilingual,
          whose F32 GGUF had no Phase 4 run), and CPU F16 `default` (ggml kernels: activations rounded to
          F16) against CPU F16 `blas` / `auto` (F32 sgemm) on the same GGUF: the rounding an F16-class
          backend adds on top of the CPU run of the same file
  q8_0    CPU Q8_0 `repack` against `default`: two kernel sets that both quantize the activations to Q8_0
          and differ in the order of the sums, as a device that quantizes activations (CUDA MMQ) differs
          from CPU `default`; plus the f16 pairs, for a device that keeps activations in F16 (Metal).
          Not used (reported as "info"): Q8_0 `default` against `auto` / `blas`, which mixes quantized and
          F32 activations; the baseline kernels must treat activations as the device does. Q8_0 against
          F32 is not used either: the weight rounding is shared by the CPU and the device run
  strict  CPU F32 GGUF against the PyTorch fp32 reference (English checkpoints; public answers and act)

Rule (DECISION.md, "Backend parity tiers"): threshold = 2 x the largest value seen over the pairs of the
class, rounded up to one significant digit. F32 GGUFs on an F16-class backend get the f16 numbers, and
since tiers v2 so do Q8_0 GGUFs on a device (no device quantizes activations); the q8_0 class numbers are
the f16-class activation_quantized row.
--check exits 1 when tiers.json does not hold the numbers this rule gives.
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity_gate as G  # noqa: E402

P4_PAIRS = [
    ("f16", "en F16 default vs F32, CLI", "final/cli_en_f32_default.jsonl", "final/cli_en_f16_default.jsonl"),
    ("f16", "en F16 blas vs F32, CLI", "final/cli_en_f32_default.jsonl", "final/cli_en_f16_blas.jsonl"),
    ("f16", "td F16 default vs F32, CLI", "final/cli_td_f32_default.jsonl", "final/cli_td_f16_default.jsonl"),
    ("f16", "td F16 blas vs F32, CLI", "final/cli_td_f32_default.jsonl", "final/cli_td_f16_blas.jsonl"),
    ("f16", "ml F16 default vs PyTorch fp32, server", "ml-ref.jsonl", "final/srv_ml_f16_default.jsonl"),
    ("f16", "ml F16 auto vs PyTorch fp32, server", "ml-ref.jsonl", "final/srv_ml_f16_auto.jsonl"),
    ("f16", "ml F16 default vs auto, server", "final/srv_ml_f16_auto.jsonl", "final/srv_ml_f16_default.jsonl"),
    ("f16", "en F16 default vs auto, server", "final/srv_en_f16_auto.jsonl", "final/srv_en_f16_default.jsonl"),
    ("f16", "td F16 default vs auto, server", "final/srv_td_f16_auto.jsonl", "final/srv_td_f16_default.jsonl"),
    ("f16", "en F16 default vs blas, CLI", "final/cli_en_f16_blas.jsonl", "final/cli_en_f16_default.jsonl"),
    ("f16", "td F16 default vs blas, CLI", "final/cli_td_f16_blas.jsonl", "final/cli_td_f16_default.jsonl"),
    ("q8_0", "ml Q8_0 repack vs default, CLI", "cli-p4/q8_0.jsonl", "cli-repack/q8_0.jsonl"),
    ("info", "ml Q8_0 default vs auto, server", "final/srv_ml_q8_0_auto.jsonl", "final/srv_ml_q8_0_default.jsonl"),
    ("info", "ml Q8_0 default vs blas, CLI", "cli-blas/q8_0.jsonl", "cli-p4/q8_0.jsonl"),
    ("strict", "en F32 vs PyTorch fp32, CLI", "en/ref.jsonl", "final/cli_en_f32_default.jsonl"),
    ("strict", "td F32 vs PyTorch fp32, CLI", "td/ref.jsonl", "final/cli_td_f32_default.jsonl"),
]

METRICS = ["flip_max_gap", "max_flip_fraction", "max_abs_dp", "mean_abs_dp", "act_max_rel_dlogit"]
# which classes feed which thresholds; "info" pairs are reported only
FEEDS = {"f16": ["f16"], "q8_0": ["q8_0", "f16"], "strict": ["strict"]}


def ceil1(x):
    """Round up to one significant digit (0.0092 -> 0.01, 0.21 -> 0.3, 8e-5 -> 8e-5)."""
    if x <= 0:
        return 0.0
    mant, exp = ("%.12e" % x).split("e")
    m = float(mant)
    up = math.ceil(m - 1e-9)
    if up == 10:
        up, exp = 1, str(int(exp) + 1)
    return float("%de%d" % (up, int(exp)))


def pair_stats(base_path, cand_path):
    permissive = {"flip_max_gap": math.inf, "max_flip_fraction": math.inf, "max_abs_dp": math.inf, "mean_abs_dp": math.inf,
                  "max_abs_dconfidence": math.inf, "max_abs_dscore_norm": math.inf, "act_max_rel_dlogit": math.inf,
                  "known_differences": list(G.KNOWN_DIFFERENCES), "allow_public_missing": True}
    r = G.gate_run(G.load_run(base_path), G.load_run(cand_path), "f16-class", permissive, max_list=100000)
    flips = r["flips"]
    return {
        "baseline": base_path, "candidate": cand_path,
        "compared": r["compared"], "public_checked": r["public_checked"], "act_checked": r["act_checked"],
        "input_mismatches": r["input_mismatches"], "refusals": r["n_failures"],
        "flips": len(flips), "flip_gaps": sorted(f["baseline_gap"] for f in flips),
        "flip_questions": [f["q"] for f in flips],
        "flip_max_gap": max((f["baseline_gap"] for f in flips), default=0.0),
        "max_flip_fraction": len(flips) / r["compared"] if r["compared"] else 0.0,
        # public answers when both runs have them; else softmax(raw logits), which is the public probability
        # vector of laya-multilingual (uncalibrated, T = 1), the only checkpoint whose pairs lack answers
        "dp_source": "public" if r["public_checked"] else "softmax(raw), T = 1",
        "max_abs_dp": r["max_abs_error"].get("prob") if r["public_checked"] else r["diagnostic"]["max_abs_dp_softmax"],
        "mean_abs_dp": r["mean_abs_dp"] if r["public_checked"] else r["diagnostic"]["mean_abs_dp_softmax"],
        "max_abs_dconfidence": r["max_abs_error"].get("confidence") if r["public_checked"] else None,
        "max_abs_dscore_norm": r["max_abs_error"].get("score") if r["public_checked"] else None,
        "act_max_rel_dlogit": r["act_max_rel_dlogit"] if r["act_checked"] else None,
        "max_abs_dlogit": r["diagnostic"]["max_abs_dlogit"], "max_tvd": r["diagnostic"]["max_tvd"],
    }


def derive(pairs):
    out = {"rule": "threshold = ceil to 1 significant digit of (2 x max over the pairs of the class)", "pairs": [], "classes": {}}
    for cls, name, b, c in pairs:
        for p in (b, c):
            if not os.path.isfile(p):
                sys.exit("missing: " + p)
        s = pair_stats(b, c)
        s["class"], s["name"] = cls, name
        out["pairs"].append(s)
        print("%-6s %-42s compared %5d flips %3d max gap %.4g max|dp| %s act rel %s" % (
            cls, name, s["compared"], s["flips"], s["flip_max_gap"],
            "-" if s["max_abs_dp"] is None else "%.3g" % s["max_abs_dp"],
            "-" if s["act_max_rel_dlogit"] is None else "%.3g" % s["act_max_rel_dlogit"]), flush=True)
    for cls, feeds in FEEDS.items():
        ps = [p for p in out["pairs"] if p["class"] in feeds]
        if not ps:
            continue
        obs, thr = {}, {}
        for m in (["act_max_rel_dlogit"] if cls == "strict" else METRICS):
            cand = [p for p in ps if p[m] is not None]
            if not cand:
                continue
            top = max(cand, key=lambda p: p[m])
            obs[m] = top[m]
            obs[m + "_from"] = top["name"]
            thr[m] = ceil1(2 * obs[m])
        if cls == "strict":
            obs["max_public_error"] = max(p["max_abs_dp"] or 0.0 for p in ps)
        out["classes"][cls] = {"pairs": [p["name"] for p in ps], "observed": obs, "threshold": thr}
    return out


def check(der, tiers_path):
    tiers = G.load_tiers(tiers_path)["tiers"]
    bad = []
    # tiers v2: a Q8_0 device run keeps F32 activations and takes the f16 numbers; the q8_0 class numbers
    # apply to a candidate that quantizes activations (f16-class activation_quantized)
    want = {"f16": tiers["f16-class"]["weights"]["f16"], "f32": tiers["f16-class"]["weights"]["f32"],
            "q8_0": tiers["f16-class"]["weights"]["q8_0"], "q8_0 activation_quantized": tiers["f16-class"]["activation_quantized"]["q8_0"]}
    for cls, t in (("f16", "f16"), ("f16", "f32"), ("f16", "q8_0"), ("q8_0", "q8_0 activation_quantized")):
        for m, v in der["classes"][cls]["threshold"].items():
            if want[t].get(m) != v:
                bad.append("f16-class/%s %s: tiers.json %s, derived %s" % (t, m, want[t].get(m), v))
    v = der["classes"]["strict"]["threshold"].get("act_max_rel_dlogit")
    if tiers["strict-f32"].get("act_max_rel_dlogit") != v:
        bad.append("strict-f32 act_max_rel_dlogit: tiers.json %s, derived %s" % (tiers["strict-f32"].get("act_max_rel_dlogit"), v))
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4", help="Phase 4 parity output folder")
    ap.add_argument("--pair", nargs=4, action="append", default=[], metavar=("CLASS", "NAME", "BASE", "CAND"))
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--check", help="tiers.json to check against the derived numbers")
    a = ap.parse_args()
    pairs = [(c, n, os.path.join(a.p4, b), os.path.join(a.p4, d)) for c, n, b, d in P4_PAIRS] if a.p4 else []
    pairs += [tuple(p) for p in a.pair]
    if not pairs:
        ap.error("give --p4 or --pair")
    der = derive(pairs)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(der, f, indent=1)
        f.write("\n")
    for cls, d in der["classes"].items():
        print("%s: %s" % (cls, json.dumps(d["threshold"])), flush=True)
    print("wrote " + a.out, flush=True)
    if a.check:
        bad = check(der, a.check)
        for b in bad:
            print("MISMATCH " + b, flush=True)
        if bad:
            sys.exit(1)
        print("tiers.json matches the derivation", flush=True)


if __name__ == "__main__":
    main()
