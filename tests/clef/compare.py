#!/usr/bin/env python3
"""Compare server responses with the Clef reference.

  python tests/clef/compare.py ref.jsonl out.jsonl [--tokens] [--show N]

Per request: refused by one side only; with --tokens, the prompt tokens (answers[first].debug.tokens,
this fork with --decision-debug) against the reference. Per question: argmax agreement, max |dP| over
options, and max |dlogit| when the response has raw logits (debug.logits, in criteria key order).
"""

import argparse
import json
import math


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def answer_probs(ans):
    if "probabilities" in ans:
        return {str(k): v for k, v in ans["probabilities"].items()}
    if "noul" in ans:
        return {"true": ans["noul"], "false": 1.0 - ans["noul"]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    ref = {r["id"]: r for r in map(json.loads, open(args.ref, encoding="utf-8"))}
    out = {r["id"]: r for r in map(json.loads, open(args.out, encoding="utf-8"))}
    # a second reference file (another dtype or device) is compared as a server with raw logits
    for rid, o in list(out.items()):
        if "status" not in o:
            answers = {}
            for qid, q in o.get("questions", {}).items():
                keys = q["options"] if q["options"] != ["true", "false"] else ["false", "true"]
                lg = dict(zip(q["options"], q["logits"]))
                answers[qid] = {"probabilities": dict(zip(q["options"], softmax(q["logits"]))),
                                "debug": {"logits": [lg[k] for k in keys]}}
                if keys == ["false", "true"]:
                    answers[qid]["type"] = "noul"
            out[rid] = {"id": rid, "status": 400 if "error" in o else 200,
                        "body": {"answers": answers} if "error" not in o else o["error"]}

    n_req = n_q = n_agree = 0
    max_dp = max_dl = 0.0
    sum_dp = 0.0
    refused, tok_diff, worst = [], [], []
    for rid, r in ref.items():
        o = out.get(rid)
        if o is None:
            continue
        n_req += 1
        ok_ref, ok_out = "error" not in r, o["status"] == 200
        if ok_ref != ok_out:
            msg = r.get("error") if not ok_ref else json.dumps(o["body"], ensure_ascii=False)[:200]
            refused.append((rid, "reference" if not ok_ref else "server", msg))
            continue
        if not ok_ref:
            continue
        answers = o["body"]["answers"]
        if args.tokens:
            first = next(iter(answers.values()))
            toks = first.get("debug", {}).get("tokens")
            if toks != r["tokens"]:
                i = next((k for k, (a, b) in enumerate(zip(toks or [], r["tokens"])) if a != b), min(len(toks or []), len(r["tokens"])))
                tok_diff.append((rid, len(toks or []), len(r["tokens"]), i))
        for qid, rq in r["questions"].items():
            if "logits" not in rq:  # reference.py --tokens-only
                continue
            ans = answers[qid]
            pr = dict(zip(rq["options"], softmax(rq["logits"])))
            po = answer_probs(ans)
            n_q += 1
            dp = max(abs(pr[k] - po[k]) for k in pr)
            sum_dp += dp
            max_dp = max(max_dp, dp)
            arg_r = max(rq["options"], key=lambda k: pr[k])
            arg_o = max(po, key=lambda k: po[k])
            n_agree += arg_r == arg_o
            if "debug" in ans:
                keys = list(po.keys()) if ans.get("type") != "noul" else ["false", "true"]
                lo = dict(zip(keys, ans["debug"]["logits"]))
                lr = dict(zip(rq["options"], rq["logits"]))
                max_dl = max(max_dl, max(abs(lo[k] - lr[k]) for k in lr))
            worst.append((dp, rid, qid, arg_r, arg_o))

    print(f"requests {n_req}, questions {n_q}")
    print(f"refused by one side only: {len(refused)}")
    for x in refused[:args.show]:
        print("  ", x)
    if args.tokens:
        print(f"prompt token mismatches: {len(tok_diff)} (id, n_server, n_ref, first diff)")
        for x in tok_diff[:args.show]:
            print("  ", x)
    if n_q:
        print(f"argmax agreement {n_agree}/{n_q} = {n_agree / n_q:.4f}")
        print(f"|dP| max {max_dp:.4g}, mean of per-question max {sum_dp / n_q:.4g}")
        if max_dl:
            print(f"|dlogit| max {max_dl:.4g}")
        worst.sort(reverse=True)
        for x in worst[:args.show]:
            print("   dP %.4g %s/%s ref=%s out=%s" % x)


if __name__ == "__main__":
    main()
