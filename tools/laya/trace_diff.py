#!/usr/bin/env python3
"""Compare two laya layer traces (LAYA_TRACE_DIR dumps) node by node.

    LAYA_TRACE_DIR=/tmp/tr-cpu   llama-laya-cli -m laya-f16.gguf -f in.json
    LAYA_TRACE_DIR=/tmp/tr-other llama-laya-cli -m laya-f16.gguf -f in.json --kernels blas
    python3 tools/laya/trace_diff.py /tmp/tr-cpu /tmp/tr-other [--call N] [--tol T] [--json out.json]

A trace directory holds trace.jsonl (one line per dumped node: call, node index, name, op, backend,
type, ne, file) and one raw little-endian float32 file per line. laya.cpp dumps only the nodes at
the ends of op chains: the residual output of every encoder layer (l_out-N), the encoder output
after the output norm (enc_out), the type-embedding sum (type_emb_out), the output of every head
layer (head_out-N), the marker rows (markers), the scorer logits (logits, logits_masked) and the act
head logits (act_logits). Nodes are matched by (call, name); call 0 is the first laya_encode of the
process (the warm-up pass of the server and the bench).

Per node the table has max |a - b|, that maximum relative to the scale of the node (max |a|, |b|
over the tensor), mean |a - b| and the number of differing values (a NaN or inf difference counts
as relative inf). The first node whose relative error is above --tol (default 0: any difference)
is where the two runs start to drift: bisect the backends there. Exit code 0 when every node matches within --tol, 1 when one does not, 2 on unusable input
(missing trace.jsonl, a node in only one trace, shapes that differ).
"""

import json
import math
import os
import sys
from array import array


def arg_value(args, name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


def load_index(d):
    path = os.path.join(d, "trace.jsonl")
    if not os.path.exists(path):
        sys.exit("no trace.jsonl in %s (was LAYA_TRACE_DIR set to an existing directory?)" % d)
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                e = json.loads(line)
                out[(e["call"], e["name"])] = e
    return out


def load_values(d, e):
    a = array("f")
    with open(os.path.join(d, e["file"]), "rb") as f:
        a.frombytes(f.read())
    if sys.byteorder != "little":
        a.byteswap()
    return a


def main():
    args = sys.argv[1:]
    call = arg_value(args, "--call", None)
    tol = float(arg_value(args, "--tol", "0"))
    json_out = arg_value(args, "--json", None)
    if len(args) != 2:
        sys.exit(__doc__)
    da, db = args
    ia, ib = load_index(da), load_index(db)
    keys = [k for k in ia if call is None or k[0] == int(call)]
    only = sorted(set(k for k in ia if call is None or k[0] == int(call)) ^ set(k for k in ib if call is None or k[0] == int(call)))
    if only:
        print("nodes in only one trace: %s" % ", ".join("%d/%s" % k for k in only[:20]), flush=True)
        sys.exit(2)
    keys.sort(key=lambda k: (k[0], ia[k]["node"]))
    rows, first_bad = [], None
    print("%5s %-16s %-10s %-18s %12s %12s %12s %10s" % ("call", "name", "op", "ne", "max|d|", "max rel", "mean|d|", "n diff"), flush=True)
    for k in keys:
        ea, eb = ia[k], ib[k]
        if ea["ne"] != eb["ne"]:
            print("%d/%s: shapes differ %s vs %s" % (k[0], k[1], ea["ne"], eb["ne"]), flush=True)
            sys.exit(2)
        a, b = load_values(da, ea), load_values(db, eb)
        if len(a) != len(b):
            print("%d/%s: sizes differ %d vs %d" % (k[0], k[1], len(a), len(b)), flush=True)
            sys.exit(2)
        mx = tot = scale = 0.0
        n_diff = 0
        nonfinite = False
        for x, y in zip(a, b):
            if math.isfinite(x) and math.isfinite(y):
                scale = max(scale, abs(x), abs(y))
            if x == y:
                continue
            n_diff += 1
            if not (math.isfinite(x) and math.isfinite(y)):
                nonfinite = True
                continue
            d = abs(x - y)
            tot += d
            mx = max(mx, d)
        rel = math.inf if nonfinite else mx / max(scale, 1e-30)
        row = {"call": k[0], "name": k[1], "op": ea.get("op"), "ne": ea["ne"], "max_abs": mx, "max_rel": rel,
               "mean_abs": tot / max(1, len(a)), "n_diff": n_diff, "n": len(a)}
        rows.append(row)
        if first_bad is None and rel > tol:
            first_bad = row
        print("%5d %-16s %-10s %-18s %12.4g %12.4g %12.4g %10d%s" % (
            k[0], k[1], str(ea.get("op")), "x".join(str(n) for n in ea["ne"]), mx, rel, row["mean_abs"], n_diff,
            "  <- first above --tol" if row is first_bad else ""), flush=True)
    print("%d nodes compared; %s" % (len(rows), "all within tol %g" % tol if first_bad is None else
                                     "first above tol %g: call %d %s" % (tol, first_bad["call"], first_bad["name"])), flush=True)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"a": os.path.abspath(da), "b": os.path.abspath(db), "tol": tol, "nodes": rows,
                       "first_above_tol": first_bad}, f, indent=1)
    sys.exit(0 if first_bad is None else 1)


if __name__ == "__main__":
    main()
