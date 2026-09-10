#!/usr/bin/env python3
"""Turn scripts/bench-amd.sh JSON output into one markdown table.

Usage:
    python3 scripts/bench-amd-report.py docs/amd/results/*.json [--baseline hip:f16]

Rows are (model, test, depth); columns are <label>/<kv>. The label comes from
the file name written by bench-amd.sh (<host>__<label>__<stamp>__<kv>.json).
With --baseline every cell also shows the ratio against that column.
"""

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

NAME_RE = re.compile(r"^(?P<host>.+?)__(?P<label>.+?)__(?P<stamp>\d{8}-\d{4})__(?P<kv>.+)\.json$")


def load(paths):
    cells = {}   # (model, test, depth) -> {column: t/s}
    columns = OrderedDict()
    for p in paths:
        m = NAME_RE.match(Path(p).name)
        if not m:
            print(f"skip {p}: name does not match <host>__<label>__<stamp>__<kv>.json", file=sys.stderr)
            continue
        col = f"{m['label']}/{m['kv']}"
        columns[col] = True
        try:
            rows = json.loads(Path(p).read_text() or "[]")
        except json.JSONDecodeError as e:
            print(f"skip {p}: {e}", file=sys.stderr)
            continue
        for r in rows:
            model = Path(r.get("model_filename", "?")).name
            test = f"pp{r['n_prompt']}" if r.get("n_prompt") else f"tg{r.get('n_gen', 0)}"
            key = (model, test, int(r.get("n_depth", 0)))
            cells.setdefault(key, {})[col] = (float(r["avg_ts"]), float(r.get("stddev_ts", 0.0)))
    return cells, list(columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--baseline", help="column to compare against, e.g. hip/f16")
    args = ap.parse_args()

    cells, columns = load(args.files)
    if not cells:
        sys.exit("no results")
    base = args.baseline
    if base and base not in columns:
        sys.exit(f"baseline column {base!r} not found; have {columns}")

    print("| model | test | depth | " + " | ".join(columns) + " |")
    print("|---|---|---:|" + "---:|" * len(columns))
    for (model, test, depth) in sorted(cells):
        row = cells[(model, test, depth)]
        out = []
        for c in columns:
            if c not in row:
                out.append("-")
                continue
            ts, sd = row[c]
            s = f"{ts:,.1f}"
            if base and base in row and c != base and row[base][0] > 0:
                s += f" ({ts / row[base][0] * 100:.0f}%)"
            out.append(s)
        print(f"| {model} | {test} | {depth} | " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
