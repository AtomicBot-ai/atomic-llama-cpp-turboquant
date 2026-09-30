#!/usr/bin/env python3
"""Turn scripts/bench-decision.sh results into markdown tables (KPIs K1, K4-K7, K9).

Usage:
    python3 scripts/bench-decision-report.py build/bench-decision [more dirs or files ...] [--stamp 20260930-0215]
        [--family laya-f16=laya-multilingual ...]

Reads <host>__<label>__<stamp>__machine.txt, llama-decision-bench JSON
(<...>__<model>__t<N>.json) and server JSON (<...>__<model>__t<N>__server.json).
Columns are <model> t<N>; with several stamps in one folder, --stamp picks one engine run
(default: the newest one with engine results). Latencies are milliseconds; p50/p95/p99
interpolate linearly.

PyTorch reference results (scripts/bench-decision-pytorch.py bench, "bench": "laya-pytorch-bench")
found in the same paths add "Engine vs PyTorch": one table per request group and thread count,
engine models against PyTorch fp32/bf16 of the same checkpoint family (newest result per family,
dtype, router mode and threads). speedup = PyTorch fp32 p50 / row p50. The family of an engine
model comes from its file name (laya-en-* -> laya, laya-td-* -> laya-typed-decisions, other
laya-* -> laya-multilingual); --family STEM=FAMILY overrides it. A PyTorch result whose
equal-input check did not pass is marked UNCHECKED: its inputs may differ from the engine's.
PyTorch results from another host than the engine run are left out of those tables.

--summary [--model-sizes ls-l.txt] reports every run in the paths instead of one stamp (a thread
sweep writes one stamp per block): the equal-input / fidelity checks found there (check JSON of
bench-decision-pytorch.py, any file name), then per checkpoint family and thread count one p50
table per request group (every engine run and PyTorch result as a column, with its stamp and load)
and one per-request speedup table with the geometric mean. The group p50 pools requests of
different lengths (a tail-sensitive statistic), so the per-request ratios and their geometric mean
are the numbers to quote. With a PyTorch --router-mode batch result, "geomean vs best PyTorch fp32"
takes the faster PyTorch mode per request. Then startup / memory (RSS and anon, file size from the
`ls -l` listing) and the first request after idle (one sample each).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^(?P<host>.+?)__(?P<label>.+?)__(?P<stamp>\d{8}-\d{4})__(?P<rest>.+)$")


def collect(paths):
    files = []
    for p in paths:
        p = Path(p)
        files += sorted(p.iterdir()) if p.is_dir() else [p]
    runs = {}  # stamp -> {"machine": dict, "bench": [...], "server": [...]}
    for f in files:
        m = NAME_RE.match(f.name)
        if not m:
            continue
        run = runs.setdefault(m["stamp"], {"machine": {}, "bench": [], "server": [], "pytorch": [], "host": m["host"], "label": m["label"]})
        if m["rest"] == "machine.txt":
            run["machine"] = read_facts(f)
        elif f.suffix == ".json":
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            if d.get("kind") == "server":
                run["server"].append(d)
            elif d.get("bench") == "llama-decision-bench":
                run["bench"].append(d)
            elif d.get("bench") == "laya-pytorch-bench":
                d["_stamp"] = m["stamp"]
                run["pytorch"].append(d)
    return runs


def read_facts(path):
    facts, key = {}, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("  ") and key:
            facts[key] = (facts[key] + "\n" + line.strip()).strip()
            continue
        k, sep, v = line.partition(":")
        if sep:
            key = k.strip()
            facts[key] = v.strip()
    return facts


def col(d):
    # kernels other than the published ones are part of the column name
    k = d.get("kernels")
    return f"{Path(d['model']).stem} t{d['n_threads']}" + (f" {k}" if k and k != "cpu" else "")


def mib(b):
    return "-" if b is None or b < 0 else f"{b / 1048576:.0f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(v, p):
    if not v:
        return None
    v = sorted(v)
    pos = p / 100 * (len(v) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def report(run):
    bench = sorted(run["bench"], key=lambda d: (Path(d["model"]).stem, d["n_threads"]))
    server = sorted(run["server"], key=lambda d: (Path(d["model"]).stem, d["n_threads"]))
    fx = run["machine"]
    out = [f"## Decision bench {run['host']} ({run['label']})", ""]

    keys = ["date_utc", "cpu", "cores", "memory_gb", "os", "power_source", "power_source_end", "low_power_mode",
            "battery", "loadavg_start", "loadavg_end", "git", "bin_dir", "suite", "repeat"]
    out += [table(["fact", "value"], [(k, fx[k].replace("\n", "; ")) for k in keys if k in fx]), ""]
    for k in ("top_cpu_start", "top_cpu_end"):
        if k in fx:
            out += [f"{k}: " + fx[k].replace("\n", "; "), ""]
    runs_load = [f"{k}: {v}" for k, v in fx.items() if k.startswith("run ") or k.startswith("server ")]
    if runs_load:
        out += ["Load and power before each run:", ""] + ["- " + r for r in runs_load] + [""]

    if bench:
        cols = [col(d) for d in bench]
        groups = []
        for d in bench:
            for g in d["groups"]:
                if g["group"] not in groups:
                    groups.append(g["group"])

        def gstat(d, g, field, stat):
            for x in d["groups"]:
                if x["group"] == g:
                    return x[field][stat]
            return None

        out += ["### K1 latency by group (in-process, ms, p50 / p95 / p99 over requests x repeats)", ""]
        rows = []
        for g in groups:
            row = [g]
            for d in bench:
                s = [gstat(d, g, "total_ms", k) for k in ("p50", "p95", "p99")]
                row.append("-" if s[0] is None else " / ".join(f"{x:.0f}" for x in s))
            rows.append(row)
        out += [table(["group"] + cols, rows), ""]

        out += ["### Per request (p50 ms; ms per 1k tokens is compute p50 over prompt tokens)", ""]
        for d in bench:
            out += [f"{col(d)}: load {d['load']['ms']:.0f} ms, first request {d['load']['first_request_ms']:.0f} ms, "
                    f"plan {d['plan']} / router {d['plan_independent']}", ""]
            rows = []
            for r in d["requests"]:
                tot = r["total_ms"]["p50"]
                rows.append([r["id"], r["n_items"], r["n_tokens"], f"{r['parse_ms']['p50']:.3f}", f"{r['tokenize_ms']['p50']:.3f}",
                             f"{r['render_ms']['p50']:.3f}", f"{r['compute_ms']['p50']:.1f}", f"{r['post_ms']['p50']:.3f}",
                             f"{tot:.1f}", f"{r['total_ms']['p95']:.1f}", f"{r['total_ms']['p99']:.1f}",
                             f"{1000 * r['compute_ms']['p50'] / max(1, r['n_tokens']):.0f}",
                             f"{r['cpu_s']['mean']:.3f}", "yes" if r["deterministic"] else "NO"])
            out += [table(["request", "items", "tokens", "parse", "tokenize", "render", "compute", "post", "total p50", "p95", "p99",
                           "ms/1k tok", "cpu_s", "det"], rows), ""]

        out += ["### K4 memory (MiB = 2^20 bytes)", "",
                "footprint: macOS phys_footprint (does not count clean file-backed pages such as the mapped "
                "token_embd), Linux RssAnon, Windows private bytes. rss counts the resident mapped pages too. "
                "Server rows: 'end' is one sample after the requests and the idle wait, not a steady state.", ""]
        rows = []
        for d in bench:
            mem = d["memory"]
            rows.append([col(d), mib(Path(d["model"]).stat().st_size) if Path(d["model"]).exists() else "-",
                         mib(mem["after_load"]["rss"]), mib(mem["end"]["rss_peak"]),
                         mib(mem["after_load"]["footprint"]), mib(mem.get("after_warmup", {}).get("footprint")),
                         mib(mem["end"]["footprint_peak"])])
        for s in server:
            rows.append([col(s) + " server", "-", f"{s['rss_kb_ready'] / 1024:.0f}", f"{s['rss_kb_end'] / 1024:.0f} (end)",
                         f"{s['footprint_kb_ready'] / 1024:.0f}", "-", f"{s['footprint_kb_end'] / 1024:.0f} (end)"])
        out += [table(["config", "file", "rss after load", "rss peak", "footprint after load", "footprint after warmup",
                       "footprint peak"], rows), ""]

        out += ["### K7 determinism (SHA-256 of raw logits)", ""]
        rows = []
        by_model = {}
        for d in bench:
            by_model.setdefault(Path(d["model"]).stem, []).append(d)
            n_bad = sum(1 for r in d["requests"] if not r["deterministic"])
            idle = " + after idle" if d.get("after_idle") else ""
            rows.append([col(d), f"{d['warmup']} + {d['repeat']}{idle}", "yes" if d["deterministic"] else f"NO ({n_bad} requests)",
                         d["suite_hash"][:16]])
        out += [table(["config", "runs", "bitwise identical", "suite hash"], rows), ""]
        for model, ds in by_model.items():
            if len(ds) > 1:
                same = len({d["suite_hash"] for d in ds}) == 1
                diff = []
                if not same:
                    ref = {r["id"]: r["hash"] for r in ds[0]["requests"]}
                    for d in ds[1:]:
                        diff += [f"{r['id']} (t{d['n_threads']})" for r in d["requests"] if ref.get(r["id"]) != r["hash"]]
                out += [f"- {model}: threads {', '.join(str(d['n_threads']) for d in ds)} -> "
                        + ("identical logits" if same else "DIFFERENT logits: " + ", ".join(diff))]
        out += [""]

        out += ["### K9 CPU seconds per request (process user + system, mean; ratio = cpu / wall)", ""]
        rows = []
        for g in groups:
            row = [g]
            for d in bench:
                c, w = gstat(d, g, "cpu_s", "mean"), gstat(d, g, "total_ms", "mean")
                row.append("-" if c is None else f"{c:.3f} ({1000 * c / w:.1f}x)")
            rows.append(row)
        out += [table(["group"] + cols, rows), ""]

    idle_rows = []
    for d in bench:
        ai = d.get("after_idle")
        if ai:
            for r in ai["runs"][:1]:
                idle_rows.append([col(d) + " in-process", ai["idle_ms"] // 1000, r["id"], f"{r['total_ms']:.1f}", f"{r['p50_ms']:.1f}",
                                  f"{r['total_ms'] / r['p50_ms']:.2f}x"])
    for s in server:
        for key, name, warm in (("idle_so1", "so-1q-noul", "warm_so1_ms"), ("idle_rt4", "rt-n4-meeting", "warm_rt4_ms")):
            r = s.get(key)
            if r:
                p50 = pct(s[warm], 50)
                idle_rows.append([col(s) + " HTTP", s["idle_s"], name, f"{r['total_ms']:.1f}", f"{p50:.1f}", f"{r['total_ms'] / p50:.2f}x"])
    if server:
        out += ["### K5 time to ready (llama-server --decision, spawn -> /health 200; 10 ms polling)", ""]
        rows = [[col(s), f"{s['listen_ms']:.0f}" if s.get("listen_ms") is not None else "-", f"{s['ready_ms']:.0f}",
                 f"{s['first_so1']['total_ms']:.1f}" if s.get("first_so1") else "-", f"{pct(s['warm_so1_ms'], 50):.1f}",
                 f"{pct(s['warm_rt4_ms'], 50):.1f}"] for s in server]
        out += [table(["config", "HTTP listening", "ready", "first so-1q (HTTP)", "warm so-1q p50", "warm rt-n4 p50"], rows), ""]
    if idle_rows:
        out += ["### K6 first request after idle (ms)", ""]
        out += [table(["config", "idle s", "request", "after idle", "warm p50", "ratio"], idle_rows), ""]
    return "\n".join(out)


def engine_family(model, overrides):
    stem = Path(model).stem
    if stem in overrides:
        return overrides[stem]
    if re.search(r"(^|[-_])en([-_]|$)", stem):
        return "laya"
    if re.search(r"(^|[-_])td([-_]|$)", stem) or "typed" in stem:
        return "laya-typed-decisions"
    return "laya-multilingual"


def pytorch_col(d):
    return f"PyTorch {d['family']} {d['dtype']}" + (" router-batch" if d.get("router_mode") == "batch" else "")


def compare_report(run, pytorch, overrides):
    """Engine vs PyTorch reference: one table per request group and thread count."""
    # newest PyTorch result per (family, dtype, router mode, threads)
    newest = {}
    for d in sorted(pytorch, key=lambda d: d["_stamp"]):
        newest[(d["family"], d["dtype"], d.get("router_mode", "sequential"), d["n_threads"])] = d
    pts = sorted(newest.values(), key=lambda d: (d["family"], d["n_threads"], d["dtype"] != "fp32", d.get("router_mode", "")))
    host = (run or {}).get("host")
    other = [d for d in pts if host and d.get("machine", {}).get("host") not in (None, host)]
    pts = [d for d in pts if d not in other]
    bench = sorted(run["bench"] if run else [], key=lambda d: (engine_family(d["model"], overrides), Path(d["model"]).stem))
    server = run["server"] if run else []
    out = ["## Engine vs PyTorch reference (same requests)", ""]
    if other:
        out += [f"Left out: {len(other)} PyTorch result(s) from another host than the engine run ({host}).", ""]

    rows = []
    for d in pts:
        eq = d.get("equal_input") or {}
        rt = d.get("runtime", {})
        mf = d.get("machine", {})
        state = (f"passed ({eq.get('n_items')} items, argmax {eq.get('argmax_same')}, max \\|dlogit\\| {eq.get('max_abs_dlogit'):.1e}, "
                 f"engine {Path(eq['engine']['model']).stem} {eq['engine'].get('kernels')})") if eq.get("passed") else "UNCHECKED"
        eff = d.get("dtype_effective", d["dtype"])
        dt = d["dtype"] + ("" if d["dtype"] == "fp32" or d.get("autocast_end") else f" (fell back to {eff})")
        rows.append([f"{pytorch_col(d)} t{d['n_threads']}", dt, state, f"torch {rt.get('torch')} / laya {rt.get('laya')}",
                     f"{d['warmup']} + {d['repeat']}", mf.get("power_source", "-"),
                     f"{mf.get('loadavg_start', '-')} -> {mf.get('loadavg_end', '-')}", d["_stamp"]])
    out += [table(["reference", "dtype", "equal-input check", "versions", "warmup + repeat", "power", "load avg", "stamp"], rows), ""]
    if pts and pts[0].get("machine", {}).get("cpu"):
        out += [f"PyTorch machine: {pts[0]['machine']['cpu']}, {pts[0]['machine'].get('cores', '')}; "
                "BLAS / backends: see runtime.config in the PyTorch JSON.", ""]
    if not bench:
        out += ["No llama-decision-bench results in the selected engine run: PyTorch rows only.", ""]

    def gget(d, g):
        return next((x for x in d["groups"] if x["group"] == g), None)

    def srv(model, t):
        return next((s for s in server if Path(s["model"]).stem == Path(model).stem and s["n_threads"] == t), None)

    def engine_load(d):
        v = (run or {}).get("machine", {}).get(f"run {Path(d['model']).stem} t{d['n_threads']}", "")
        m = re.search(r"loadavg ([\d.]+)", v)
        return m.group(1) if m else "-"

    def pt_load(d):
        v = d.get("machine", {}).get("loadavg_start", "")
        return v.split()[0] if v else "-"

    threads = sorted({d["n_threads"] for d in bench} | {d["n_threads"] for d in pts})
    groups = []
    for d in bench + pts:
        for g in d["groups"]:
            if g["group"] not in groups:
                groups.append(g["group"])
    out += ["speedup = PyTorch fp32 p50 (same family, threads, group) / row p50. cpu_s: process user + system per request, "
            "mean. RSS MiB after load / peak. ready: spawn -> model loaded; first: spawn -> first answer (so-1q-noul; engine "
            "over HTTP from llama-server --decision, PyTorch in process from the Python start). Engine load is in-process "
            "(llama-decision-bench, no process start). loadavg: 1-minute load average when the run started; rows "
            "measured at different loads do not compare.", ""]
    for t in threads:
        for g in groups:
            ref = {d["family"]: gget(d, g) for d in pts if d["n_threads"] == t and d["dtype"] == "fp32"
                   and d.get("router_mode", "sequential") == "sequential" and gget(d, g)}
            rows = []
            for d in bench:
                if d["n_threads"] != t or not gget(d, g):
                    continue
                x = gget(d, g)
                fam = engine_family(d["model"], overrides)
                r = ref.get(fam)
                s = srv(d["model"], t)
                mem = d["memory"]
                rows.append([f"engine {Path(d['model']).stem}" + (f" {d['kernels']}" if d.get("kernels") and d["kernels"] != "cpu" else ""), fam,
                             f"{x['total_ms']['p50']:.1f}", f"{x['total_ms']['p95']:.1f}",
                             f"{r['total_ms']['p50'] / x['total_ms']['p50']:.2f}x" if r else "-", f"{x['cpu_s']['mean']:.3f}",
                             f"{mib(mem['after_load']['rss'])} / {mib(mem['end']['rss_peak'])}", f"{d['load']['ms']:.0f}",
                             f"{s['ready_ms']:.0f}" if s else "-",
                             f"{s['ready_ms'] + s['first_so1']['total_ms']:.0f}" if s and s.get("first_so1") else "-", engine_load(d)])
            for d in pts:
                if d["n_threads"] != t or not gget(d, g):
                    continue
                x = gget(d, g)
                r = ref.get(d["family"])
                mem = d["memory"]
                rd = d.get("ready", {})
                eq = "" if (d.get("equal_input") or {}).get("passed") else " UNCHECKED"
                rows.append([pytorch_col(d) + eq, d["family"], f"{x['total_ms']['p50']:.1f}", f"{x['total_ms']['p95']:.1f}",
                             f"{r['total_ms']['p50'] / x['total_ms']['p50']:.2f}x" if r else "-", f"{x['cpu_s']['mean']:.3f}",
                             f"{mib(mem['after_load']['rss'])} / {mib(mem['end']['rss_peak'])}", f"{d['load']['ms']:.0f}",
                             f"{rd['spawn_to_loaded_ms']:.0f}" if rd.get("spawn_to_loaded_ms") is not None else "-",
                             f"{rd['spawn_to_first_answer_ms']:.0f}" if rd.get("spawn_to_first_answer_ms") is not None else "-", pt_load(d)])
            if rows:
                out += [f"### {g}, t{t}", ""]
                out += [table(["config", "family", "p50 ms", "p95 ms", "speedup", "cpu_s", "RSS MiB", "load ms", "ready ms",
                               "first ms", "loadavg"], rows), ""]

    idle = []
    for d in pts:
        ai = d.get("after_idle")
        if ai:
            r = ai["runs"][0]
            idle.append([f"{pytorch_col(d)} t{d['n_threads']}", ai["idle_ms"] // 1000, r["id"], f"{r['total_ms']:.1f}",
                         f"{r['p50_ms']:.1f}", f"{r['total_ms'] / r['p50_ms']:.2f}x" if r["p50_ms"] else "-"])
    if idle:
        out += ["### PyTorch first request after idle (ms; engine rows in K6 above)", ""]
        out += [table(["config", "idle s", "request", "after idle", "warm p50", "ratio"], idle), ""]
    return "\n".join(out)


#
# --summary: engine vs PyTorch across every run in the paths (thread counts live in different stamps)
#

def collect_checks(paths):
    """decision-equal-input-check JSON (bench-decision-pytorch.py check) under the paths, any file name."""
    out = []
    for p in paths:
        p = Path(p)
        for f in (sorted(p.iterdir()) if p.is_dir() else [p]):
            if f.suffix != ".json" or NAME_RE.match(f.name) or f.stat().st_size > 64 << 20:
                continue
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(d, dict) and d.get("kind") == "decision-equal-input-check":
                d["_file"] = f.name
                out.append(d)
    return out


def read_sizes(path):
    """`ls -l` output -> {file stem: bytes}."""
    sizes = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 9 and parts[4].isdigit():
            sizes[Path(parts[-1]).stem] = int(parts[4])
    return sizes


def geomean(v):
    v = [x for x in v if x and x > 0]
    if not v:
        return None
    return math.exp(sum(math.log(x) for x in v) / len(v))


def engine_name(d):
    """Column name of one engine run: model stem plus what differs from the default run."""
    stem = Path(d["model"]).stem
    parts = []
    if d.get("plan") and d["plan"] != "sequential":
        parts.append(d["plan"])
    if not d.get("engine_params", {}).get("mmap", True):
        parts.append("no-mmap")
    if d.get("kernels") and d["kernels"] != "cpu":
        parts.append(d["kernels"])
    if "repack" in d["_label"] and d.get("kernels") == "cpu":
        parts.append("repack-req")  # requested, no x86 repack exists: a second run of the default kernels
    return " ".join([stem] + parts)


def engine_order(d):
    """f32, f16, q8_0, then the rest; variants (packed, repack-req, ...) after their base run."""
    stem = Path(d["model"]).stem
    prec = stem.rsplit("-", 1)[-1]
    return (stem.rsplit("-", 1)[0], {"f32": 0, "f16": 1, "q8_0": 2}.get(prec, 3), engine_name(d), d["_stamp"])


def summary_report(runs, checks, overrides, sizes):
    eng = []
    for stamp, r in runs.items():
        for d in r["bench"]:
            d["_stamp"], d["_label"] = stamp, r["label"]
            v = r["machine"].get(f"run {Path(d['model']).stem} t{d['n_threads']}", "")
            m = re.search(r"loadavg ([\d.]+)", v)
            d["_load"] = m.group(1) if m else "-"
            d["_server"] = next((s for s in r["server"] if Path(s["model"]).stem == Path(d["model"]).stem
                                 and s["n_threads"] == d["n_threads"]), None)
            eng.append(d)
    pts = [d for r in runs.values() for d in r["pytorch"]]
    for d in pts:
        d["_load"] = (d.get("machine", {}).get("loadavg_start") or "-").split()[0]
    fam_of = lambda d: d["family"] if d["bench"] == "laya-pytorch-bench" else engine_family(d["model"], overrides)

    def pt_name(d):
        return f"PyTorch {d['dtype']}" + (" batch" if d.get("router_mode") == "batch" else "")

    def by_id(d, key):
        return {x[key]: x for x in d["groups" if key == "group" else "requests"]}

    out = ["## Summary across runs (bench-decision-report.py --summary)", ""]

    if checks:
        out += ["### Equal inputs and fidelity (engine logits vs PyTorch fp32 on the same token ids)", "",
                "passed = every item has identical input ids, the same argmax and max |dlogit| <= tolerance. The "
                "tolerance is meant for F32; F16 / Q8_0 rows are expected to exceed it and are listed for their "
                "argmax agreement and probability error.", ""]
        rows = []
        for c in sorted(checks, key=lambda c: (c["reference"]["family"], Path(c["engine"]["model"]).stem)):
            rows.append([c["reference"]["family"], Path(c["engine"]["model"]).stem, c["engine"].get("kernels"), c["engine"].get("n_threads"),
                         f"{c['n_identical_inputs']}/{c['n_items']}", f"{c['argmax_same']}/{c['n_identical_inputs']}",
                         f"{c['max_abs_dlogit']:.1e}" if c.get("max_abs_dlogit") is not None else "-",
                         f"{c['mean_abs_dlogit']:.1e}" if c.get("mean_abs_dlogit") is not None else "-",
                         f"{c['max_abs_dp']:.1e}" if c.get("max_abs_dp") is not None else "-",
                         f"{c['tol_logit']:g}", "yes" if c["passed"] else "no", c["_file"]])
        out += [table(["family", "engine model", "kernels", "t", "identical ids", "argmax same", "max \\|dlogit\\|",
                       "mean \\|dlogit\\|", "max \\|dp\\|", "tol", "passed", "file"], rows), ""]

    families = sorted({fam_of(d) for d in eng + pts})
    for fam in families:
        fe = [d for d in eng if fam_of(d) == fam]
        fp = [d for d in pts if d["family"] == fam]
        threads = sorted({d["n_threads"] for d in fe} & {d["n_threads"] for d in fp})
        for t in threads:
            # --no-mmap runs are memory probes (see the memory table), not a latency configuration
            E = sorted([d for d in fe if d["n_threads"] == t and d.get("engine_params", {}).get("mmap", True)], key=engine_order)
            P = sorted([d for d in fp if d["n_threads"] == t], key=lambda d: (d["dtype"] != "fp32", d.get("router_mode", ""), d["_stamp"]))
            ref = next((d for d in reversed(P) if d["dtype"] == "fp32" and d.get("router_mode", "sequential") == "sequential"), None)
            if ref is None:
                continue
            refb = next((d for d in reversed(P) if d["dtype"] == "fp32" and d.get("router_mode") == "batch"), None)
            cols = [(engine_name(d), d) for d in E] + [(pt_name(d), d) for d in P]
            out += [f"### {fam}, t{t}: p50 ms per request group (speedup = PyTorch fp32 sequential p50 / cell p50)", ""]
            legend = []
            for n, d in cols:
                legend.append([n, d["_stamp"], d["_load"], f"{d['warmup']} + {d['repeat']}",
                               d.get("plan", "-") + " / " + d.get("plan_independent", "-") if d["bench"] == "llama-decision-bench"
                               else f"system_one, router {d.get('router_mode', 'sequential')}"])
            out += [table(["column", "stamp (UTC)", "loadavg at start", "warmup + repeat", "plan (items / router)"], legend), ""]
            groups = [g["group"] for g in ref["groups"]]
            rg = by_id(ref, "group")
            rows = []
            for g in groups:
                row = [g]
                for n, d in cols:
                    x = by_id(d, "group").get(g)
                    if not x:
                        row.append("-")
                        continue
                    p50 = x["total_ms"]["p50"]
                    row.append(f"{p50:.1f}" if d is ref else f"{p50:.1f} ({rg[g]['total_ms']['p50'] / p50:.2f}x)")
                rows.append(row)
            out += [table(["group"] + [n for n, _ in cols], rows), ""]

            # per request: the group p50 pools requests of different lengths; the geometric mean does not
            rr = by_id(ref, "id")
            rb = by_id(refb, "id") if refb else {}
            ecols = [(n, d) for n, d in cols if d["bench"] == "llama-decision-bench"]
            if refb:
                ecols.append((pt_name(refb), refb))
            rows = []
            ratios = {n: [] for n, _ in ecols}
            ratios_best = {n: [] for n, _ in ecols}
            for rid, r in rr.items():
                row = [rid, r["n_tokens"]]
                for n, d in ecols:
                    x = by_id(d, "id").get(rid)
                    if not x:
                        row.append("-")
                        continue
                    s = r["total_ms"]["p50"] / x["total_ms"]["p50"]
                    ratios[n].append(s)
                    best = min(r["total_ms"]["p50"], rb[rid]["total_ms"]["p50"]) if rid in rb else r["total_ms"]["p50"]
                    ratios_best[n].append(best / x["total_ms"]["p50"])
                    row.append(f"{s:.2f}x")
                rows.append(row)
            def gm(v):  # a column that ran only some requests (--filter) gets no geometric mean
                return f"{geomean(v):.2f}x" if len(v) == len(rr) else f"- ({len(v)} of {len(rr)} requests)"
            rows.append(["geomean", ""] + [gm(ratios[n]) for n, _ in ecols])
            if refb:
                rows.append(["geomean vs best PyTorch fp32", ""] + [gm(ratios_best[n]) for n, _ in ecols])
            out += [f"Per request, {fam} t{t}: speedup = PyTorch fp32 sequential p50 / column p50"
                    + ("; best PyTorch = the faster of sequential and batch for that request" if refb else ""), ""]
            out += [table(["request", "tokens"] + [n for n, _ in ecols], rows), ""]

    out += ["### Startup and memory", "",
            "load: model load in process (engine: llama-decision-bench; PyTorch: Agent construction). ready: engine "
            "llama-server spawn -> /health 200; PyTorch process start -> model loaded. first: the same start -> first "
            "so-1q-noul answer (engine over HTTP). RSS counts mapped file pages that were touched; anon is Linux RssAnon "
            "(the engine maps token_embd, so its RSS grows with the vocabulary a workload touches, up to anon + file). "
            "MiB = 2^20 bytes.", ""]
    rows = []
    for d in sorted(eng, key=lambda d: (fam_of(d), d["n_threads"]) + engine_order(d)):
        m, s = d["memory"], d["_server"]
        size = sizes.get(Path(d["model"]).stem)
        rows.append([f"engine {engine_name(d)}", fam_of(d), d["n_threads"], d["_stamp"], f"{d['load']['ms']:.0f}",
                     f"{s['ready_ms']:.0f}" if s else "-",
                     f"{s['ready_ms'] + s['first_so1']['total_ms']:.0f}" if s and s.get("first_so1") else "-",
                     f"{mib(m['after_load']['rss'])} / {mib(m['end']['rss_peak'])}",
                     f"{mib(m['after_load']['footprint'])} / {mib(m['end']['footprint'])}", mib(size) if size else "-"])
    for d in sorted(pts, key=lambda d: (d["family"], d["n_threads"], d["dtype"], d.get("router_mode", ""))):
        m, rd = d["memory"], d.get("ready", {})
        rows.append([pt_name(d), d["family"], d["n_threads"], d["_stamp"], f"{d['load']['ms']:.0f}",
                     f"{rd['spawn_to_loaded_ms']:.0f}" if rd.get("spawn_to_loaded_ms") is not None else "-",
                     f"{rd['spawn_to_first_answer_ms']:.0f}" if rd.get("spawn_to_first_answer_ms") is not None else "-",
                     f"{mib(m['after_load']['rss'])} / {mib(m['end']['rss_peak'])}",
                     f"{mib(m['after_load']['footprint'])} / {mib(m['end']['footprint'])}", "-"])
    out += [table(["config", "family", "t", "stamp", "load ms", "ready ms", "first ms", "RSS after load / peak",
                   "anon after load / end", "file MiB"], rows), ""]

    rows = []
    for d in eng + pts:
        ai = d.get("after_idle")
        if ai and ai.get("runs"):
            r = ai["runs"][0]
            name = f"engine {engine_name(d)}" if d["bench"] == "llama-decision-bench" else f"{pt_name(d)} {d['family']}"
            rows.append([name, d["n_threads"], ai["idle_ms"] // 1000, r["id"], f"{r['total_ms']:.1f}", f"{r['p50_ms']:.1f}",
                         f"{r['total_ms'] / r['p50_ms']:.2f}x" if r["p50_ms"] else "-"])
    if rows:
        out += ["### First request after idle (one sample per config, in process)", ""]
        out += [table(["config", "t", "idle s", "request", "after idle ms", "warm p50 ms", "ratio"], rows), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--stamp", help="engine run stamp to report (default: newest with engine results)")
    ap.add_argument("--family", action="append", default=[], metavar="STEM=FAMILY",
                    help="checkpoint family of an engine model file stem (engine vs PyTorch tables)")
    ap.add_argument("--summary", action="store_true",
                    help="engine vs PyTorch across every run in the paths (all stamps), plus equal-input checks")
    ap.add_argument("--model-sizes", metavar="FILE", help="--summary: `ls -l` of the model files (file MiB column)")
    args = ap.parse_args()
    overrides = {}
    for f in args.family:
        k, sep, v = f.partition("=")
        if not sep:
            print(f"--family {f}: expected STEM=FAMILY", file=sys.stderr)
            return 1
        overrides[k] = v
    runs = collect(args.paths)
    if not runs:
        print("no bench-decision results found", file=sys.stderr)
        return 1
    if args.summary:
        if args.model_sizes and not Path(args.model_sizes).is_file():
            print(f"no {args.model_sizes}", file=sys.stderr)
            return 1
        hosts = sorted({r["host"] for r in runs.values()})
        if len(hosts) > 1:
            print(f"--summary pairs runs of one machine; the paths hold {', '.join(hosts)}", file=sys.stderr)
            return 1
        sizes = read_sizes(args.model_sizes) if args.model_sizes else {}
        print(summary_report(runs, collect_checks(args.paths), overrides, sizes))
        return 0
    pytorch = [d for r in runs.values() for d in r["pytorch"]]
    engine_runs = [s for s, r in runs.items() if r["bench"] or r["server"]]
    stamp = args.stamp or (max(engine_runs) if engine_runs else None)
    if stamp is not None and stamp not in runs:
        print(f"no run with stamp {stamp}; have {', '.join(sorted(runs))}", file=sys.stderr)
        return 1
    parts = []
    if stamp is not None:
        parts.append(report(runs[stamp]))
    if pytorch:
        parts.append(compare_report(runs.get(stamp), pytorch, overrides))
    print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
