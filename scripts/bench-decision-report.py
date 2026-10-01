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

Parity gate: no speed number without a passing parity check of the same identity. Every engine
result (llama-decision-bench, llama-server) carries an "identity" (tests/laya/parity/identity.py);
--parity takes gate JSONs of tests/laya/verify_reference.py compare --gate ... --json (files or
folders, searched recursively). A row is printed only when a gate with verdict "pass" and no
identity problems has, as its candidate, a run of the same build tree, GGUF and device
(identity.speed_identity_problems); a CPU run also counts when it was the baseline of such a gate.
Other rows are listed under "Refused by the parity gate" with the reason, without numbers.
--allow-ungated-cpu prints CPU rows that have no parity record (marked "ungated"; for CPU-only
folders measured before the gate existed); a GPU row is never printed without one.

CPU vs GPU results of scripts/bench-decision-device.py (bench JSONs with an "ab" block, *__ready.json,
*__machine.json) get their own section: latency per group and config pooled over the blocks,
the paired speedup against --ref (default: the first config) per cycle, and resources (time to
ready, RSS, GPU memory, weights in device memory, placement, GPU utilization and clocks, load).
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests" / "laya" / "parity"))
import identity as ident  # noqa: E402

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
        run = runs.setdefault(m["stamp"], {"machine": {}, "bench": [], "server": [], "pytorch": [], "ab": [], "ready": [],
                                           "machine_json": {}, "chat": None, "host": m["host"], "label": m["label"]})
        if m["rest"] == "machine.txt":
            run["machine"] = read_facts(f)
        elif f.suffix == ".json":
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            if m["rest"] == "machine.json":
                run["machine_json"] = d
            elif d.get("kind") == "ready":
                run["ready"].append(d)
            elif d.get("kind") == "chat":
                run["chat"] = d
            elif d.get("bench") == "llama-decision-bench" and d.get("ab"):
                run["ab"].append(d)
            elif d.get("kind") == "server":
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


def report(run, gate):
    bench, refused = gate_rows(run["bench"], gate, col)
    server, refused_s = gate_rows(run["server"], gate, lambda s: col(s) + " server")
    refused += refused_s
    bench = sorted(bench, key=lambda d: (Path(d["model"]).stem, d["n_threads"]))
    server = sorted(server, key=lambda d: (Path(d["model"]).stem, d["n_threads"]))
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
    kept = [[col(d), d["_parity"]] for d in bench] + [[col(s) + " server", s["_parity"]] for s in server]
    if kept:
        out += ["### Parity of the rows above", "", table(["config", "parity"], kept), ""]
    if refused:
        out += ["### Refused by the parity gate (no numbers)", "", table(["config", "reason"], refused), ""]
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


def compare_report(run, pytorch, overrides, gate):
    """Engine vs PyTorch reference: one table per request group and thread count."""
    if run:
        run = dict(run)
        run["bench"], _ = gate_rows(run["bench"], gate, col)  # refused rows are listed in the run's own section
        run["server"], _ = gate_rows(run["server"], gate, col)
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


#
# parity gate
#

def load_parity(paths):
    """Gate records from verify_reference.py compare --json files: {run name: {"gate": ..., "identity": ...}}."""
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files += sorted(x for x in p.rglob("*.json") if not x.name.endswith(".identity.json"))
        elif p.is_file():
            files.append(p)
        else:
            print(f"--parity {p}: not found", file=sys.stderr)
    recs = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        for name, v in d.items():
            if not (isinstance(v, dict) and isinstance(v.get("gate"), dict) and isinstance(v.get("identity"), dict)):
                continue
            g = v["gate"]
            recs.append({"file": str(f), "name": name, "tier": g.get("tier"), "verdict": g.get("verdict"),
                         "problems": v.get("identity_problems") or [], "candidate": v["identity"].get("candidate"),
                         "baseline": v["identity"].get("baseline")})
    return recs


def closeness(why):
    """Rank of a non-covering record for the "closest" reason: another GGUF or backend is far away."""
    return sum(4 if w.startswith(("GGUF", "backend")) else 2 if w.startswith("device") else 1 for w in why)


class ParityGate:
    def __init__(self, recs, allow_ungated_cpu=False):
        self.recs = recs
        self.allow_ungated_cpu = allow_ungated_cpu

    def check(self, run_id):
        """(ok, text): the covering gate, or why the row is refused."""
        if not run_id:
            return False, "no identity record in the result"
        near = None
        for r in self.recs:
            usable = r["verdict"] == "pass" and not r["problems"]
            roles = [("candidate", r["candidate"])]
            if (r["baseline"] or {}).get("runtime", {}).get("backend", "cpu") == "cpu":
                roles.append(("baseline", r["baseline"]))
            for role, pid in roles:
                why = ident.speed_identity_problems(run_id, pid)
                if not why:
                    if usable:
                        return True, f"{r['tier']} pass ({Path(r['file']).name}, {role})"
                    why = [f"its gate {Path(r['file']).name} did not pass ({r['verdict']}" + (", identity problems" if r["problems"] else "") + ")"]
                if near is None or closeness(why) < closeness(near):
                    near = why
        rt = run_id.get("runtime") or {}
        if self.allow_ungated_cpu and rt.get("backend", "cpu") == "cpu":
            return True, "ungated (CPU, --allow-ungated-cpu)"
        if not self.recs:
            return False, "no parity gate JSON given (--parity)"
        return False, "no passing parity gate of this identity; closest: " + "; ".join(near or ["-"])


def gate_rows(items, gate, label):
    """Splits results into (kept, refused rows) by the parity gate."""
    kept, refused = [], []
    for d in items:
        ok, why = gate.check(d.get("identity"))
        if ok:
            d["_parity"] = why
            kept.append(d)
        else:
            refused.append([label(d), why])
    return kept, refused


#
# CPU vs GPU (scripts/bench-decision-device.py)
#

def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def fmt(x, f="{:.0f}"):
    return "-" if x is None else f.format(x)


def device_report(run, gate, ref_name=None):
    facts = run["machine_json"]
    out = [f"## Decision bench CPU vs GPU {run['host']} ({run['label']})", ""]
    rows = [(k, facts[k]) for k in ("date_utc", "host", "os", "cpu", "logical_cpus", "cgroup_cpu_max", "blocks", "repeat", "warmup",
                                    "ready", "load_max", "foreign_max", "background", "chat_model", "loadavg_start", "loadavg_end")
            if facts.get(k) is not None]
    for g in facts.get("gpus") or []:
        rows.append((f"gpu {g.get('index')}", f"{g.get('name')}, driver {g.get('driver')}, {g.get('memory_total_mib')} MiB, "
                                              f"max SM {g.get('max_sm_mhz')} MHz / mem {g.get('max_mem_mhz')} MHz, power limit {g.get('power_limit_w')} W"))
    out += [table(["fact", "value"], [(k, str(v).replace("|", "/")) for k, v in rows]), ""]
    configs = facts.get("configs") or []
    if configs:
        out += [table(["config", "device", "gpu", "threads", "kernels", "precision", "env", "bin"],
                      [[c["name"], c["device"], c.get("gpu") or "-", c.get("threads") or "-", c.get("kernels") or "-", c.get("precision") or "-",
                        " ".join(f"{k}={v}" for k, v in (c.get("env") or {}).items()) or "-", c["bin"]] for c in configs]), ""]
    names = [c["name"] for c in configs] or sorted({d["ab"]["config"]["name"] for d in run["ab"]})
    ref_name = ref_name or (names[0] if names else None)

    by = {}
    for d in run["ab"]:
        by.setdefault((Path(d["model"]).stem, d["ab"]["config"]["name"]), []).append(d)
    ready = {(Path(r["model"]).stem, r["config"]["name"]): r for r in run["ready"]}
    models = []
    for (m, _c) in by:
        if m not in models:
            models.append(m)

    # parity gate per (model, config): every block must be covered
    status, refused = {}, []
    for (m, c), ds in by.items():
        checks = [gate.check(d.get("identity")) for d in ds]
        bad = [w for ok, w in checks if not ok]
        status[(m, c)] = (not bad, bad[0] if bad else checks[0][1])
        if bad:
            refused.append([m, c, bad[0]])
    out += ["### Parity gate", "", table(["model", "config", "parity"],
                                         [[m, c, ("" if ok else "REFUSED: ") + w] for (m, c), (ok, w) in status.items()]), ""]

    for m in models:
        cfgs = [c for c in names if (m, c) in by and status[(m, c)][0]]
        if not cfgs:
            continue
        out += [f"### {m}", ""]
        groups = []
        for c in cfgs:
            for d in by[(m, c)]:
                for g in d["groups"]:
                    if g["group"] not in groups:
                        groups.append(g["group"])

        def pooled(c, g):
            v = []
            for d in by[(m, c)]:
                for r in d["requests"]:
                    if r["group"] == g:
                        v += r["runs_total_ms"]
            return v

        def block_p50(c, g):
            res = {}
            for d in by[(m, c)]:
                v = [x for r in d["requests"] if r["group"] == g for x in r["runs_total_ms"]]
                if v:
                    res[d["ab"]["block"]] = pct(v, 50)
            return res

        n_blocks = max(len(by[(m, c)]) for c in cfgs)
        out += [f"Latency per request, ms: p50 / p95 over requests x repeats x {n_blocks} blocks (in process, llama-decision-bench).", ""]
        rows = []
        for g in groups:
            row = [g]
            for c in cfgs:
                v = pooled(c, g)
                row.append("-" if not v else f"{pct(v, 50):.1f} / {pct(v, 95):.1f}")
            rows.append(row)
        out += [table(["group"] + cfgs, rows), ""]

        if ref_name in cfgs and len(cfgs) > 1:
            others = [c for c in cfgs if c != ref_name]
            out += [f"Speedup against {ref_name}: {ref_name} p50 / config p50 of the same cycle; median (min - max) over the cycles. "
                    "Above 1: faster than the reference.", ""]
            rows = []
            for g in groups:
                rb = block_p50(ref_name, g)
                row = [g]
                for c in others:
                    cb = block_p50(c, g)
                    r = [rb[b] / cb[b] for b in sorted(rb) if b in cb and cb[b] > 0]
                    row.append("-" if not r else f"{statistics.median(r):.2f}x ({min(r):.2f} - {max(r):.2f}, n={len(r)})")
                rows.append(row)
            out += [table(["group"] + [f"{c} vs {ref_name}" for c in others], rows), ""]

        rows = []
        for c in cfgs:
            ds = by[(m, c)]
            att = [a for d in ds for a in d["ab"]["attempts"]]
            final = [d["ab"]["attempts"][-1] for d in ds]
            gpu = []
            for a in final:
                gs = (a.get("gpu") or {}).get("gpus") or {}
                if gs:  # the GPU this config kept busy
                    gpu.append(max(gs.values(), key=lambda x: (x.get("util") or {}).get("mean") or 0))
                elif (a.get("gpu") or {}).get("kind") == "ioreg":  # macOS: the whole GPU, no clocks or power
                    gpu.append({"util": a["gpu"].get("Device Utilization %")})
            gmem = [(a.get("gpu") or {}).get("process_mem_mib_max") for a in final]
            # macOS: memory the GPU driver has in use, all processes (ioreg "In use system memory")
            gmem += [x["max"] / 1048576 for x in [((a.get("gpu") or {}).get("In use system memory")) for a in final] if x]
            rd = ready.get((m, c), {}).get("runs", [])
            rd_ok = [r for r in rd if "ready_ms" in r]
            props = (rd_ok[0].get("props") if rd_ok else None) or {}
            wdev = ((props.get("memory") or ds[0].get("memory") or {}).get("weights_device_bytes"))
            pl = ds[0].get("placement") or props.get("placement") or {}
            cpu_n = (pl.get("backends") or {}).get("CPU")
            hashes = {d["suite_hash"] for d in ds}
            det = all(d["deterministic"] for d in ds) and len(hashes) == 1
            la = [a["loadavg_before"][0] for a in final if a.get("loadavg_before")]
            fc = [a.get("foreign_cores") for a in final if a.get("foreign_cores") is not None]
            util = [x["util"]["mean"] for x in gpu if x.get("util")]
            sm = [x["sm_mhz"]["median"] for x in gpu if x.get("sm_mhz")]
            mm = [x["mem_mhz"]["median"] for x in gpu if x.get("mem_mhz")]
            pw = [x["power_w"]["mean"] for x in gpu if x.get("power_w")]
            def cpu_per_req(group):
                return med([g["cpu_s"]["mean"] for d in ds for g in d["groups"] if g["group"] == group])
            on_gpu = ds[0].get("device", "cpu") != "cpu"

            def g_only(x):
                return x if on_gpu else "-"
            rows.append([c, fmt(med([d["load"]["ms"] for d in ds])),
                         fmt(med([r["ready_ms"] for r in rd_ok])) + (f" (n={len(rd_ok)})" if rd_ok else ""),
                         fmt(med([r.get("first_so1_ms") for r in rd_ok]), "{:.1f}"),
                         f"{mib(med([d['memory']['after_load']['rss'] for d in ds]))} / {mib(med([d['memory']['end']['rss_peak'] for d in ds]))}",
                         fmt(med([r.get("rss_kb_ready") for r in rd_ok]) and med([r.get("rss_kb_ready") for r in rd_ok]) / 1024),
                         g_only(fmt(max([x for x in gmem if x is not None], default=None))),
                         mib(wdev) if wdev else "0",
                         f"{pl.get('nodes', '-')} / {pl.get('splits', '-')} / {cpu_n if cpu_n is not None else '-'}",
                         g_only(fmt(med(util))), g_only(f"{fmt(med(sm))} / {fmt(med(mm))}"), g_only(fmt(med(pw))),
                         f"{fmt(cpu_per_req('systemone 1q'), '{:.3f}')} / {fmt(cpu_per_req('router N=8'), '{:.3f}')}",
                         f"{min(la):.1f} - {max(la):.1f}" if la else "-", fmt(max(fc) if fc else None, "{:.2f}"),
                         str(len(att) - len(final)), "yes" if det else "NO"])
        out += ["Resources: load = in-process model load (median over blocks); ready = llama-server --decision spawn -> /health 200, "
                "first = the first so-1q-noul request after ready (HTTP); RSS MiB after load / peak (bench) and at ready (server); "
                "GPU MiB = the most nvidia-smi showed for the bench process (macOS: the most ioreg showed in use by the GPU driver, all processes); weights = weights in device memory (/props); placement "
                "= graph nodes / splits / nodes on the CPU (64-token graph); util %, SM / memory MHz and W: GPU samples every 200 ms "
                "during the blocks (median of the block means; GPU configs only); CPU s = process CPU seconds per request of "
                "systemone 1q / router N=8 (median of the block means); "
                "load = 1-minute load average before the blocks; foreign = cores other processes used (max over blocks); re-run = "
                "disturbed blocks measured again; det = the same logits hash in every run of every block.", ""]
        out += [table(["config", "load ms", "ready ms", "first ms", "RSS", "RSS ready", "GPU MiB", "weights MiB", "placement",
                       "util %", "SM / mem MHz", "W", "CPU s 1q / N=8", "load", "foreign", "re-run", "det"], rows), ""]
        chat = run.get("chat")
        if chat and chat.get("samples"):
            alone = [x["tg_ts"] for x in chat["samples"] if x["phase"] == "alone" and x.get("tg_ts")]
            rows = []
            for c in cfgs:
                win = [(a["t_start"], a["t_start"] + a["wall_s"]) for d in by[(m, c)] for a in d["ab"]["attempts"][-1:]]
                v = [x["tg_ts"] for x in chat["samples"] if x["phase"] == "load" and x.get("tg_ts")
                     and any(t0 <= x["t0"] and x["t1"] <= t1 for t0, t1 in win)]
                rows.append([c, fmt(med(v), "{:.1f}") + f" (n={len(v)})",
                             "-" if not v or not alone else f"{med(v) / med(alone):.2f}x"])
            out += [f"Chat model ({Path(chat['model']).name}, {chat['n_predict']} tokens per completion, -ngl {chat['ngl']}) generating "
                    f"the whole time: generation tokens/s of the completions that ran entirely inside the blocks of a config "
                    f"(median); alone: {fmt(med(alone), '{:.1f}')} tokens/s (n={len(alone)}, before the blocks).", ""]
            out += [table(["config", "chat tokens/s during the blocks", "vs alone"], rows), ""]
        hs = {c: {d["suite_hash"] for d in by[(m, c)]} for c in cfgs}
        out += ["Suite logits hash over the blocks: " + ", ".join(f"{c} {sorted(h)[0][:12]}" + ("" if len(h) == 1 else f" (+{len(h) - 1} more)")
                                                              for c, h in hs.items())
                + ". Different devices compute different bits; agreement is the parity gate's job.", ""]
    if refused:
        out += ["### Refused by the parity gate (no numbers)", "", table(["model", "config", "reason"], refused), ""]
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
    ap.add_argument("--parity", action="append", default=[], metavar="PATH",
                    help="parity gate JSON or folder (verify_reference.py compare --json); repeatable")
    ap.add_argument("--allow-ungated-cpu", action="store_true",
                    help="print CPU rows without a parity record, marked ungated (never GPU rows)")
    ap.add_argument("--ref", help="reference config of the CPU vs GPU speedups (default: the first config)")
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
    gate = ParityGate(load_parity(args.parity), args.allow_ungated_cpu)
    pytorch = [d for r in runs.values() for d in r["pytorch"]]
    engine_runs = [s for s, r in runs.items() if r["bench"] or r["server"] or r["ab"]]
    stamp = args.stamp or (max(engine_runs) if engine_runs else None)
    if stamp is not None and stamp not in runs:
        print(f"no run with stamp {stamp}; have {', '.join(sorted(runs))}", file=sys.stderr)
        return 1
    parts = []
    if stamp is not None:
        if runs[stamp]["bench"] or runs[stamp]["server"]:
            parts.append(report(runs[stamp], gate))
        if runs[stamp]["ab"]:
            parts.append(device_report(runs[stamp], gate, args.ref))
    if pytorch:
        parts.append(compare_report(runs.get(stamp), pytorch, overrides, gate))
    print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
