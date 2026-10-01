#!/usr/bin/env python3
"""PyTorch Laya reference on the llama-decision-bench suite: equal inputs first, then timing.

Usage (run with the laya reference venv, e.g. ~/.cache/laya-ref/.venv/bin/python):
    # 1. engine side: render every suite request through llama-server --decision --decision-debug
    #    (router states + token ids + raw logits of the engine)
    bench-decision-pytorch.py inputs --server build/bin/llama-server --model laya-f32.gguf -t 2 \
        [--kernels K] [--spec FILE] [--suite tests/decision/bench/suite.jsonl] -o inputs.json
    # 2. equal-input check: reference fp32 input ids and raw logits against the engine F32 ones
    bench-decision-pytorch.py check --ckpt multilingual --inputs inputs.json -t 2 [--tol-logit 1e-3] -o check.json
    # 3. timing (refused unless check.json passed for the same suite and checkpoint)
    bench-decision-pytorch.py bench --ckpt multilingual --inputs inputs.json --check check.json -t 2 \
        [--dtype fp32|bf16] [--repeat 10] [--warmup 1] [--idle-ms 60000] [--router-mode sequential|batch] \
        [--filter TEXT] [--label pytorch] -o OUT_DIR
    python3 scripts/bench-decision-report.py OUT_DIR ENGINE_DIR     # side-by-side tables

--ckpt is a checkpoint directory (with rl_agent_config.json) or a name found in the Hugging Face
cache: multilingual (convaiinnovations/laya-multilingual), laya (English), typed-decisions.

Same requests. systemone bodies go to Agent.system_one as they are. A router request is one
independent noul sequence per candidate over "<card-v1>\\n\\nsuccess criterion: ...\\n\\ntask: ...".
That state is not rendered here: `inputs` takes it from the engine (POST /v1/decision/render), so
there is no second card renderer that could drift. The router question is spec.router.question of
the model (GGUF decision.spec or --spec), else the built-in one (DECISION.md, "laya mapping"). The
check then compares the reference input ids with the engine token ids byte for byte, so a wrong
question or state fails there, before any timing.

Same boundaries as llama-decision-bench "total": body text -> JSON, request checks, tokenize,
forward, post-processing and the response text. Process start, model load and HTTP are outside.
The engine also renders the cards inside its total (string work, microseconds); here the states
come pre-rendered. Router timing: --router-mode sequential (default) calls system_one once per
candidate, one sequence each, as the engine plan does; batch sends all candidates through one
predict_batch (padded rows, the PyTorch throughput path) and is labelled as such.

Measured per request: wall ms (warmup and repeat rounds interleaved over the suite, like the
engine), CPU seconds (getrusage user + system of the process, all threads), p50/p95/p99 with
linear interpolation. Per process: RSS / footprint after load, after warmup, peak (macOS
task_info as in decision-bench.cpp, Linux /proc/self/status), time to ready (process start ->
torch imported -> model loaded -> first answer), and with --idle-ms the first request after an
idle gap (every request once, as decision-bench --idle-ms). torch.set_num_threads(t),
set_num_interop_threads(1), inference_mode. fp32 is the reference; bf16 is the reference's own CPU
autocast (LAYA_CPU_AMP=bf16); when autocast falls back to fp32 the result says so.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import platform
import re
import resource
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

T_SCRIPT = time.time()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SUITE = os.path.join(ROOT, "tests", "decision", "bench", "suite.jsonl")
BENCH_NAME = "laya-pytorch-bench"
HF_NAMES = {
    "multilingual": "convaiinnovations/laya-multilingual",
    "laya-multilingual": "convaiinnovations/laya-multilingual",
    "laya": "convaiinnovations/laya",
    "english": "convaiinnovations/laya",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
    "laya-typed-decisions": "convaiinnovations/laya-typed-decisions",
}
# DECISION.md, "laya mapping"; used when the model spec has no router question
ROUTER_DEFAULT_QUESTION = {
    "type": "noul",
    "instructions": "Will the executor meet the success criterion on this task?",
    "criteria": {"true": "meets the criterion", "false": "does not meet the criterion"},
}


def die(msg):
    print("error: " + msg, file=sys.stderr, flush=True)
    sys.exit(1)


#
# suite, statistics, machine facts
#

def load_suite(path, filt=None):
    if not os.path.isfile(path):
        die("no %s. Run:  python3 tests/decision/bench/gen_suite.py" % path)
    raw = open(path, "rb").read()
    reqs, seen = [], set()
    for n, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if not isinstance(r.get("id"), str) or not isinstance(r.get("body"), dict):
            die("%s:%d: needs a string id and an object body" % (path, n))
        if r["id"] in seen:
            die("%s:%d: duplicate id %s" % (path, n, r["id"]))
        seen.add(r["id"])
        r.setdefault("endpoint", "systemone")
        r.setdefault("group", r["endpoint"])
        if r["endpoint"] not in ("systemone", "router"):
            die("%s:%d: endpoint must be systemone or router" % (path, n))
        if filt and filt not in r["id"] and filt not in r["group"]:
            continue
        reqs.append(r)
    if not reqs:
        die("no requests in %s%s" % (path, " match " + filt if filt else ""))
    return reqs, hashlib.sha256(raw).hexdigest()


def pct(v, p):
    if not v:
        return 0.0
    v = sorted(v)
    pos = p / 100.0 * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def stats(v):
    return {"n": len(v), "mean": sum(v) / len(v) if v else 0.0, "min": min(v) if v else 0.0, "p50": pct(v, 50),
            "p95": pct(v, 95), "p99": pct(v, 99), "max": max(v) if v else 0.0}


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def power_source():
    if sys.platform == "darwin":
        m = re.search(r"'(.*?)'", sh(["pmset", "-g", "batt"]))
        return m.group(1) if m else ""
    for p in ("/sys/class/power_supply/AC/online", "/sys/class/power_supply/ACAD/online", "/sys/class/power_supply/ADP1/online"):
        if os.path.exists(p):
            return "AC Power" if open(p).read().strip() == "1" else "Battery Power"
    return ""


def load_avg():
    try:
        return " ".join("%.2f" % x for x in os.getloadavg())
    except OSError:
        return ""


def top_cpu():
    out = sh(["ps", "-Ao", "pcpu=,comm=", "-r"] if sys.platform == "darwin" else ["ps", "-eo", "pcpu=,comm=", "--sort=-pcpu"])
    rows = []
    for line in out.splitlines()[:5]:
        p, _, c = line.strip().partition(" ")
        rows.append("%s %s" % (p, c.strip().split("/")[-1]))
    return rows


def machine_facts():
    f = {"host": platform.node(), "kernel": " ".join(platform.uname()[i] for i in (0, 2, 4)), "python": sys.version.split()[0]}
    if sys.platform == "darwin":
        f["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
        f["cores"] = "%s (%s P + %s E)" % (sh(["sysctl", "-n", "hw.physicalcpu"]), sh(["sysctl", "-n", "hw.perflevel0.physicalcpu"]),
                                          sh(["sysctl", "-n", "hw.perflevel1.physicalcpu"]))
        f["memory_gb"] = int(sh(["sysctl", "-n", "hw.memsize"]) or 0) // 2**30
        f["os"] = "macOS %s (%s)" % (sh(["sw_vers", "-productVersion"]), sh(["sw_vers", "-buildVersion"]))
        m = re.search(r"(lowpowermode|powermode)\s+(\d+)", sh(["pmset", "-g"]))
        f["low_power_mode"] = m.group(2) if m else ""
    else:
        try:
            f["cpu"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        except Exception:
            f["cpu"] = platform.processor()
        f["cores"] = "%d logical" % os.cpu_count()
        try:
            f["memory_gb"] = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) // 2**20
        except Exception:
            pass
    return f


#
# process facts (same numbers as decision-bench.cpp bench_mem_now / bench_cpu_seconds)
#

def cpu_seconds():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


_libc = ctypes.CDLL(ctypes.util.find_library("c")) if sys.platform == "darwin" else None


def mem_now():
    m = {"rss": -1, "rss_peak": -1, "footprint": -1, "footprint_peak": -1}
    if sys.platform == "darwin":
        # task_vm_info_data_t as uint64 words: resident_size [2], resident_size_peak [3],
        # phys_footprint [18] (rev1), ledger_phys_footprint_peak [21] (rev3)
        buf = (ctypes.c_uint64 * 64)()
        count = ctypes.c_uint32(64 * 2)
        task = ctypes.c_uint32.in_dll(_libc, "mach_task_self_")
        if _libc.task_info(task, 22, ctypes.byref(buf), ctypes.byref(count)) == 0:  # TASK_VM_INFO
            n = count.value // 2
            m["rss"], m["rss_peak"] = int(buf[2]), int(buf[3])
            if n > 18:
                m["footprint"] = int(buf[18])
            if n > 21:
                m["footprint_peak"] = int(buf[21])
    elif os.path.exists("/proc/self/status"):
        for line in open("/proc/self/status"):
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "RssAnon"):
                b = int(v.split()[0]) * 1024
                m[{"VmRSS": "rss", "VmHWM": "rss_peak", "RssAnon": "footprint"}[k]] = b
    return m


def process_start():
    """Wall time of the process start and where it came from."""
    if os.environ.get("BENCH_SPAWN_T"):
        return float(os.environ["BENCH_SPAWN_T"]), "BENCH_SPAWN_T"
    try:
        if sys.platform == "darwin":
            # sysctl kern.proc.pid: kinfo_proc starts with extern_proc.p_starttime (timeval)
            mib = (ctypes.c_int * 4)(1, 14, 1, os.getpid())
            buf = ctypes.create_string_buffer(1024)
            size = ctypes.c_size_t(1024)
            if _libc.sysctl(mib, 4, buf, ctypes.byref(size), None, 0) == 0:
                sec = int.from_bytes(buf.raw[0:8], "little")
                usec = int.from_bytes(buf.raw[8:12], "little")
                t = sec + usec * 1e-6
                if 0 < T_SCRIPT - t < 600:
                    return t, "kern.proc.pid"
        elif os.path.exists("/proc/self/stat"):
            ticks = int(open("/proc/self/stat").read().rsplit(")", 1)[1].split()[19])
            since = time.clock_gettime(time.CLOCK_BOOTTIME) - ticks / os.sysconf("SC_CLK_TCK")
            return time.time() - since, "/proc/self/stat"
    except Exception:
        pass
    return T_SCRIPT, "script start (interpreter start not counted)"


#
# router question of the model spec
#

def router_question(model_path, spec_path):
    spec = None
    if spec_path:
        spec = json.load(open(spec_path, encoding="utf-8"))
    else:
        sys.path.insert(0, os.path.join(ROOT, "gguf-py"))
        try:
            import gguf
            from gguf.scripts.gguf_new_metadata import get_field_data
        except ImportError as e:
            die("cannot import gguf-py (%s); run with the laya venv python or pass --spec" % e)
        text = get_field_data(gguf.GGUFReader(model_path, "r"), "decision.spec")
        spec = json.loads(text) if text else None
    q = dict(ROUTER_DEFAULT_QUESTION)
    r = (spec or {}).get("router") or {}
    if r.get("question"):
        q = dict(r["question"])
        if q.get("criteria") is None:
            q["criteria"] = ROUTER_DEFAULT_QUESTION["criteria"]
    return q, ("spec" if r.get("question") else "built-in")


#
# inputs: the engine's view of every request
#

def start_server(server, model, threads, kernels, spec, log_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    cmd = [server, "--decision", "--decision-debug", "--decision-allow-uncalibrated", "-m", model, "--device", "none",
           "-t", str(threads), "--host", "127.0.0.1", "--port", str(port), "--decision-max-items", "16"]
    cmd += (["--decision-kernels", kernels] if kernels else []) + (["--decision-spec", spec] if spec else [])
    print(" ".join(cmd), flush=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, LC_ALL="C"))
    base = "http://127.0.0.1:%d" % port
    t0 = time.time()
    while True:
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return proc, base, cmd
        except Exception:
            pass
        if proc.poll() is not None or time.time() - t0 > 180:
            proc.kill()
            die("server did not start, see " + log_path)
        time.sleep(0.2)


def post(base, path, body):
    data = json.dumps(body, ensure_ascii=True).encode("ascii")
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        die("%s: http %d: %s" % (path, e.code, e.read().decode("utf-8", "replace")[:400]))


def cmd_inputs(a):
    for p in (a.server, a.model):
        if not os.path.exists(p):
            die("missing " + p)
    reqs, suite_sha = load_suite(a.suite, a.filter)
    rq, rq_source = router_question(a.model, a.spec)
    out = {"kind": "decision-inputs", "version": 1, "suite": a.suite, "suite_sha256": suite_sha,
           "router_question": rq, "router_question_source": rq_source, "requests": []}
    proc, base, cmd = start_server(a.server, a.model, a.threads, a.kernels, a.spec, a.output + ".server.log")
    try:
        props = json.loads(urllib.request.urlopen(base + "/props").read())
        out["engine"] = {"model": a.model, "cmd": cmd, "build_info": props.get("build_info"), "decision": props["decision"]}
        d = props["decision"]
        print("engine: layout %s, plan %s / router %s, kernels %s, %s threads, input_contract %s, special_tokens %s" % (
            d["layout"], d["plan"]["name"], d["plan"]["router"], d["plan"]["kernels"], d["plan"]["n_threads"],
            d.get("input_contract"), d.get("special_tokens")), flush=True)
        for r in reqs:
            rec = {"id": r["id"], "endpoint": r["endpoint"], "group": r["group"], "body": r["body"], "items": []}
            if r["endpoint"] == "systemone":
                res = post(base, "/v1/systemone", r["body"])
                for qid in r["body"]["questions"]:
                    dbg = res["answers"][qid]["debug"]
                    rec["items"].append({"id": qid, "tokens": dbg["tokens"], "logits": dbg["logits"], "n_state_cut": dbg["n_state_cut"]})
            else:
                ren = post(base, "/v1/decision/render", r["body"])["items"]
                sc = post(base, "/v1/router/score", r["body"])["scores"]
                if [x["id"] for x in ren] != [c["id"] for c in r["body"]["candidates"]] or [x["id"] for x in sc] != [x["id"] for x in ren]:
                    die("%s: render / score candidate order differs from the request" % r["id"])
                for x, s in zip(ren, sc):
                    if x["instructions"] != rq["instructions"] or x["type"] != "noul" or x["keys"] != ["false", "true"]:
                        die("%s: engine router question %r differs from %r (%s); pass --spec" % (r["id"], x["instructions"], rq["instructions"], rq_source))
                    if s["debug"]["tokens"] != x["tokens"]:
                        die("%s/%s: /v1/router/score tokens differ from /v1/decision/render" % (r["id"], x["id"]))
                    rec["items"].append({"id": x["id"], "state": x["state"], "tokens": x["tokens"], "logits": s["debug"]["logits"],
                                         "n_state_cut": x["n_state_cut"]})
            n_tok = sum(len(i["tokens"]) for i in rec["items"])
            print("  %-16s %-10s %2d items %5d tokens" % (r["id"], r["endpoint"], len(rec["items"]), n_tok), flush=True)
            out["requests"].append(rec)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("wrote %s (%d requests; router question: %s)" % (a.output, len(out["requests"]), rq_source), flush=True)


#
# reference loading
#

def resolve_ckpt(name):
    if os.path.isfile(os.path.join(name, "rl_agent_config.json")):
        path = os.path.abspath(name)
        m = re.search(r"models--([^/]+)--([^/]+)/snapshots/[^/]+(?:/(\w[\w-]*))?$", path)
        family = (m.group(3) and "laya-" + m.group(3)) or (m.group(2) if m else os.path.basename(path))
        return path, family
    repo = HF_NAMES.get(name)
    if not repo:
        die("--ckpt %s: not a checkpoint directory and not one of %s" % (name, ", ".join(sorted(HF_NAMES))))
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    snaps = os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots")
    cands = sorted((os.path.join(snaps, s) for s in os.listdir(snaps)) if os.path.isdir(snaps) else [],
                   key=os.path.getmtime, reverse=True)
    cands = [c for c in cands if os.path.isfile(os.path.join(c, "rl_agent_config.json"))]
    if not cands:
        die("no %s in %s. Run:  hf download %s" % (repo, hub, repo))
    return cands[0], repo.split("/")[1]


def load_torch(threads):
    t0 = time.time()
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch, (time.time() - t0) * 1000


def load_agent(torch, ckpt, dtype):
    os.environ["HF_HUB_OFFLINE"] = "1"
    if dtype == "bf16":
        os.environ["LAYA_CPU_AMP"] = "bf16"
    else:
        os.environ.pop("LAYA_CPU_AMP", None)
    from laya.agent import Agent
    agent = Agent(ckpt, device="cpu")
    agent.model.eval()
    if agent.device.type != "cpu":
        die("reference is not on the CPU")
    if dtype == "fp32" and (agent.amp_enabled or agent.dtype != torch.float32):
        die("reference fp32 is not fp32 (LAYA_CPU_AMP set?)")
    if dtype == "bf16" and not agent.amp_enabled:
        die("reference did not enable bf16 autocast")
    return agent


def torch_facts(torch, agent):
    import laya
    return {"torch": torch.__version__, "laya": getattr(laya, "__version__", "?"), "num_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "mkldnn": torch.backends.mkldnn.is_available(),
            "openmp": torch.backends.openmp.is_available(), "config": torch.__config__.show(),
            "parallel_info": torch.__config__.parallel_info(), "max_len": agent.cfg.get("max_len"),
            "head_max_len": agent.cfg.get("head_max_len")}


def read_inputs(path, suite_filter=None):
    if not os.path.isfile(path):
        die("no %s. Run:  bench-decision-pytorch.py inputs --server BIN/llama-server --model laya-f32.gguf -t 2 -o %s" % (path, path))
    d = json.load(open(path, encoding="utf-8"))
    if d.get("kind") != "decision-inputs":
        die("%s is not a bench-decision-pytorch.py inputs file" % path)
    if suite_filter:
        d["requests"] = [r for r in d["requests"] if suite_filter in r["id"] or suite_filter in r["group"]]
        if not d["requests"]:
            die("no requests match " + suite_filter)
    return d


#
# check: reference fp32 input ids and raw logits against the engine
#

def reference_rows(torch, agent, state, questions):
    """Agent request path without the answer rounding: {qid: (input_ids, raw logits)}."""
    from laya.agent import Agent
    from laya.common import collate_items
    qids = list(questions)
    for qid in qids:
        Agent._check_question(qid, questions[qid])
    internal = {qid: Agent._to_internal(questions[qid]) for qid in qids}
    enc = agent._encode_state(state, qids, internal)
    b = collate_items([enc], agent.tok.pad_token_id)
    with torch.inference_mode():
        logits, _ = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    logits = logits.float().numpy()
    return {qid: (enc[r]["ids"], logits[r, :len(enc[r]["markers"])].tolist()) for r, qid in enumerate(qids)}


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def cmd_check(a):
    inp = read_inputs(a.inputs)
    ckpt, family = resolve_ckpt(a.ckpt)
    torch, _ = load_torch(a.threads)
    agent = load_agent(torch, ckpt, "fp32")
    print("reference %s (%s), fp32, %d threads; engine %s" % (family, ckpt, a.threads, inp["engine"]["model"]), flush=True)
    rq = inp["router_question"]
    rows, bad = [], []
    for r in inp["requests"]:
        if r["endpoint"] == "systemone":
            ref = reference_rows(torch, agent, r["body"]["state"], r["body"]["questions"])
            pairs = [(it["id"], ref[it["id"]], it) for it in r["items"]]
        else:
            pairs = []
            for it in r["items"]:
                ref = reference_rows(torch, agent, it["state"], {it["id"]: rq})
                pairs.append((it["id"], ref[it["id"]], it))
        for qid, (ids, logits), it in pairs:
            where = "%s/%s" % (r["id"], qid)
            if ids != it["tokens"]:
                first = next((i for i in range(min(len(ids), len(it["tokens"]))) if ids[i] != it["tokens"][i]), min(len(ids), len(it["tokens"])))
                bad.append("%s: input ids differ at %d (reference %d tokens, engine %d)" % (where, first, len(ids), len(it["tokens"])))
                continue
            d = max(abs(x - y) for x, y in zip(logits, it["logits"]))
            am = max(range(len(logits)), key=logits.__getitem__) == max(range(len(it["logits"])), key=it["logits"].__getitem__)
            dp = max(abs(x - y) for x, y in zip(softmax(logits), softmax(it["logits"])))
            rows.append({"item": where, "n_tokens": len(ids), "max_abs_dlogit": d, "argmax_same": am, "max_abs_dp": dp,
                         "reference_logits": logits, "engine_logits": it["logits"]})
            if d > a.tol_logit or not am:
                bad.append("%s: max|dlogit| %.3e%s" % (where, d, "" if am else ", argmax differs"))
    worst = max(rows, key=lambda x: x["max_abs_dlogit"]) if rows else None
    res = {"kind": "decision-equal-input-check", "version": 1, "passed": not bad, "suite_sha256": inp["suite_sha256"],
           "inputs": os.path.abspath(a.inputs), "engine": {"model": inp["engine"]["model"], "kernels": inp["engine"]["decision"]["plan"]["kernels"],
                                                         "n_threads": inp["engine"]["decision"]["plan"]["n_threads"]},
           "reference": {"ckpt": ckpt, "family": family, "dtype": "fp32", "n_threads": a.threads}, "tol_logit": a.tol_logit,
           "n_items": len(rows) + sum(1 for b in bad if "input ids" in b), "n_identical_inputs": len(rows),
           "argmax_same": sum(1 for x in rows if x["argmax_same"]),
           "max_abs_dlogit": worst["max_abs_dlogit"] if worst else None, "max_abs_dlogit_at": worst["item"] if worst else None,
           "mean_abs_dlogit": sum(x["max_abs_dlogit"] for x in rows) / max(1, len(rows)),
           "max_abs_dp": max((x["max_abs_dp"] for x in rows), default=None), "failures": bad, "items": rows}
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("equal inputs: %d / %d items with identical input ids, argmax same %d / %d, max|dlogit| %.3e (%s), max|dp| %.3e" % (
        res["n_identical_inputs"], res["n_items"], res["argmax_same"], len(rows), res["max_abs_dlogit"] or 0, res["max_abs_dlogit_at"],
        res["max_abs_dp"] or 0), flush=True)
    for b in bad[:20]:
        print("  FAIL " + b, flush=True)
    print("%s -> %s" % ("PASSED" if not bad else "FAILED (inputs differ: fix the harness, not the engine)", a.output), flush=True)
    return 0 if not bad else 1


#
# bench: timing through the reference API
#

def cmd_bench(a):
    t_spawn, spawn_src = process_start()
    inp = read_inputs(a.inputs, a.filter)
    ckpt, family = resolve_ckpt(a.ckpt)
    check = None
    if a.check:
        if not os.path.isfile(a.check):
            die("no %s. Run:  bench-decision-pytorch.py check --ckpt %s --inputs %s -o %s" % (a.check, a.ckpt, a.inputs, a.check))
        check = json.load(open(a.check))
        if not check.get("passed"):
            die("%s did not pass: timings would compare different inputs" % a.check)
        if check["suite_sha256"] != inp["suite_sha256"] or os.path.realpath(check["reference"]["ckpt"]) != os.path.realpath(ckpt):
            die("%s was made for another suite or checkpoint" % a.check)
    elif not a.unchecked:
        die("--check CHECK.json is required (or --unchecked, which marks the result untrusted)")
    os.makedirs(a.output, exist_ok=True)

    rq = inp["router_question"]
    reqs = []
    for r in inp["requests"]:
        text = json.dumps(r["body"], ensure_ascii=False)
        reqs.append({"id": r["id"], "endpoint": r["endpoint"], "group": r["group"], "text": text,
                     "states": [(it["id"], it["state"]) for it in r["items"]] if r["endpoint"] == "router" else None,
                     "n_items": len(r["items"]), "n_tokens": sum(len(it["tokens"]) for it in r["items"])})

    facts = machine_facts()
    facts.update({"power_source": power_source(), "loadavg_start": load_avg(), "top_cpu_start": top_cpu()})
    mem_start = mem_now()
    t_import0 = time.time()
    torch, import_ms = load_torch(a.threads)
    t_torch = time.time()
    cpu0 = cpu_seconds()
    agent = load_agent(torch, ckpt, a.dtype)
    t_loaded = time.time()
    load_ms, load_cpu = (t_loaded - t_torch) * 1000, cpu_seconds() - cpu0
    mem_load = mem_now()
    print("pytorch %s %s, %d threads (interop %d), router %s, %d requests, warmup %d, repeat %d, import %.0f ms, load %.0f ms" % (
        family, a.dtype, torch.get_num_threads(), torch.get_num_interop_threads(), a.router_mode, len(reqs), a.warmup, a.repeat,
        import_ms, load_ms), flush=True)

    def run(r):
        c0 = cpu_seconds()
        t0 = time.perf_counter()
        with torch.inference_mode():
            body = json.loads(r["text"])
            if r["endpoint"] == "systemone":
                res = agent.system_one(body["state"], body["questions"])
            elif a.router_mode == "batch":
                out = agent.predict_batch([s for _, s in r["states"]], {"router": rq})
                res = {"object": "router.scores", "scores": [{"id": cid, "p_success": o["answers"]["router"]["noul"]}
                                                             for (cid, _), o in zip(r["states"], out)]}
            else:
                scores = []
                for cid, state in r["states"]:
                    o = agent.system_one(state, {cid: rq})
                    scores.append({"id": cid, "p_success": o["answers"][cid]["noul"]})
                res = {"object": "router.scores", "scores": scores}
            text = json.dumps(res)
        t1 = time.perf_counter()
        c1 = cpu_seconds()
        if r["endpoint"] == "router" and len(res["scores"]) != len(r["states"]):
            die("%s: %d scores for %d candidates" % (r["id"], len(res["scores"]), len(r["states"])))
        ans = res.get("answers", res.get("scores"))
        return (t1 - t0) * 1000, c1 - c0, hashlib.sha256(json.dumps(ans, sort_keys=True).encode()).hexdigest()

    series = {r["id"]: {"total_ms": [], "cpu_s": [], "hashes": set()} for r in reqs}
    first = None
    for w in range(a.warmup):
        for r in reqs:
            ms, _, h = run(r)
            if first is None:
                first = (r["id"], ms, time.time())
            series[r["id"]]["hashes"].add(h)
    mem_warm = mem_now()
    for k in range(a.repeat):
        for r in reqs:
            ms, cs, h = run(r)
            if first is None:
                first = (r["id"], ms, time.time())
            s = series[r["id"]]
            s["total_ms"].append(ms)
            s["cpu_s"].append(cs)
            s["hashes"].add(h)
        print("pytorch: repeat %d/%d" % (k + 1, a.repeat), flush=True)
    mem_end = mem_now()
    after_idle = None
    if a.idle_ms > 0:
        print("pytorch: idle %d ms" % a.idle_ms, flush=True)
        time.sleep(a.idle_ms / 1000.0)
        runs = []
        for r in reqs:
            ms, cs, h = run(r)
            series[r["id"]]["hashes"].add(h)
            runs.append({"id": r["id"], "total_ms": ms, "cpu_s": cs, "p50_ms": pct(series[r["id"]]["total_ms"], 50)})
        after_idle = {"idle_ms": a.idle_ms, "runs": runs}
    amp_end = bool(agent.amp_enabled)
    dtype_eff = str(agent.dtype).replace("torch.", "") if amp_end else "float32"
    if a.dtype == "bf16" and not amp_end:
        print("warning: bf16 autocast fell back to fp32 on this CPU/torch build; the bf16 rows are fp32", flush=True)

    jreqs, groups, order = [], {}, []
    for r in reqs:
        s = series[r["id"]]
        if r["group"] not in groups:
            order.append(r["group"])
            groups[r["group"]] = {"total_ms": [], "cpu_s": []}
        groups[r["group"]]["total_ms"] += s["total_ms"]
        groups[r["group"]]["cpu_s"] += s["cpu_s"]
        jreqs.append({"id": r["id"], "endpoint": r["endpoint"], "group": r["group"], "n_items": r["n_items"], "n_tokens": r["n_tokens"],
                      "deterministic": len(s["hashes"]) == 1, "n_hashes": len(s["hashes"]), "total_ms": stats(s["total_ms"]),
                      "cpu_s": stats(s["cpu_s"]), "runs_total_ms": s["total_ms"], "runs_cpu_s": s["cpu_s"]})
    facts.update({"loadavg_end": load_avg(), "power_source_end": power_source(), "top_cpu_end": top_cpu()})
    tag = "pytorch-%s-%s%s" % (family, a.dtype, "-batch" if a.router_mode == "batch" else "")
    res = {
        "bench": BENCH_NAME, "version": 1, "model": ckpt, "family": family, "suite": inp["suite"], "suite_sha256": inp["suite_sha256"],
        "inputs": os.path.abspath(a.inputs), "n_threads": a.threads, "interop_threads": torch.get_num_interop_threads(),
        "dtype": a.dtype, "dtype_effective": dtype_eff, "autocast_end": amp_end, "router_mode": a.router_mode,
        "repeat": a.repeat, "warmup": a.warmup, "machine": facts, "runtime": torch_facts(torch, agent),
        "load": {"import_torch_ms": import_ms, "ms": load_ms, "cpu_s": load_cpu, "first_request_ms": first[1] if first else None,
                 "first_request_id": first[0] if first else None},
        "ready": {"process_start": spawn_src, "spawn_to_script_ms": (T_SCRIPT - t_spawn) * 1000,
                  "spawn_to_torch_ms": (t_torch - t_spawn) * 1000, "spawn_to_loaded_ms": (t_loaded - t_spawn) * 1000,
                  "spawn_to_first_answer_ms": (first[2] - t_spawn) * 1000 if first else None,
                  "note": "input checks before the import: %.0f ms" % ((t_import0 - T_SCRIPT) * 1000)},
        "memory": {"start": mem_start, "after_load": mem_load, "after_warmup": mem_warm, "end": mem_end},
        "deterministic": all(len(s["hashes"]) == 1 for s in series.values()),
        "equal_input": ({"check": os.path.abspath(a.check), "passed": True, "engine": check["engine"],
                         "max_abs_dlogit": check["max_abs_dlogit"], "argmax_same": check["argmax_same"], "n_items": check["n_items"]}
                        if check else {"passed": False, "note": "--unchecked"}),
        "groups": [{"group": g, "total_ms": stats(groups[g]["total_ms"]), "cpu_s": stats(groups[g]["cpu_s"])} for g in order],
        "requests": jreqs, "after_idle": after_idle,
    }
    stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
    host = (platform.node().split(".")[0] or "host")
    path = os.path.join(a.output, "%s__%s__%s__%s__t%d.json" % (host, a.label, stamp, tag, a.threads))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    print("\n| %-16s | %5s | %6s | %9s | %9s | %9s | %7s | %3s |" % ("request", "items", "tokens", "p50 ms", "p95 ms", "p99 ms", "cpu_s", "det"), flush=True)
    for r in jreqs:
        print("| %-16s | %5d | %6d | %9.1f | %9.1f | %9.1f | %7.3f | %3s |" % (
            r["id"], r["n_items"], r["n_tokens"], r["total_ms"]["p50"], r["total_ms"]["p95"], r["total_ms"]["p99"], r["cpu_s"]["mean"],
            "yes" if r["deterministic"] else "NO"), flush=True)
    mib = lambda b: b / 2**20 if b >= 0 else -1
    print("\nmemory MiB: rss after load %.0f, rss peak %.0f, footprint after load %.0f, footprint peak %.0f" % (
        mib(mem_load["rss"]), mib(mem_end["rss_peak"]), mib(mem_load["footprint"]), mib(mem_end["footprint_peak"])), flush=True)
    rd = res["ready"]
    print("ready (%s): torch %.0f ms, loaded %.0f ms, first answer %.0f ms" % (
        spawn_src, rd["spawn_to_torch_ms"], rd["spawn_to_loaded_ms"], rd["spawn_to_first_answer_ms"] or -1), flush=True)
    if after_idle:
        f0 = after_idle["runs"][0]
        print("after %d ms idle: %s %.1f ms (its p50 %.1f ms)" % (a.idle_ms, f0["id"], f0["total_ms"], f0["p50_ms"]), flush=True)
    print("wrote " + path, flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inputs", help="render the suite through llama-server --decision --decision-debug")
    p.add_argument("--server", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("-t", "--threads", type=int, default=2)
    p.add_argument("--kernels")
    p.add_argument("--spec")
    p.add_argument("--suite", default=DEFAULT_SUITE)
    p.add_argument("--filter")
    p.add_argument("-o", "--output", required=True)
    p = sub.add_parser("check", help="reference fp32 against the engine logits of an inputs file")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("-t", "--threads", type=int, default=2)
    p.add_argument("--tol-logit", type=float, default=1e-3)
    p.add_argument("-o", "--output", required=True)
    p = sub.add_parser("bench", help="time the reference on the suite")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--inputs", required=True)
    p.add_argument("--check")
    p.add_argument("--unchecked", action="store_true")
    p.add_argument("-t", "--threads", type=int, default=4)
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--idle-ms", type=int, default=0)
    p.add_argument("--router-mode", choices=("sequential", "batch"), default="sequential")
    p.add_argument("--filter")
    p.add_argument("--label", default="pytorch")
    p.add_argument("-o", "--output", required=True, help="output directory")
    a = ap.parse_args()
    if getattr(a, "threads", 1) < 1 or getattr(a, "repeat", 1) < 1 or getattr(a, "warmup", 0) < 0:
        die("threads and repeat must be >= 1, warmup >= 0")
    if a.cmd == "bench" and "__" in a.label:
        die("--label must not contain '__' (the report splits file names on it)")
    return {"inputs": cmd_inputs, "check": cmd_check, "bench": cmd_bench}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
