"""Parity of llama-laya-cli against the PyTorch reference (the `laya` package) on decision items.

    <laya-python> tests/laya/verify_reference.py ref <checkpoint-dir> <items.jsonl> <ref.jsonl> [--english] [--shard I/N] [--threads T] [--api-every N]
    python3       tests/laya/verify_reference.py cat <ref.jsonl> <shard.jsonl> [<shard.jsonl> ...]
    python3       tests/laya/verify_reference.py cli <llama-laya-cli> <model.gguf> <ref.jsonl> <cli.jsonl> [--jobs J] [-t T] [--kernels K] [--english]
                  [--jsonl] [--device cpu|gpu|auto] [--gpu N] [--precision default|strict] [--plan packed|sequential]
    python3       tests/laya/verify_reference.py server <llama-server> <model.gguf> <ref.jsonl> <server.jsonl> [-t T] [--kernels K] [--english]
                  [--device cpu|gpu|auto] [--gpu N] [--precision default|strict]
    python3       tests/laya/verify_reference.py compare <ref.jsonl> <cli.jsonl> [<cli.jsonl> ...] [--json out.json]
                  [--gate strict-f32|f16-class [--weights f16|f32|q8_0] [--tiers tiers.json] [--allow-identity-mismatch]]

items.jsonl has one {"id", "cat", "state", "questions"} object per line (tests/laya/parity/gen_corpus.py writes it).

ref      runs the reference on the CPU in fp32 through the Agent's own request path (_check_question,
         _to_internal, _encode_state, collate_items, DecisionModel.forward) and records per question the
         input_ids, marker positions, raw scorer logits and raw act logits. Every 5th item also goes
         through agent.system_one: its calibrated answers check the temperatures (temperature_by_options
         buckets included) end to end (--api-every 1: every item; the public-output gate needs that).
         --english keeps only the items whose letters are all ASCII.
         Shards (--shard I/N) write separate files; the `cat` command joins them with their identities.
cli      runs llama-laya-cli on the same items (one process per item, -t 1 by default), with
         ensure_ascii JSON inputs so the CLI sees exactly the Python objects the reference saw.
         The CLI packs all questions of an item into one graph (block-diagonal mask); --plan sequential
         runs one graph per question instead, as the server does (on a device the plans can differ in bits).
         --jsonl: llama-laya-cli --jsonl instead, one process per job over a contiguous slice of the items
         (the model loads once per job; each line is exactly the -f run of that item, see laya-cli.cpp).
         --device / --gpu / --precision go to llama-laya-cli; the identity takes the device, kernels,
         precision and graph placement from the CLI's "laya-cli: runtime {json}" stderr line.
server   runs the same items through one llama-server --decision --decision-debug process over HTTP
         (POST /v1/systemone, one request at a time), i.e. the code path the server really runs: plan
         `sequential` (one graph per question), the server's default threads (the performance cores)
         unless -t, and the kernels of --kernels (default: the server's own default, `auto`);
         --device / --gpu / --precision become --decision-device / --decision-gpu / --decision-precision.
         Records
         debug.tokens / debug.logits / debug.act_logits and action.act_probability per question; the
         server derives the marker positions from the tokens, so compare checks the input ids only.
         (A server build from before action.act_probability has no act logits: compare skips the act
         head for such a run, the gate counts it as server_no_act_head.)
         /props.decision.plan goes to <server.jsonl>.props.json.
compare  matches inputs first (input_ids and marker_pos equal byte for byte), then reports per CLI file:
         argmax agreement, max / mean |dlogit|, mean / max TVD of softmax(logits), act-head |d|, and
         the calibrated probabilities of the CLI answers against the reference system_one answers.
         --gate TIER turns it into the pass / fail public-output gate of tests/laya/parity/parity_gate.py
         with the thresholds of tests/laya/parity/tiers.json (DECISION.md, "Backend parity tiers"):
         strict-f32 = F32 GGUF against this reference; f16-class = a device run against the CPU run of
         the same GGUF (then the first file is that CPU run). It first checks the identity records
         (<file>.identity.json: corpus sha256 and subset, checkpoint revision or GGUF sha256, backend,
         the activation class of the baseline kernels; a candidate that quantizes activations takes the
         activation_quantized numbers) and refuses runs that do not fit the tier unless
         --allow-identity-mismatch. Exit code 1 when
         a candidate fails, 2 on an identity refusal. The JSON gets the verdict, the failures, the known
         intentional differences (counted apart) and the identities.

Every run writes <out>.identity.json (tests/laya/parity/identity.py): build tree sha256, GGUF or
checkpoint, corpus sha256, backend / device / kernels / plan / threads, host.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity"))
import identity as ident  # noqa: E402
import parity_gate  # noqa: E402


def arg_value(args, name, default=None):
    if name in args:
        i = args.index(name)
        v = args[i + 1]
        del args[i:i + 2]
        return v
    return default


def is_english(item) -> bool:
    s = json.dumps([item["state"], item["questions"]], ensure_ascii=False)
    return all(c.isascii() for c in s if c.isalpha())


def cmd_ref(args):
    shard = arg_value(args, "--shard", "0/1")
    threads = int(arg_value(args, "--threads", "4"))
    api_every = int(arg_value(args, "--api-every", "5"))
    english = "--english" in args
    args = [a for a in args if a != "--english"]
    if len(args) != 3:
        sys.exit(__doc__)
    ckpt, items_path, out_path = args
    for p in (os.path.join(ckpt, "rl_agent_config.json"), items_path):
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    k, n = (int(x) for x in shard.split("/"))

    import torch
    os.environ.pop("LAYA_CPU_AMP", None)
    torch.set_num_threads(threads)
    from laya.agent import Agent
    from laya.common import collate_items, render_options

    items = [json.loads(line) for line in open(items_path, encoding="utf-8")]
    if english:
        items = [it for it in items if is_english(it)]
    items = items[k::n]

    import laya
    ident.write_sidecar(out_path, ident.make_identity(
        "ref", corpus=items_path, subset="english" if english else "all", n_items=len(items),
        runtime={"device": "cpu", "backend": "torch-cpu", "precision": "fp32", "n_threads": threads, "plan": "agent-batch"},
        extra={"model": ident.checkpoint_identity(ckpt), "shard": shard, "api_every": api_every,
               "reference": {"laya": getattr(laya, "__version__", None), "torch": torch.__version__}}))
    agent = Agent(ckpt, device="cpu")
    agent.model.eval()
    assert agent.device.type == "cpu" and not agent.amp_enabled and agent.dtype == torch.float32
    assert all(p.dtype == torch.float32 for p in agent.model.parameters())
    tok = agent.tok
    print("shard %s: %d items, max_len %d, head_max_len %d" % (shard, len(items), agent.cfg["max_len"], agent.cfg["head_max_len"]), flush=True)

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for i, it in enumerate(items):
            rec = {"id": it["id"], "cat": it["cat"], "state": it["state"], "questions": it["questions"], "per_question": {}}
            try:
                qids = list(it["questions"].keys())
                for qid in qids:
                    Agent._check_question(qid, it["questions"][qid])
                internal = {qid: Agent._to_internal(it["questions"][qid]) for qid in qids}
                enc = agent._encode_state(it["state"], qids, internal)
                b = collate_items([enc], tok.pad_token_id)
                with torch.no_grad():
                    logits, act = agent.model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
                logits = logits.float().numpy()
                act = act.float().numpy()
                act_sm = torch.softmax(torch.from_numpy(act), -1).numpy()
                for r, qid in enumerate(qids):
                    nk = len(enc[r]["markers"])
                    rec["per_question"][qid] = {
                        "qtype": internal[qid]["t"],
                        "options": render_options(internal[qid]),
                        "input_ids": enc[r]["ids"],
                        "marker_pos": enc[r]["markers"],
                        "raw_logits": logits[r, :nk].tolist(),
                        "act_raw_logits": act[r].tolist(),
                        "act_probability": float(act_sm[r, 0]),
                    }
                if api_every > 0 and i % api_every == 0:
                    rec["api"] = agent.system_one(it["state"], it["questions"])["answers"]
            except Exception as e:  # recorded: the CLI must refuse the same items
                rec["error"] = "%s: %s" % (type(e).__name__, e)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print("shard %s %d/%d %.0fs" % (shard, i, len(items), time.time() - t0), flush=True)
    print("shard %s done: %d items %.0fs" % (shard, len(items), time.time() - t0), flush=True)


def cli_runtime(stderr_text):
    """The "laya-cli: runtime {json}" line of a llama-laya-cli run (None for a build without it)."""
    for line in stderr_text.splitlines():
        if line.startswith("laya-cli: runtime "):
            return json.loads(line[len("laya-cli: runtime "):])
    return None


def cmd_cli(args):
    jobs = int(arg_value(args, "--jobs", "8"))
    threads = arg_value(args, "-t", "1")
    kernels = arg_value(args, "--kernels", None)
    device = arg_value(args, "--device", None)
    gpu = arg_value(args, "--gpu", None)
    precision = arg_value(args, "--precision", None)
    plan = arg_value(args, "--plan", None)
    english = "--english" in args
    jsonl = "--jsonl" in args
    args = [a for a in args if a not in ("--english", "--jsonl")]
    if len(args) != 4:
        sys.exit(__doc__)
    cli, model, ref_path, out_path = args
    for p in (cli, model, ref_path):
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    items = [json.loads(line) for line in open(ref_path, encoding="utf-8")]
    if english:
        items = [it for it in items if is_english(it)]
    extra = (["--kernels", kernels] if kernels else []) + (["--device", device] if device else []) + \
            (["--gpu", gpu] if gpu else []) + (["--precision", precision] if precision else []) + \
            (["--plan", plan] if plan else [])
    runtime = {"device": "cpu", "backend": "cpu", "kernels": kernels or "cli-default", "plan": plan or "packed", "n_threads": int(threads),
               "precision": precision or "default"}
    seen = []  # the runtime lines of the CLI runs

    def write_identity():
        rt = dict(runtime)
        if seen:
            r = seen[0]
            rt.update({"device": r["device"], "backend": ident.backend_of(r["device"]), "precision": r["precision"],
                       "kernels_resolved": r["kernels"], "placement": r.get("placement"),
                       "device_description": r.get("device_description", "")})
            if r["device"] != "cpu":
                rt["kernels"] = r["kernels"]  # a device computes with its own kernels ("metal", "cuda", ...)
            if any(x.get("device") != r["device"] or x.get("placement") != r.get("placement") for x in seen[1:]):
                rt["runtime_mismatch"] = seen
        ident.write_sidecar(out_path, ident.make_identity(
            "cli", exes=[cli], build=ident.build_root_of(cli), model=model, corpus=ref_path,
            subset="english" if english else "all", n_items=len(items), runtime=rt,
            extra={"mode": "jsonl" if jsonl else "file"}))

    write_identity()
    tmpdir = tempfile.mkdtemp(prefix="laya_cli_")
    env = dict(os.environ, LC_ALL="C")

    def record(it, d):
        rec = {"id": it["id"]}
        if "error" in d:
            rec["error"] = "jsonl: " + d["error"]
        else:
            rec["per_question"] = d["per_question"]
            rec["answers"] = d["answers"]
        return rec

    def run(it):
        path = os.path.join(tmpdir, it["id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"state": it["state"], "questions": it["questions"]}, f, ensure_ascii=True)
        cmd = [cli, "-m", model, "-f", path, "-t", threads] + extra
        p = subprocess.run(cmd, capture_output=True, env=env)
        os.unlink(path)
        rt = cli_runtime(p.stderr.decode("utf-8", "replace"))
        if rt and len(seen) < 2:
            seen.append(rt)
        rec = {"id": it["id"]}
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", "replace").strip().splitlines()
            rec["error"] = "rc=%d: %s" % (p.returncode, err[-1] if err else "")
            return rec
        d = json.loads(p.stdout.decode("utf-8"))
        rec["per_question"] = d["per_question"]
        rec["answers"] = d["answers"]
        return rec

    def run_slice(part):
        # one llama-laya-cli --jsonl process over part (a list of items), records in item order
        k, chunk = part
        path = os.path.join(tmpdir, "part-%03d.jsonl" % k)
        with open(path, "w", encoding="utf-8") as f:
            for it in chunk:
                f.write(json.dumps({"state": it["state"], "questions": it["questions"]}, ensure_ascii=True) + "\n")
        cmd = [cli, "-m", model, "--jsonl", path, "-t", threads] + extra
        with open(path + ".err", "wb") as err:  # the CLI writes bytes to the fd
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=err, env=env)
        with open(path + ".err", encoding="utf-8", errors="replace") as f:
            err_text = f.read()
        rt = cli_runtime(err_text)
        if rt:
            seen.append(rt)
        lines = p.stdout.decode("utf-8").split("\n")[:-1]
        if p.returncode != 0 or len(lines) != len(chunk):
            tail = err_text.strip().splitlines()
            msg = "rc=%d, %d of %d lines: %s" % (p.returncode, len(lines), len(chunk), tail[-1] if tail else "")
            recs = [record(it, json.loads(l)) for it, l in zip(chunk, lines)]
            return recs + [{"id": it["id"], "error": msg} for it in chunk[len(lines):]]
        os.unlink(path)
        os.unlink(path + ".err")
        return [record(it, json.loads(l)) for it, l in zip(chunk, lines)]

    t0 = time.time()
    if jsonl:
        n_jobs = max(1, min(jobs, len(items)))
        step = (len(items) + n_jobs - 1) // n_jobs
        parts = [(k, items[k * step:(k + 1) * step]) for k in range(n_jobs) if items[k * step:(k + 1) * step]]
        print("cli --jsonl: %d processes, %s" % (len(parts), " ".join([cli, "-m", model, "-t", threads] + extra)), flush=True)
        with open(out_path, "w", encoding="utf-8") as out, ThreadPoolExecutor(len(parts)) as ex:
            for recs in ex.map(run_slice, parts):
                for rec in recs:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print("cli %d/%d %.0fs" % (sum(1 for _ in open(out_path, encoding="utf-8")), len(items), time.time() - t0), flush=True)
    else:
        with open(out_path, "w", encoding="utf-8") as out, ThreadPoolExecutor(jobs) as ex:
            for i, rec in enumerate(ex.map(run, items)):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    print("cli %d/%d %.0fs" % (i, len(items), time.time() - t0), flush=True)
    write_identity()
    try:
        os.rmdir(tmpdir)
    except OSError:
        print("kept %s (failed parts)" % tmpdir, flush=True)
    if seen:
        print("device %s, kernels %s, placement %s" % (seen[0]["device"], seen[0]["kernels"], json.dumps(seen[0].get("placement"))), flush=True)
    print("cli done: %d items %.0fs -> %s" % (len(items), time.time() - t0, out_path), flush=True)


def cmd_server(args):
    import socket
    import urllib.error
    import urllib.request

    threads = arg_value(args, "-t", None)
    kernels = arg_value(args, "--kernels", None)
    device = arg_value(args, "--device", None)
    gpu = arg_value(args, "--gpu", None)
    precision = arg_value(args, "--precision", None)
    english = "--english" in args
    args = [a for a in args if a != "--english"]
    if len(args) != 4:
        sys.exit(__doc__)
    server, model, ref_path, out_path = args
    for p in (server, model, ref_path):
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    items = [json.loads(line) for line in open(ref_path, encoding="utf-8")]
    if english:
        items = [it for it in items if is_english(it)]
    max_q = max(len(it["questions"]) for it in items)

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = "http://127.0.0.1:%d" % port
    cmd = [server, "--decision", "--decision-debug", "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "--decision-max-items", str(max(16, max_q))]
    cmd += (["-t", threads] if threads else []) + (["--decision-kernels", kernels] if kernels else [])
    cmd += (["--decision-device", device] if device else []) + (["--decision-gpu", gpu] if gpu else []) + \
           (["--decision-precision", precision] if precision else [])
    env = dict(os.environ, LC_ALL="C")
    log = open(out_path + ".log", "wb")  # the server writes bytes to the fd
    print(" ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)

    def post(body):
        data = json.dumps(body, ensure_ascii=True).encode("ascii")
        req = urllib.request.Request(base + "/v1/systemone", data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8", "replace") or "null")

    t0 = time.time()
    try:
        while True:
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            if proc.poll() is not None or time.time() - t0 > 120:
                sys.exit("server did not start, see " + log.name)
            time.sleep(0.2)
        props = json.loads(urllib.request.urlopen(base + "/props").read())["decision"]
        identity = ident.make_identity("server", exes=[server], build=ident.build_root_of(server), model=model, corpus=ref_path,
                                       subset="english" if english else "all", n_items=len(items),
                                       runtime=ident.runtime_from_props(props))
        ident.write_sidecar(out_path, identity)
        with open(out_path + ".props.json", "w", encoding="utf-8") as f:
            json.dump({"cmd": cmd, "plan": props["plan"], "memory": props.get("memory"), "props": props, "identity": identity}, f, indent=1)
        print("plan %s, kernels %s, %d threads, device %s, placement %s" % (
            props["plan"]["name"], props["plan"]["kernels"], props["plan"]["n_threads"], props.get("device"),
            json.dumps(props.get("placement"))), flush=True)
        with open(out_path, "w", encoding="utf-8") as out:
            for i, it in enumerate(items):
                status, body = post({"state": it["state"], "questions": it["questions"]})
                rec = {"id": it["id"]}
                if status != 200:
                    rec["error"] = "http %d: %s" % (status, json.dumps(body, ensure_ascii=False)[:300])
                else:
                    rec["per_question"] = {}
                    for qid, a in body["answers"].items():
                        pq = {"input_ids": a["debug"]["tokens"], "raw_logits": a["debug"]["logits"]}
                        if "act_logits" in a["debug"]:
                            pq["act_raw_logits"] = a["debug"]["act_logits"]
                            pq["act_probability"] = a["action"]["act_probability"]
                        rec["per_question"][qid] = pq
                    rec["answers"] = body["answers"]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 100 == 0:
                    print("server %d/%d %.0fs" % (i, len(items), time.time() - t0), flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=30)
    print("server done: %d items %.0fs -> %s" % (len(items), time.time() - t0, out_path), flush=True)


def softmax(z):
    m = max(z)
    e = [math.exp(x - m) for x in z]
    s = sum(e)
    return [x / s for x in e]


def answer_probs(a):
    if "probabilities" in a:
        p = a["probabilities"]
        return list(p.values()) if isinstance(p, dict) else list(p)
    return [1.0 - a["noul"], a["noul"]]


def compare_one(ref, cli):
    mism, rows = [], []
    for iid, r in ref.items():
        c = cli.get(iid)
        qids = r["questions"] if isinstance(r.get("questions"), dict) else (r.get("per_question") or {"-": None})
        for qid in qids:
            if "error" in r:
                if c is None or "error" not in c:
                    mism.append((iid, qid, "cli_accepts_ref_error"))
                continue
            if c is None or "error" in c:
                mism.append((iid, qid, "cli_error: " + (c["error"] if c else "missing")))
                continue
            rq, cq = r["per_question"][qid], c["per_question"].get(qid)
            if cq is None:
                mism.append((iid, qid, "cli_missing_question"))
            elif cq["input_ids"] != rq["input_ids"] or cq.get("marker_pos", rq.get("marker_pos")) != rq.get("marker_pos"):
                a, b = rq["input_ids"], cq["input_ids"]
                first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                mism.append((iid, qid, "input_ids differ at %d (ref %d tokens, cli %d)" % (first, len(a), len(b))))
            else:
                rows.append((iid, qid, r, c, rq, cq))

    n_log = 0
    sum_d = max_d = 0.0
    max_at = None
    arg_ok = 0
    flips = []
    tvd_sum = tvd_max = 0.0
    act_max = act_dp = 0.0
    api_n = 0
    api_max = 0.0
    api_arg_ok = 0
    for iid, qid, r, c, rq, cq in rows:
        a, b = rq["raw_logits"], cq["raw_logits"]
        d = [abs(x - y) for x, y in zip(a, b)]
        n_log += len(d)
        sum_d += sum(d)
        if max(d) > max_d:
            max_d, max_at = max(d), "%s/%s" % (iid, qid)
        ok = max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        arg_ok += ok
        if not ok:
            srt = sorted(a, reverse=True)
            flips.append("%s/%s ref logit gap %.4f max|dlogit| %.4f" % (iid, qid, srt[0] - srt[1], max(d)))
        pa, pb = softmax(a), softmax(b)
        tvd = 0.5 * sum(abs(x - y) for x, y in zip(pa, pb))
        tvd_sum += tvd
        tvd_max = max(tvd_max, tvd)
        if "act_raw_logits" in cq and "act_raw_logits" in rq:  # older server runs have no act head output
            act_max = max(act_max, max(abs(x - y) for x, y in zip(rq["act_raw_logits"], cq["act_raw_logits"])))
            if "act_probability" in rq and "act_probability" in cq:  # older reference files have no act_probability
                act_dp = max(act_dp, abs(rq["act_probability"] - cq["act_probability"]))
        if "api" in r and qid in r["api"]:
            ra, ca = answer_probs(r["api"][qid]), answer_probs(c["answers"][qid])
            api_n += 1
            api_max = max(api_max, max(abs(x - y) for x, y in zip(ra, ca)))
            api_arg_ok += max(range(len(ra)), key=ra.__getitem__) == max(range(len(ca)), key=ca.__getitem__)
    nq = len(rows)
    return {
        "questions": nq + len(mism), "identical_inputs": nq, "mismatches": len(mism),
        "mismatch_list": ["%s/%s: %s" % m for m in mism[:50]],
        "argmax_agree": arg_ok, "argmax_flips": flips,
        "max_abs_dlogit": max_d, "max_abs_dlogit_at": max_at, "mean_abs_dlogit": sum_d / max(1, n_log),
        "mean_tvd": tvd_sum / max(1, nq), "max_tvd": tvd_max,
        "act_max_abs_dlogit": act_max, "act_max_abs_dprob": act_dp,
        "calibrated_checked": api_n, "calibrated_max_abs_dprob": api_max, "calibrated_argmax_agree": api_arg_ok,
    }


def cmd_compare(args):
    json_out = arg_value(args, "--json", None)
    tier = arg_value(args, "--gate", None)
    weights = arg_value(args, "--weights", None)
    tiers_path = arg_value(args, "--tiers", None)
    allow_id = "--allow-identity-mismatch" in args
    args = [a for a in args if a != "--allow-identity-mismatch"]
    if len(args) < 2:
        sys.exit(__doc__)
    for p in args:
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    cfg = None
    if tier:
        try:
            tiers = parity_gate.load_tiers(tiers_path)
            cfg = parity_gate.tier_config(tiers, tier, weights)
        except ValueError as e:
            sys.exit(str(e))
    ref = parity_gate.load_run(args[0])
    base_id = ident.load_sidecar(args[0])
    results = {}
    refused = failed = False
    for path in args[1:]:
        cli = parity_gate.load_run(path)
        name = os.path.basename(path)
        t = compare_one(ref, cli)
        cand_id = ident.load_sidecar(path)
        t["identity"] = {"baseline": base_id, "candidate": cand_id}
        if tier:
            cfg = parity_gate.tier_config(tiers, tier, weights)
            if tier == "f16-class" and cand_id and parity_gate.activation_class(cand_id) == "quantized":
                # a candidate that quantizes activations takes the activation_quantized numbers (tiers v2)
                try:
                    cfg = parity_gate.tier_config(tiers, tier, weights, activation_quantized=True)
                except ValueError as e:
                    sys.exit(str(e))
            why = parity_gate.check_identity(base_id, cand_id, tier, cfg)
            t["identity_problems"] = why
            if why and not allow_id:
                print("%s: identity does not fit tier %s:\n  %s" % (name, tier, "\n  ".join(why)), flush=True)
                t["gate"] = {"verdict": "refused", "tier": tier}
                refused = True
            else:
                for w in why:
                    print("%s: identity mismatch allowed: %s" % (name, w), flush=True)
                t["gate"] = parity_gate.gate_run(ref, cli, tier, cfg)
                failed |= t["gate"]["verdict"] != "pass"
        results[name] = t
    print("%-28s %6s %6s %11s %11s %11s %9s %9s %10s %12s" % (
        "cli", "match", "mism", "argmax", "max|dlog|", "mean|dlog|", "meanTVD", "maxTVD", "act max|d|", "calib max|dP|"))
    for name, t in results.items():
        print("%-28s %6d %6d %5d/%-5d %11.3e %11.3e %9.2e %9.2e %10.2e %6.1e (%d q)" % (
            name, t["identical_inputs"], t["mismatches"], t["argmax_agree"], t["identical_inputs"], t["max_abs_dlogit"],
            t["mean_abs_dlogit"], t["mean_tvd"], t["max_tvd"], t["act_max_abs_dlogit"], t["calibrated_max_abs_dprob"],
            t["calibrated_checked"]), flush=True)
        for m in t["mismatch_list"][:10]:
            print("   mismatch " + m, flush=True)
        for f in t["argmax_flips"][:10]:
            print("   argmax flip " + f, flush=True)
        g = t.get("gate")
        if g and g["verdict"] != "refused":
            print("   gate %s: %s  (%d compared, %d public answers, %d allowed flips, max error %s, mean |dp| %.3g, act rel %.3g, known %s, input mismatches %s)" % (
                g["tier"], g["verdict"].upper(), g["compared"], g["public_checked"], len(g["allowed_flips"]),
                json.dumps({k: float("%.3g" % v) for k, v in g["max_abs_error"].items()}), g["mean_abs_dp"], g["act_max_rel_dlogit"],
                json.dumps(g["known_differences"]), json.dumps(g["input_mismatches"])), flush=True)
            for f in g["failures"][:20]:
                print("   FAIL " + f, flush=True)
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    if refused:
        sys.exit(2)
    if failed:
        sys.exit(1)


def cmd_cat(args):
    if len(args) < 2:
        sys.exit(__doc__)
    out_path, shards = args[0], args[1:]
    for p in shards:
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    ids = [ident.load_sidecar(p) for p in shards]
    if any(i is None for i in ids):
        sys.exit("a shard has no identity record")
    keep = ("corpus", "model", "runtime", "reference", "api_every")
    for i in ids[1:]:
        for k in keep:
            a, b = dict(ids[0].get(k) or {}) if isinstance(ids[0].get(k), dict) else ids[0].get(k), i.get(k)
            if isinstance(a, dict):
                a.pop("n_items", None)
                b = {kk: vv for kk, vv in (b or {}).items() if kk != "n_items"}
            if a != b:
                sys.exit("shards differ in %s" % k)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in shards:
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    n += 1
    joined = dict(ids[0])
    joined["corpus"] = dict(joined["corpus"], n_items=n)
    joined["shard"] = "joined from %d shards" % len(shards)
    joined["shards"] = [os.path.abspath(p) for p in shards]
    ident.write_sidecar(out_path, joined)
    print("joined %d items from %d shards -> %s" % (n, len(shards), out_path), flush=True)


def main():
    cmds = {"ref": cmd_ref, "cat": cmd_cat, "cli": cmd_cli, "server": cmd_server, "compare": cmd_compare}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        sys.exit(__doc__)
    cmds[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
