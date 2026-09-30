"""Parity of llama-laya-cli against the PyTorch reference (the `laya` package) on decision items.

    <laya-python> tests/laya/verify_reference.py ref <checkpoint-dir> <items.jsonl> <ref.jsonl> [--english] [--shard I/N] [--threads T]
    python3       tests/laya/verify_reference.py cli <llama-laya-cli> <model.gguf> <ref.jsonl> <cli.jsonl> [--jobs J] [-t T] [--kernels K]
    python3       tests/laya/verify_reference.py server <llama-server> <model.gguf> <ref.jsonl> <server.jsonl> [-t T] [--kernels K]
    python3       tests/laya/verify_reference.py compare <ref.jsonl> <cli.jsonl> [<cli.jsonl> ...] [--json out.json]

items.jsonl has one {"id", "cat", "state", "questions"} object per line (the laya-eval item set).

ref      runs the reference on the CPU in fp32 through the Agent's own request path (_check_question,
         _to_internal, _encode_state, collate_items, DecisionModel.forward) and records per question the
         input_ids, marker positions, raw scorer logits and raw act logits. Every 5th item also goes
         through agent.system_one: its calibrated answers check the temperatures (temperature_by_options
         buckets included) end to end. --english keeps only the items whose letters are all ASCII.
         Shards (--shard I/N) write separate files; `cat` them for cli / compare.
cli      runs llama-laya-cli on the same items (one process per item, -t 1 by default), with
         ensure_ascii JSON inputs so the CLI sees exactly the Python objects the reference saw.
         The CLI packs all questions of an item into one graph (block-diagonal mask).
server   runs the same items through one llama-server --decision --decision-debug process over HTTP
         (POST /v1/systemone, one request at a time), i.e. the code path the server really runs: plan
         `sequential` (one graph per question), the server's default threads (the performance cores)
         unless -t, and the kernels of --kernels (default: the server's own default, `auto`). Records
         debug.tokens / debug.logits per question; the server has no act head output and derives the
         marker positions from the tokens, so compare checks the input ids only and skips act metrics.
         /props.decision.plan goes to <server.jsonl>.props.json.
compare  matches inputs first (input_ids and marker_pos equal byte for byte), then reports per CLI file:
         argmax agreement, max / mean |dlogit|, mean / max TVD of softmax(logits), act-head |d|, and
         the calibrated probabilities of the CLI answers against the reference system_one answers.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor


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
                if i % 5 == 0:
                    rec["api"] = agent.system_one(it["state"], it["questions"])["answers"]
            except Exception as e:  # recorded: the CLI must refuse the same items
                rec["error"] = "%s: %s" % (type(e).__name__, e)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0:
                print("shard %s %d/%d %.0fs" % (shard, i, len(items), time.time() - t0), flush=True)
    print("shard %s done: %d items %.0fs" % (shard, len(items), time.time() - t0), flush=True)


def cmd_cli(args):
    jobs = int(arg_value(args, "--jobs", "8"))
    threads = arg_value(args, "-t", "1")
    kernels = arg_value(args, "--kernels", None)
    if len(args) != 4:
        sys.exit(__doc__)
    cli, model, ref_path, out_path = args
    for p in (cli, model, ref_path):
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    items = [json.loads(line) for line in open(ref_path, encoding="utf-8")]
    tmpdir = tempfile.mkdtemp(prefix="laya_cli_")
    env = dict(os.environ, LC_ALL="C")

    def run(it):
        path = os.path.join(tmpdir, it["id"] + ".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"state": it["state"], "questions": it["questions"]}, f, ensure_ascii=True)
        cmd = [cli, "-m", model, "-f", path, "-t", threads] + (["--kernels", kernels] if kernels else [])
        p = subprocess.run(cmd, capture_output=True, env=env)
        os.unlink(path)
        rec = {"id": it["id"]}
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", "replace").strip().splitlines()
            rec["error"] = "rc=%d: %s" % (p.returncode, err[-1] if err else "")
            return rec
        d = json.loads(p.stdout.decode("utf-8"))
        rec["per_question"] = d["per_question"]
        rec["answers"] = d["answers"]
        return rec

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out, ThreadPoolExecutor(jobs) as ex:
        for i, rec in enumerate(ex.map(run, items)):
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print("cli %d/%d %.0fs" % (i, len(items), time.time() - t0), flush=True)
    os.rmdir(tmpdir)
    print("cli done: %d items %.0fs -> %s" % (len(items), time.time() - t0, out_path), flush=True)


def cmd_server(args):
    import socket
    import urllib.error
    import urllib.request

    threads = arg_value(args, "-t", None)
    kernels = arg_value(args, "--kernels", None)
    if len(args) != 4:
        sys.exit(__doc__)
    server, model, ref_path, out_path = args
    for p in (server, model, ref_path):
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    items = [json.loads(line) for line in open(ref_path, encoding="utf-8")]
    max_q = max(len(it["questions"]) for it in items)

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    base = "http://127.0.0.1:%d" % port
    cmd = [server, "--decision", "--decision-debug", "-m", model, "--host", "127.0.0.1", "--port", str(port),
           "--decision-max-items", str(max(16, max_q))]
    cmd += (["-t", threads] if threads else []) + (["--decision-kernels", kernels] if kernels else [])
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
        with open(out_path + ".props.json", "w", encoding="utf-8") as f:
            json.dump({"cmd": cmd, "plan": props["plan"], "memory": props.get("memory")}, f, indent=1)
        print("plan %s, kernels %s, %d threads" % (props["plan"]["name"], props["plan"]["kernels"], props["plan"]["n_threads"]), flush=True)
        with open(out_path, "w", encoding="utf-8") as out:
            for i, it in enumerate(items):
                status, body = post({"state": it["state"], "questions": it["questions"]})
                rec = {"id": it["id"]}
                if status != 200:
                    rec["error"] = "http %d: %s" % (status, json.dumps(body, ensure_ascii=False)[:300])
                else:
                    rec["per_question"] = {qid: {"input_ids": a["debug"]["tokens"], "raw_logits": a["debug"]["logits"]}
                                           for qid, a in body["answers"].items()}
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
        for qid in r["questions"]:
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
            elif cq["input_ids"] != rq["input_ids"] or cq.get("marker_pos", rq["marker_pos"]) != rq["marker_pos"]:
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
            flips.append("%s/%s ref gap %.4f max|dlogit| %.4f" % (iid, qid, srt[0] - srt[1], max(d)))
        pa, pb = softmax(a), softmax(b)
        tvd = 0.5 * sum(abs(x - y) for x, y in zip(pa, pb))
        tvd_sum += tvd
        tvd_max = max(tvd_max, tvd)
        if "act_raw_logits" in cq:  # the server has no act head output
            act_max = max(act_max, max(abs(x - y) for x, y in zip(rq["act_raw_logits"], cq["act_raw_logits"])))
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
    if len(args) < 2:
        sys.exit(__doc__)
    ref = {}
    for line in open(args[0], encoding="utf-8"):
        d = json.loads(line)
        ref[d["id"]] = d
    results = {}
    for path in args[1:]:
        cli = {json.loads(line)["id"]: json.loads(line) for line in open(path, encoding="utf-8")}
        results[os.path.basename(path)] = compare_one(ref, cli)
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
    if json_out:
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("ref", "cli", "server", "compare"):
        sys.exit(__doc__)
    {"ref": cmd_ref, "cli": cmd_cli, "server": cmd_server, "compare": cmd_compare}[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
