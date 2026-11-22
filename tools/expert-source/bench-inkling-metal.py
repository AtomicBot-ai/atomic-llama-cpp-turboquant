#!/usr/bin/env python3
"""Measure stock llama.cpp and budgeted Inkling DSpark on a fixed 4000/64 request."""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
BUDGETS = tuple(range(8, 21, 2))
PROMPT_SHA256 = "a7515e85dd84a09fba80c6c2436c36be8bef0097eeb825e5e72ae52c475a81b0"
HTTP = build_opener(ProxyHandler({}))


def server_command(args, budget):
    command = [str(args.stock_server if budget is None else args.server), "-m", str(args.model),
               "-fit", "off", "-c", "4096", "-b", "1024", "-ub", "1024", "-np", "1", "-fa", "on",
               "-ctk", "q4_0", "-ctv", "q4_0", "-ngl", "all", "-t", "8", "-tb", "8",
               "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-warmup", "--metrics",
               "--host", "127.0.0.1", "--port", "0"]
    if budget is None:
        return command + ["--jinja"]
    return command + ["-md", str(args.draft), "--spec-type", "draft-dspark", "--spec-draft-n-max", "3",
                      "--spec-draft-n-ubatch", "32", "-ctkd", "q4_0", "-ctvd", "q4_0", "-ngld", "99",
                      "--engine-budget-mib", str(1024 * budget + 100), "--temp", "0",
                      "--ui", "--alias", "inkling-2-mini"]


def wait_ready(process, log):
    deadline = time.monotonic() + 300
    while process.poll() is None and time.monotonic() < deadline:
        match = re.search(r"listening on (http://127\.0\.0\.1:\d+)\b", log.read_text(errors="replace"))
        if match:
            url = match[1]
            try:
                with HTTP.open(url + "/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        return url
            except (URLError, TimeoutError):
                pass
        time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready (exit {process.poll()}). See {log}")


def post(url, payload):
    return HTTP.open(Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=900)


def fixed_prompt(url):
    # The original benchmark repeated the archive without a separator and cropped native token IDs.
    text = (ROOT / "docs/atomic-engine-qwen38-context.txt").read_text(encoding="utf-8") * 2
    with post(url + "/tokenize", dict(content=text, add_special=False, parse_special=False)) as response:
        tokens = json.load(response)["tokens"][:4000]
    if len(tokens) != 4000 or hashlib.sha256(struct.pack("<4000i", *tokens)).hexdigest() != PROMPT_SHA256:
        raise ValueError("Tokenized input differs from the original 4000-token Inkling fixture")
    return tokens


def measure(url, prompt):
    payload = dict(prompt=prompt, n_predict=64, temperature=0, seed=42, ignore_eos=True,
                   cache_prompt=False, id_slot=0, stream=True, return_tokens=True)
    tokens = []
    timings = {}
    first = None
    start = time.perf_counter()
    with post(url + "/completion", payload) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("tokens"):
                if first is None:
                    first = time.perf_counter()
                tokens.extend(event["tokens"])
            if event.get("stop"):
                timings = event["timings"]
                break
    if (first is None or len(tokens) != 64 or timings.get("predicted_n") != 64
            or timings.get("prompt_n") != len(prompt) or timings.get("cache_n") != 0):
        raise RuntimeError(f"Incomplete or cached benchmark response: {len(tokens)} tokens, {timings}")
    return dict(ttft_s=first - start, decode_tps=timings["predicted_per_second"], timings=timings, tokens=tokens)


def run(args, budget, directory, env):
    log = directory / ("stock.log" if budget is None else f"atomic-{budget}gib.log")
    command = server_command(args, budget)
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            url = wait_ready(process, log)
            row = measure(url, fixed_prompt(url))
            if budget is not None:
                with HTTP.open(url + "/metrics", timeout=10) as response:
                    metrics = response.read().decode()
                counters = {k: int(float(v)) for k, v in re.findall(
                    r"^llamacpp:engine_memory_budget_(\w+)_bytes\s+(\S+)$", metrics, re.M)}
                cap = (budget * 1024 + 100) * 2**20
                if counters["applied"] != cap or counters["high_water"] > cap:
                    raise RuntimeError(f"Unexpected engine budget counters: {counters}")
                row.update(budget_mib=cap // 2**20, peak_managed_bytes=counters["high_water"])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    return dict(budget_gib=budget, command=command, log=log.name, **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "build-metal/bin/llama-server")
    parser.add_argument("--stock-server", type=Path, default=ROOT / ".work/inkling-stock/build/bin/llama-server")
    parser.add_argument("--model", type=Path, default=ROOT / ".work/models/Inkling-2-Mini-AD-Q4_K.gguf")
    parser.add_argument("--draft", type=Path, default=ROOT / ".work/models/dspark-Inkling-2-Mini-Q8_0-folded-final.gguf")
    parser.add_argument("--budgets", type=int, nargs="+", choices=BUDGETS, default=BUDGETS,
                        help="Atomic Engine budgets in GiB, each with an extra 100 MiB for KV (default: 8 to 20, step 2)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".work/inkling-bench",
                        help="parent directory for a new result directory per invocation")
    args = parser.parse_args()
    for key in ("server", "stock_server", "model", "draft"):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_file() or (key.endswith("server") and not os.access(path, os.X_OK)):
            parser.error(f"Check --{key.replace('_', '-')}: {path}")
        setattr(args, key, path)
    if args.stock_server.samefile(args.server):
        parser.error("Use separate stock and Atomic Engine server binaries")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    results = dict(prompt_sha256_i32le=PROMPT_SHA256, rows=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "ATOMIC_"))}
    print(f"Results: {directory}\n{'Profile':<18} {'TTFT (s)':>9} {'Decode (tok/s)':>15} {'vs stock':>10}", flush=True)
    for budget in (None, *dict.fromkeys(args.budgets)):
        label = "stock llama.cpp" if budget is None else f"{budget} GiB + KV"
        row = run(args, budget, directory, env)
        reference = results["rows"][0] if results["rows"] else row
        row["speedup"] = row["decode_tps"] / reference["decode_tps"]
        row["matches_stock"] = row["tokens"] == reference["tokens"]
        results["rows"].append(row)
        temp = directory / "results.tmp"
        temp.write_text(json.dumps(results, indent=2) + "\n")
        temp.replace(directory / "results.json")
        if not row["matches_stock"]:
            raise ValueError(f"{label}: output differs from stock. See {directory}")
        print(f"{label:<18} {row['ttft_s']:9.2f} {row['decode_tps']:15.2f} {row['speedup']:9.2f}x", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(str(error))
