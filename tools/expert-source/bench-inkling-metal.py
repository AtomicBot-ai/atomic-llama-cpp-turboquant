#!/usr/bin/env python3
"""Measure stock llama.cpp and budgeted Inkling DSpark on a fixed 4000/64 request."""

import argparse
import copy
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
MODEL_DIR = ROOT / ".work/models"
DEFAULT_DEPTHS = {"IQ2_M": 7, "IQ3_S": 3, "Q4_K": 3}
BUDGETS = tuple(range(8, 21, 2))
PROMPT_SHA256 = "a7515e85dd84a09fba80c6c2436c36be8bef0097eeb825e5e72ae52c475a81b0"
HANDOVER_SHA256 = "52ed46c1daf1ccecfd766dfc4e40cac4df57b934c721df82158ae76bae7e9ce5"
HTTP = build_opener(ProxyHandler({}))


def server_command(args, budget):
    command = [str(args.stock_server if budget is None else args.server), "-m", str(args.model),
               "-fit", "off", "-c", "4096", "-b", "1024", "-ub", "1024", "-np", "1", "-fa", "on",
               "-ctk", "q4_0", "-ctv", "q4_0", "-ngl", "all", "-t", "8", "-tb", "8",
               "--cache-ram", "0", "--ctx-checkpoints", "0", "--no-warmup", "--metrics",
               "--host", "127.0.0.1", "--port", "0"]
    if budget is None:
        return command + ["--jinja"]
    if getattr(args, "draft_depth", None) is not None:
        command += ["--spec-draft-n-max", str(args.draft_depth)]
    return command + ["-md", str(args.draft), "--spec-type", "draft-dspark",
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


def handover_prompt():
    path = Path(__file__).with_name("inkling-handover-4000.json")
    tokens = json.loads(path.read_text())
    if len(tokens) != 4000 or hashlib.sha256(struct.pack("<4000i", *tokens)).hexdigest() != HANDOVER_SHA256:
        raise ValueError("Handover token IDs differ from the original 4000-token fixture")
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


def engine_metrics(url):
    with HTTP.open(url + "/metrics", timeout=10) as response:
        return {k: int(v) for k, v in re.findall(
            r"^llamacpp:engine_memory_(\w+)\s+(\d+)$", response.read().decode(), re.M)}


def run(args, budget, directory, env):
    log = directory / ("stock.log" if budget is None else f"atomic-{budget}gib.log")
    command = server_command(args, budget)
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
        try:
            url = wait_ready(process, log)
            prompt = handover_prompt() if getattr(args, "prompt", "archive") == "handover" else fixed_prompt(url)
            before = engine_metrics(url) if budget is not None else {}
            row = measure(url, prompt)
            if budget is not None:
                counters = engine_metrics(url)
                cap = (budget * 1024 + 100) * 2**20
                peak = counters["budget_high_water_bytes"]
                if counters["budget_applied_bytes"] != cap or peak > cap:
                    raise RuntimeError(f"Unexpected engine budget counters: {counters}")
                row.update(budget_mib=cap // 2**20, peak_managed_bytes=peak)
                for key in ("expert_logical_read_bytes", "expert_cache_misses"):
                    if key in before and key in counters:
                        row[key] = counters[key] - before[key]
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
    parser.add_argument("--model", type=Path, help="Run one target GGUF (default: all three published Metal profiles)")
    parser.add_argument("--draft", type=Path, default=ROOT / ".work/models/atomicchat-dspark-3095098/dspark-Inkling-2-Mini-Q8_0.gguf")
    parser.add_argument("--prompt", choices=("archive", "handover"), default="archive")
    parser.add_argument("--draft-depth", type=int, choices=(3, 4, 7), help="DSpark proposals per round (default: model profile)")
    parser.add_argument("--compare-draft", type=Path, help="Compare another draft against --draft in two opposite orders")
    parser.add_argument("--budgets", type=int, nargs="+", choices=BUDGETS, default=BUDGETS,
                        help="Atomic Engine budgets in GiB, each with an extra 100 MiB for KV (default: 8 to 20, step 2)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".work/inkling-bench",
                        help="parent directory for a new result directory per invocation")
    args = parser.parse_args()
    for key in ("server", "stock_server", "draft"):
        path = getattr(args, key).expanduser().resolve()
        if not path.is_file() or (key.endswith("server") and not os.access(path, os.X_OK)):
            parser.error(f"Check --{key.replace('_', '-')}: {path}")
        setattr(args, key, path)
    if args.compare_draft:
        args.compare_draft = args.compare_draft.expanduser().resolve()
        if not args.compare_draft.is_file() or args.compare_draft.samefile(args.draft):
            parser.error("--compare-draft requires a different local draft GGUF")
    if args.stock_server.samefile(args.server):
        parser.error("Use separate stock and Atomic Engine server binaries")
    if args.model:
        models = {args.model.stem: args.model.expanduser().resolve()}
    else:
        quants = ("IQ2_M",) if args.compare_draft else DEFAULT_DEPTHS
        models = {quant: MODEL_DIR / f"Inkling-2-Mini-AD-{quant}.gguf" for quant in quants}
    for path in models.values():
        if not path.is_file():
            parser.error(f"Missing target GGUF: {path}")
    matrix = len(models) > 1
    if matrix:
        try:
            from inkling_matrix import write_matrix
        except ModuleNotFoundError as error:
            if error.name == "PIL":
                parser.error("Install Pillow for the matrix image: python3 -m pip install Pillow")
            raise
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run-", dir=args.output_dir)).resolve()
    digest = HANDOVER_SHA256 if args.prompt == "handover" else PROMPT_SHA256
    results = dict(prompt=args.prompt, prompt_sha256_i32le=digest, rows=[])
    env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_", "GGML_", "ATOMIC_"))}
    print(f"Results: {directory}", flush=True)
    for name, model in models.items():
        model_dir = directory / name if matrix else directory
        model_dir.mkdir(exist_ok=True)
        run_args = copy.copy(args)
        run_args.model = model
        print(f"\n{name}", flush=True)
        measure_model(run_args, directory, model_dir, name, results, env)
    if matrix:
        write_matrix(directory, results, list(models), list(dict.fromkeys(args.budgets)),
                     DEFAULT_DEPTHS, args.draft_depth, args.draft.name)


def measure_model(args, directory, model_dir, quant, results, env):
    print(f"{'Profile':<18} {'TTFT (s)':>9} {'Decode (tok/s)':>15} {'vs stock':>10}", flush=True)
    draft_add = os.environ.get("LLAMA_EXPERIMENTAL_INKLING_W2_PARALLEL_ADD")
    cases = [(None, "stock", args.draft, 1)]
    for budget in dict.fromkeys(args.budgets):
        if args.compare_draft:
            for rep, order in enumerate((("comparison", "primary"), ("primary", "comparison")), 1):
                for name in order:
                    cases.append((budget, name, args.draft if name == "primary" else args.compare_draft, rep))
        else:
            cases.append((budget, "primary", args.draft, 1))
    comparison_tokens = None
    reference = None
    for budget, name, draft, rep in cases:
        label = "stock llama.cpp" if budget is None else f"{budget} GiB + KV"
        folder = model_dir
        if budget is not None and args.compare_draft:
            label += f" {name} #{rep}"
            folder = model_dir / f"{budget}-{name}-{rep}"
            folder.mkdir()
        run_args = copy.copy(args)
        run_args.draft = draft
        run_env = env
        if budget is not None and draft_add is not None:
            run_env = dict(env, LLAMA_EXPERIMENTAL_INKLING_W2_PARALLEL_ADD=draft_add)
        row = run(run_args, budget, folder, run_env)
        row.update(quant=quant, drafter=name, repetition=rep, log=str((folder / row["log"]).relative_to(directory)))
        if budget is None:
            reference = row
        row["speedup"] = row["decode_tps"] / reference["decode_tps"]
        row["matches_stock"] = row["tokens"] == reference["tokens"]
        if args.compare_draft and budget is not None:
            if comparison_tokens is None:
                comparison_tokens = row["tokens"]  # First Atomic run uses --compare-draft.
            row["matches_comparison"] = row["tokens"] == comparison_tokens
        results["rows"].append(row)
        temp = directory / "results.tmp"
        temp.write_text(json.dumps(results, indent=2) + "\n")
        temp.replace(directory / "results.json")
        if row.get("matches_comparison") is False:
            raise ValueError(f"{label}: output differs from the first comparison draft run. See {directory}")
        if not row["matches_stock"] and not args.compare_draft:
            raise ValueError(f"{label}: output differs from stock. See {directory}")
        note = ""
        if not row["matches_stock"]:
            first = next(i for i, (a, b) in enumerate(zip(row["tokens"], reference["tokens"]), 1) if a != b)
            note = f"  differs from stock at token {first}"
        print(f"{label:<18} {row['ttft_s']:9.2f} {row['decode_tps']:15.2f} {row['speedup']:9.2f}x{note}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        sys.exit(str(error))
