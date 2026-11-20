"""Start a local Flash Next server, measure Handover once, then serve the web chat."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import os
import socket
import time
from urllib.error import URLError
from urllib.request import build_opener, ProxyHandler, Request


def decode(client, base, tokens, count):
    payload = dict(prompt=tokens, n_predict=count, stream=True, return_tokens=True,
                   temperature=0, seed=42, ignore_eos=True, cache_prompt=False, id_slot=0)
    request = Request(base + "/completion", json.dumps(payload).encode(),
                      {"Content-Type": "application/json"})
    output = []
    first = last = None
    timings = {}
    start = time.monotonic()
    with client.open(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            ids = event.get("tokens")
            if ids:
                last = time.monotonic()
                if first is None:
                    if len(ids) != 1:
                        raise RuntimeError("First stream event contained multiple tokens")
                    first = last
                output.extend(ids)
            timings = event.get("timings") or timings
    if len(output) != count or timings.get("prompt_n") != len(tokens) or timings.get("cache_n") != 0:
        raise RuntimeError("Unexpected token count or cached prompt")
    return output, first - start, (count - 1) / (last - first)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("server", "model", "draft"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--budget-gib", type=int, choices=range(8, 25, 2), default=20)
    parser.add_argument("--measure-only", action="store_true", help="Exit after the measured request")
    args = parser.parse_args()
    fixture = Path(__file__).with_name("flash-next-handover.json")
    record = json.loads(fixture.read_text())
    gold = record["output_ids"]
    for path in (args.server, args.model, args.draft):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    cmd = [str(args.server.resolve()), "-m", str(args.model), "-md", str(args.draft),
           "--spec-type", "draft-dflash", "--spec-draft-n-max", "3", "--spec-draft-n-ubatch", "128",
           "--engine-budget-mib", str(args.budget_gib * 1024), "-fit", "off", "-c", "4096",
           "-b", "4096", "-ub", "8", "-np", "1", "-ctk", "bf16", "-ctv", "bf16", "-ngl", "99",
           "-ngld", "99", "-t", "8", "-tb", "8", "--cache-ram", "0", "--ctx-checkpoints", "0",
           "--no-warmup", "--metrics", "--ui", "--alias", "flash-next", "--host", "127.0.0.1",
           "--port", str(args.port)]
    base = f"http://127.0.0.1:{args.port}"
    client = build_opener(ProxyHandler({}))
    process = subprocess.Popen(cmd, env={**os.environ, "LLAMA_EXPERIMENTAL_BUDGET_DFLASH2": "1"})
    try:
        deadline = time.monotonic() + 180
        while process.poll() is None and time.monotonic() < deadline:
            try:
                with client.open(base + "/health", timeout=2) as response:
                    if json.load(response).get("status") == "ok":
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(0.25)
        else:
            raise RuntimeError("Server failed to become ready")
        decode(client, base, record["warmup_ids"], 8)
        ids, ttft, tps = decode(client, base, record["input_ids"], 64)
        matches = sum(a == b for a, b in zip(ids, gold))
        print(f"Handover (912 in, 64 out): TTFT {ttft:.2f} s | live decode {tps:.2f} tok/s | {matches}/64 IDs matched", flush=True)
        if matches != 64:
            raise RuntimeError("Reference output mismatch")
        if not args.measure_only:
            print(f"Web chat: {base}/   API: {base}/v1   Press Ctrl-C to stop.", flush=True)
            return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
