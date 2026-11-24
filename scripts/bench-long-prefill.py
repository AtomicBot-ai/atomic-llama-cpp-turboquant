#!/usr/bin/env python3
"""Long-input prefill benchmark for the Inkling 2 Mini llama-server.

Measures prompt-processing (prefill) throughput, TTFT and decode speed at
several prompt sizes, with cache reuse disabled so every run pays the full
prefill cost. A warm-up request runs first (Metal pipeline compile + weight
page-in); the measured run follows.

Usage: bench-long-prefill.py <port> [label] [sizes]
  sizes: comma list of token targets, default 2500,16500,40000
"""
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

PORT = sys.argv[1] if len(sys.argv) > 1 else "8082"
LABEL = sys.argv[2] if len(sys.argv) > 2 else f"port {PORT}"
SIZES = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else
                          ["2500", "16500", "40000"])]

# ~33 tokens per repetition
PARA = (
    "The quick brown fox jumps over the lazy dog near the river bank while "
    "the old clock on the wall keeps ticking and the kettle begins to boil "
    "for the evening tea that the family shares after a long day of work "
    "in the garden and the orchard behind the stone house on the hill. "
)
QUESTION = "\n\nBased on the passage above, answer in one sentence: what does the family do after work?"


def ask(prompt, max_tokens=128, tag=""):
    body = {
        "model": "inkling-2-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code} {e.read()[:120]!r}"
    dt = time.monotonic() - t0
    t = d.get("timings") or {}
    u = d.get("usage") or {}
    return {
        "wall": dt,
        "ttft": (t.get("prompt_ms", 0) or 0) / 1000,
        "prompt_n": t.get("prompt_n") or u.get("prompt_tokens"),
        "prompt_tps": t.get("prompt_per_second"),
        "gen_n": t.get("predicted_n") or u.get("completion_tokens"),
        "gen_tps": t.get("predicted_per_second"),
        "finish": d["choices"][0].get("finish_reason"),
    }, None


def server_pid():
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
                         capture_output=True, text=True).stdout.strip()
    return out.splitlines()[0] if out else None


def footprint(pid):
    out = subprocess.run(["footprint", pid], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "Footprint:" in line:
            num, unit = line.split("Footprint:")[1].split()[:2]
            return float(num) * (1 if unit == "GB" else 1 / 1024)
    return None


print(f"=== {LABEL} (port {PORT}) ===")
pid = server_pid()
results = []
for size in SIZES:
    reps = max(1, round(size / 33))
    prompt = (PARA * reps)[: size * 4] + QUESTION
    warm, err = ask(prompt, tag="warm")
    if err:
        print(f"{size:>6} tok: warm-up FAILED {err}")
        continue
    r, err = ask(prompt, tag="meas")
    if err:
        print(f"{size:>6} tok: FAILED {err}")
        continue
    results.append((size, r))
    print(f"{size:>6} tok: prompt {r['prompt_n']:>6} tok @ {r['prompt_tps']:7.1f} t/s "
          f"({r['prompt_n']/r['prompt_tps']:5.1f}s) | TTFT {r['ttft']:6.1f}s | "
          f"gen {r['gen_n']:>3} @ {r['gen_tps']:5.1f} t/s | wall {r['wall']:6.1f}s | {r['finish']}")
if pid:
    fp = footprint(pid)
    if fp:
        print(f"footprint: {fp:.2f} GiB")
print("DONE")
