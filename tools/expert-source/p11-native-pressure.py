#!/usr/bin/env python3
"""Qualify native warning/critical shrink and normal recovery on a dedicated Mac."""
import argparse
import http.client
import json
import mmap
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time


MIB = 1024**2
METRIC_PREFIX = "llamacpp:engine_memory_"


def system_available():
    output = subprocess.check_output(
        ["/usr/bin/memory_pressure", "-Q"], text=True, timeout=5)
    return int(re.search(r"free percentage: (\d+)%", output)[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-mac", action="store_true",
                        help="confirm this machine is dedicated to the pressure run")
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-load-mib", type=int, required=True)
    parser.add_argument("--min-available-percent", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "darwin" or not args.isolated_mac:
        parser.error("run only inside the dedicated macOS environment with --isolated-mac")
    if args.max_load_mib < 1 or not 1 <= args.min_available_percent <= 99:
        parser.error("load and system-availability limits must be positive")
    if not args.server.is_file() or not args.model.is_file():
        parser.error("server and model files must exist")

    prompt = (Path(__file__).resolve().parents[2] /
              "docs/atomic-turbine-completion.txt").read_text()
    budget = 5120 * MIB
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    def request(path, payload=None, timeout=60):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            conn.request("GET" if payload is None else "POST", path,
                         None if payload is None else json.dumps(payload),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            return response.status, response.read().decode()
        finally:
            conn.close()

    def completion():
        status, body = request("/completion", {
            "prompt": prompt, "temperature": 0, "n_predict": 32, "return_tokens": True,
            "cache_prompt": False, "id_slot": 0,
        })
        if status != 200:
            raise RuntimeError(f"completion returned HTTP {status}")
        value = json.loads(body)
        answer = {key: value[key] for key in ("content", "tokens", "stop_type",
                                              "tokens_evaluated", "tokens_predicted")}
        if answer != {"content": "8", "tokens": [23, 248046], "stop_type": "eos",
                      "tokens_evaluated": 199, "tokens_predicted": 2}:
            raise RuntimeError(f"unexpected Qwen control answer: {answer}")
        return answer

    report = {"passed": False, "samples": [], "max_load_mib": args.max_load_mib,
              "min_available_percent": args.min_available_percent}
    started = time.monotonic()

    def sample(phase, loaded):
        status, body = request("/metrics", timeout=10)
        if status != 200:
            raise RuntimeError(f"metrics returned HTTP {status}")
        gauges = {}
        for line in body.splitlines():
            if line.startswith(METRIC_PREFIX):
                name, value = line.split()
                gauges[name.removeprefix(METRIC_PREFIX)] = int(value)
        available = system_available()
        report["samples"].append({
            "phase": phase, "seconds": time.monotonic() - started, "load_mib": loaded,
            "available_percent": available, "metrics": gauges,
        })
        if gauges["policy_mode"] != 2 or not (
            0 < gauges["budget_used_bytes"] <= gauges["budget_applied_bytes"] <= budget
        ):
            raise RuntimeError("invalid automatic-mode budget snapshot")
        return available, gauges

    command = [
        str(args.server.resolve()), "-m", str(args.model.resolve()), "--device", "MTL0",
        "-c", "512", "-b", "512", "-ub", "32", "-np", "1", "-t", "8", "-tb", "8",
        "-ngl", "99", "-fit", "off", "--engine-budget-mib", "5120", "--engine-memory-auto",
        "--engine-memory-min-mib", "4800", "--cache-ram", "0", "--ctx-checkpoints", "0",
        "--no-warmup", "--metrics", "--host", "127.0.0.1", "--port", str(port),
    ]
    with args.output.with_suffix(".server.log").open("x") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    if request("/health", timeout=2)[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("server did not reach readiness")
            report["before"] = completion()
            available, baseline = sample("baseline", 0)
            if baseline["budget_applied_bytes"] != budget or baseline["pressure_level"] >= 2:
                raise RuntimeError("start from a full budget without existing pressure")

            gauges = baseline
            mappings = []
            loaded = 0
            block = os.urandom(MIB)
            try:
                while True:
                    if gauges["budget_applied_bytes"] < budget:
                        if gauges["pressure_level"] < 2 or (
                            gauges["budget_released_bytes"] <= baseline["budget_released_bytes"]
                        ):
                            raise RuntimeError("shrink observed without native pressure and buffer release")
                        report["shrunk_bytes"] = gauges["budget_applied_bytes"]
                        break
                    if available <= args.min_available_percent or loaded >= args.max_load_mib:
                        raise RuntimeError("load limit reached without a native shrink")
                    step = min(256, args.max_load_mib - loaded)
                    region = mmap.mmap(-1, step * MIB)
                    mappings.append(region)
                    for _ in range(step):
                        region.write(block)
                    loaded += step
                    time.sleep(1.25)
                    available, gauges = sample("load", loaded)
            finally:
                for region in mappings:
                    region.close()

            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                _, gauges = sample("recovery", 0)
                if gauges["pressure_level"] == 1 and gauges["budget_applied_bytes"] == budget:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("native normal event and full budget recovery not observed")
            report["after"] = completion()
            sample("post-completion", 0)
            report["passed"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
