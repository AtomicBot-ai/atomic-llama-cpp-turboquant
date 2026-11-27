#!/usr/bin/env python3
"""Run one Inkling comparison on the Windows ARM64 RTX Spark host."""

import argparse
import json
import runpy
import subprocess
from pathlib import Path

# Use precisely the same archive, greedy request and TTFT definition as on Metal.
bench = runpy.run_path(str(Path(__file__).with_name("bench-inkling-metal.py")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-mib", type=int, help="Fixed engine cap including KV (e.g. 8292 for 8 GiB + 100 MiB)")
    parser.add_argument("--draft", type=Path)
    parser.add_argument("--draft-depth", type=int, choices=(3, 7), help="Default 7 for AD-IQ2_M and 3 for the other profiles")
    parser.add_argument("--reference", type=Path, help="Unlimited result for this model and geometry")
    parser.add_argument("--context", type=int, choices=(1024, 4096), default=4096)
    parser.add_argument("--prefill-rows", type=int, help="Set both batch and microbatch width for the fixed 4096-token context")
    args = parser.parse_args()
    if args.prefill_rows is not None and (args.context != 4096 or not 1 <= args.prefill_rows <= args.context):
        parser.error("--prefill-rows requires the 4096-token context and a width between 1 and 4096")
    if args.budget_mib is not None and args.budget_mib <= 0:
        parser.error("the memory budget must be positive")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    model = str(args.model.resolve(strict=True))
    if reference and (reference['model'] != model or reference['context'] != args.context or
                      reference['budget_mib'] is not None or reference['draft'] is not None):
        parser.error("reference must be the target-only unlimited run for this model and context")
    batch = str(args.prefill_rows or 1024)
    ubatch = "8" if args.context == 1024 else batch
    command = [str(args.server.resolve(strict=True)), "-m", model, "--device", "CUDA0", "-ngl", "all",
               "-c", str(args.context), "-b", batch, "-ub", ubatch, "-np", "1", "-t", "8", "-tb", "8",
               "-fit", "off", "-fa", "on", "-ctk", "q4_0", "-ctv", "q4_0", "--cache-ram", "0",
               "--ctx-checkpoints", "0", "--no-warmup", "--metrics", "--host", "127.0.0.1", "--port", "0"]
    if args.budget_mib:
        command += ["--engine-budget-mib", str(args.budget_mib)]
    if args.draft:
        depth = args.draft_depth or (7 if args.model.stem.endswith("AD-IQ2_M") else 3)
        command += ["-md", str(args.draft.resolve(strict=True)), "-ngld", "all", "--spec-type", "draft-dspark",
                    "--spec-draft-n-max", str(depth), "--spec-draft-n-ubatch", "32", "-ctkd", "q4_0", "-ctvd", "q4_0"]
    log = args.output.with_suffix(".log")
    with log.open("x") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        try:
            url = bench["wait_ready"](process, log)
            prompt = bench["fixed_prompt"](url)[:912 if args.context == 1024 else 4000]
            before = bench["engine_metrics"](url)
            row = bench["measure"](url, prompt)
            after = bench["engine_metrics"](url)
            peak_ws = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", f"(Get-Process -Id {process.pid}).PeakWorkingSet64"], text=True)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait()
    row.update(model=model, context=args.context, budget_mib=args.budget_mib, command=command,
               draft=str(args.draft) if args.draft else None, peak_working_set_bytes=int(peak_ws),
               metrics=after, expert_logical_read_bytes=after.get("expert_logical_read_bytes", 0)
               - before.get("expert_logical_read_bytes", 0))
    if args.draft:
        row["draft_active"] = row["timings"].get("draft_n", 0) > 0
    if reference:
        row["matches_reference"] = row["tokens"] == reference["tokens"]
    args.output.write_text(json.dumps(row, indent=2) + "\n")
    if args.budget_mib:
        cap = args.budget_mib << 20
        if (after.get("budget_requested_bytes") != cap or after.get("budget_applied_bytes") != cap
                or after.get("policy_decisions") != 0 or not 0 < after["budget_high_water_bytes"] <= cap):
            raise RuntimeError(f"Invalid fixed memory budget or managed peak; see {args.output}")
    if args.draft and not row["draft_active"]:
        raise RuntimeError(f"No draft proposals measured; see {args.output}")
    if row.get("matches_reference") is False:
        raise RuntimeError(f"Output differs from the unlimited CUDA reference; see {args.output}")


if __name__ == "__main__":
    main()
