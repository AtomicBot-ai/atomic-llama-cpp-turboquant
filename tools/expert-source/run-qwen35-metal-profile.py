#!/usr/bin/env python3
"""Launch the fixed-budget Qwen3.6-35B-A3B profile in the regular Metal server."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "docs" / "atomic-engine-qwen35-fixed-grid.json"


def compact_draft_for(model: Path):
    record = ROOT / "docs" / "atomic-engine-qwen36-low-budget-draft.json"
    if not record.is_file():
        return None
    reference = json.loads(record.read_text())["compact_draft"]
    local = ROOT / reference["path"]
    for path in dict.fromkeys((model.parent / local.name, local)):
        if path.is_file() and path.stat().st_size == reference["bytes"]:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() == reference["sha256"]:
                return path
    return None


def main():
    record = json.loads(GRID.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gib", type=int, choices=[*range(8, 25), 26])
    parser.add_argument("--model", type=Path, default=Path(record["local_model"]))
    parser.add_argument("--draft", type=Path, help="override the adjacent qwen35-dflash2.gguf")
    parser.add_argument("--server", type=Path, default=ROOT / "build-metal/bin/llama-server")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fast-prefill", action=argparse.BooleanOptionalAction, default=None,
                        help="use the latency-tuned Qwen profile (default at 8–24 GiB). Greedy outputs can differ")
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args()
    command = [str(args.server), "-m", str(args.model), *record["server_args"],
               "--engine-budget-mib", str(args.gib * 1024), "--port", str(args.port)]
    fast_default = (args.gib <= 24 and os.environ.get("LLAMA_QWEN35_FAST_PREFILL", "1") == "1")
    fast = fast_default if args.fast_prefill is None else args.fast_prefill
    draft = args.draft
    if (draft is None and fast and args.gib == 8 and
            args.model.name == Path(record["local_model"]).name and
            os.environ.get("LLAMA_EXPERIMENTAL_BUDGET_DFLASH2") != "0"):
        draft = compact_draft_for(args.model)
    if draft is not None:
        command.extend(["-md", str(draft)])
    settings = {}
    if args.fast_prefill is not None:
        settings["LLAMA_QWEN35_FAST_PREFILL"] = "1" if fast else "0"
    if fast and args.gib == 8 and draft and os.environ.get("LLAMA_EXPERIMENTAL_BUDGET_DFLASH2") != "0":
        settings["LLAMA_EXPERIMENTAL_BUDGET_DFLASH2"] = "1"
    if args.fast_prefill and args.gib == 26:
        settings["LLAMA_QWEN35_PARTIAL_WIDE_PREFILL"] = "auto"
        settings["LLAMA_QWEN35_GQA8"] = "1"
        settings["LLAMA_QWEN35_RESIDENT_VERIFY"] = "1"
    if args.print_command:
        print(shlex.join([*(f"{k}={v}" for k, v in settings.items()), *command]))
        return
    os.execvpe(command[0], command, {**os.environ, **settings})


if __name__ == "__main__":
    main()
