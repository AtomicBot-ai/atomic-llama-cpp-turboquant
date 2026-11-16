#!/usr/bin/env python3
"""Build the focused Qwen4Exp numerical/window driver against current libraries.

Build llama-server and llama-expert-infer first. No model/context source patches
are required: QWEN_WINDOW_RUNTIME=1 enables the fork's idle-only window control.
The output executable accepts MODEL native|stream PROMPT_FILE OUTPUT_STEM.
"""
from pathlib import Path
import argparse
import json
import os
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", type=Path, default=Path("build-metal"))
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
build, work = args.build.resolve(), args.output.resolve()
if work in (build, root):
    parser.error("output must be a separate diagnostic directory")
work.mkdir(parents=True, exist_ok=True)
env = os.environ | {"DEVELOPER_DIR": "/Library/Developer/CommandLineTools"}
entries = json.loads((build / "compile_commands.json").read_text())
entry = next(e for e in entries if e["file"] == str(root / "tools/expert-source/inference.cpp"))
command = shlex.split(entry["command"])
obj = work / "compare.o"
command[command.index("-o") + 1] = str(obj)
command[command.index("-c") + 1] = str(root / "tools/expert-source/qwen38-window-compare.cpp")
subprocess.run(command, cwd=entry["directory"], env=env, check=True)
directory = build / "tools/expert-source"
link = shlex.split((directory / "CMakeFiles/llama-expert-infer.dir/link.txt").read_text())
command = [str(obj) if a.endswith("/inference.cpp.o") else str(work / "window-compare")
           if a == "../../bin/llama-expert-infer" else a for a in link]
subprocess.run(command, cwd=directory, env=env, check=True)
print(work / "window-compare")
