"""Build an isolated Flash Next verifier and time known prefixes on the handover fixture.

Run from an idle Metal host with an up-to-date static Release build (Unix Makefiles,
compile_commands.json, llama-server). The verifier and any experimental store are rebuilt in a temporary directory.
"""
import argparse
import http.client
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"


def build_verifier(build, tmp, patch, compare_pool=False):
    sources = {"tools/server/server-context.cpp": "tools/server/libserver-context.a"}
    if compare_pool:
        sources["src/llama-expert-store.cpp"] = "src/libllama.a"
    for path in sources:
        copy = tmp / path
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, copy)
    subprocess.run(["patch", "-p1"], input=patch, text=True, cwd=tmp, check=True)
    commands = json.loads((build / "compile_commands.json").read_text())
    binary = tmp / "flash-verify-server"
    replacements = {"llama-server": binary}
    for path, library in sources.items():
        source = ROOT / path
        row = next(row for row in commands if Path(row["file"]) == source)
        args = shlex.split(row["command"])
        obj = tmp / Path(args[args.index("-o") + 1]).name
        args[args.index("-o") + 1] = str(obj)
        args[args.index("-c") + 1] = str(tmp / path)
        subprocess.run([*args, "-I", source.parent], cwd=row["directory"], check=True)
        archive = tmp / Path(library).name
        shutil.copyfile(build / library, archive)
        subprocess.run(["ar", "r", archive, obj], check=True)
        subprocess.run(["ranlib", archive], check=True)
        replacements[archive.name] = archive

    link_dir = build / "tools/server"
    link = shlex.split((link_dir / "CMakeFiles/llama-server.dir/link.txt").read_text())
    subprocess.run([replacements.get(Path(arg).name, arg) for arg in link], cwd=link_dir, check=True)
    return binary


def measure(binary, tmp, width, reference, fixtures, model, *, size_lru=False):
    args = reference["server_args"].copy()
    args[args.index("-m") + 1] = str(model)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}/"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("LLAMA_", "ATOMIC_", "GGML_", "QMETAL_", "MTL_CAPTURE"))}
    env.pop("FLASH_VERIFY_SIZE_LRU", None)
    if size_lru:
        env["FLASH_VERIFY_SIZE_LRU"] = "1"
    env["FLASH_VERIFY_WIDTH"] = str(width)

    def completion(ids, count):
        body = dict(prompt=ids, n_predict=count, temperature=0, seed=42, ignore_eos=True,
                    cache_prompt=False, return_tokens=True, stream=True, id_slot=0)
        req = urllib.request.Request(url + "completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with opener.open(req, timeout=1200) as response:
            return response.read()

    log = tmp / f"width-{width}.log"
    with log.open("w") as fd:
        proc = subprocess.Popen([binary, *args, "--port", str(port)], env=env,
                                stdout=fd, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(log.read_text()[-2000:])
                try:
                    with opener.open(url + "health", timeout=1):
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(0.15)
            else:
                raise TimeoutError("verifier startup")
            warmup = completion(fixtures["archive"]["ids"][-8:], 8)
            events = [json.loads(line[5:]) for line in warmup.splitlines()
                      if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]
            if sum(len(event.get("tokens", [])) for event in events) != 8:
                raise RuntimeError("incomplete eight-token warmup")
            try:
                completion(fixtures["handover"]["ids"], 64)
            except (http.client.IncompleteRead, http.client.RemoteDisconnected):
                pass  # The probe prints its result and exits in the decode callback.
            if proc.wait(timeout=15):
                raise RuntimeError(log.read_text()[-2000:])
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
    marker = "FLASH_VERIFY_RESULT "
    line = next(line for line in log.read_text().splitlines() if line.startswith(marker))
    result = json.loads(line[len(marker):])
    blocks = result["blocks"]
    if (result["width"] != width or result["predicted_ids"] != reference["reference_output_ids"]
            or sum(block["rows"] for block in blocks) != 63
            or result["peak_managed_bytes"] > 8 * 1024**3):
        raise ValueError("width, greedy IDs, position count or budget mismatch")
    result["target_argmax_tps"] = 63e6 / sum(block["with_argmax_us"] for block in blocks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build-metal")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new result JSON")
    parser.add_argument("--compare-pool", action="store_true",
                        help="compare stock and size-aware eviction at width eight in ABBA order")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output already exists")
    reference = json.loads((DOCS / "atomic-engine-flash-next-verify-ceiling-8gib.json").read_text())
    fixtures = json.loads((DOCS / "atomic-engine-flash-next-default-8-16gib.json").read_text())["fixtures"]
    with tempfile.TemporaryDirectory(prefix="flash-verify-") as directory:
        tmp = Path(directory)
        patch = reference["benchmark_patch"]
        if args.compare_pool:
            patch = Path(__file__).with_name("flash-next-verify-size-lru.patch").read_text()
        binary = build_verifier(args.build.resolve(), tmp, patch, args.compare_pool)
        model = args.model or reference["model"]
        results = []
        if args.compare_pool:
            for enabled in (False, True, True, False):
                result = measure(binary, tmp, 8, reference, fixtures, model, size_lru=enabled)
                result["pool_policy"] = "size_lru" if enabled else "stock"
                results.append(result)
        else:
            results = [measure(binary, tmp, width, reference, fixtures, model) for width in (1, 8)]
    with args.out.open("x") as fd:
        json.dump(results, fd, indent=2)
        fd.write("\n")


if __name__ == "__main__":
    main()
