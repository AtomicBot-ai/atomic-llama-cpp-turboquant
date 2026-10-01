#!/usr/bin/env python3
"""Identity records for parity and bench runs (Phase 4b; DECISION.md "Backend parity tiers").

    python3 tests/laya/parity/identity.py write  -o run.identity.json [--build DIR] [--exe BIN ...] [--model GGUF] [--corpus FILE] [--props FILE] [--kind K]
    python3 tests/laya/parity/identity.py merge  <result.json> [same options]      # adds an "identity" key to a JSON object
    python3 tests/laya/parity/identity.py build-sha <build-dir> [--exe BIN ...]
    python3 tests/laya/parity/identity.py source-sha [<source-tree>] [--build DIR]

A number means something only together with what produced it. An identity names:
  build   sha256 over the executables that ran, every libggml* / libllama* / ggml*.dll / llama*.dll
          under the build tree (backends are loaded with dlopen, so linker output such as otool or
          ldd would miss them) and CMakeCache.txt; per-file hashes are kept too
  source  sha256 manifest of the source files the laya numbers depend on (SOURCE_PATHS under the
          source tree of the build, CMAKE_HOME_DIRECTORY of its CMakeCache.txt), as read when the
          identity is written: a binary hash alone does not say which source it was built from
  model   sha256 of the GGUF, general.architecture / name / file_type (read from the header,
          stdlib only), the weight type, and the checkpoint revision (the converter writes the
          snapshot hash as general.name)
  corpus  sha256 of the input file, the subset (all / english) and the item count; a run that
          reads a reference file inherits the corpus of that file's identity
  runtime device / backend / kernels / plan / threads / precision, from the server's /props
  host    OS, machine, CPU brand
parity_gate.check_identity() refuses to compare runs whose identities do not fit the tier;
speed_identity_problems() decides whether a passing parity gate covers a speed run (a bench result
of llama-decision-bench, whose runtime block comes from the result itself), for
scripts/bench-decision-report.py.
"""

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys

SCHEMA = "laya-parity-identity/1"

# general.file_type (llama_ftype) -> weight type name
FILE_TYPES = {0: "f32", 1: "f16", 2: "q4_0", 3: "q4_1", 7: "q8_0", 8: "q5_0", 9: "q5_1", 10: "q2_k",
              11: "q3_k_s", 12: "q3_k_m", 13: "q3_k_l", 14: "q4_k_s", 15: "q4_k_m", 16: "q5_k_s",
              17: "q5_k_m", 18: "q6_k", 32: "bf16"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_runtime_lib(name):
    n = name.lower()
    if not (n.startswith(("libggml", "libllama", "ggml", "llama"))):
        return False
    return n.endswith((".dylib", ".dll")) or ".so" in n


def build_files(build_dir, exes=()):
    """Files that decide what a run computes: the given executables, the runtime libraries, CMakeCache.txt."""
    root = os.path.realpath(build_dir)
    files = {}

    def add(p):
        rp = os.path.realpath(p)
        if not os.path.isfile(rp):
            return
        rel = os.path.relpath(rp, root) if rp.startswith(root + os.sep) else rp
        files[rel.replace(os.sep, "/")] = rp

    for e in exes:
        add(e)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "CMakeFiles")
        for fn in sorted(filenames):
            if is_runtime_lib(fn):
                add(os.path.join(dirpath, fn))
    add(os.path.join(root, "CMakeCache.txt"))
    return files


# what the laya numbers depend on: the engine, the decision library, the server adapter and the ggml backends
SOURCE_PATHS = ("tools/laya", "tools/decision", "tools/server/server-decision.cpp", "ggml/src", "ggml/include")
SOURCE_SKIP_DIRS = {"__pycache__", ".git"}


def source_root_of(build_dir):
    """The source tree a CMake build dir was configured from (CMAKE_HOME_DIRECTORY), or None."""
    cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cache):
        return None
    with open(cache, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("CMAKE_HOME_DIRECTORY:"):
                root = line.split("=", 1)[1].strip()
                return root if os.path.isdir(root) else None
    return None


def source_identity(root, paths=SOURCE_PATHS):
    """sha256 manifest of the files under paths (relative to root): total, per path, and per file."""
    per_file = {}
    for rel_top in paths:
        top = os.path.join(root, rel_top)
        if os.path.isfile(top):
            per_file[rel_top] = file_sha256(top)
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = sorted(d for d in dirnames if d not in SOURCE_SKIP_DIRS)
            for fn in sorted(filenames):
                if fn.endswith(".pyc"):
                    continue
                p = os.path.join(dirpath, fn)
                per_file[os.path.relpath(p, root).replace(os.sep, "/")] = file_sha256(p)

    def digest(items):
        h = hashlib.sha256()
        for rel, sha in items:
            h.update(rel.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
        return h.hexdigest()

    items = sorted(per_file.items())
    by_path = {top: digest([(r, sh) for r, sh in items if r == top or r.startswith(top + "/")]) for top in paths}
    return {"root": os.path.realpath(root), "sha256": digest(items), "paths": by_path, "files": per_file}


def build_identity(build_dir, exes=()):
    files = build_files(build_dir, exes)
    per_file = {rel: file_sha256(p) for rel, p in sorted(files.items())}
    h = hashlib.sha256()
    for rel, sha in per_file.items():
        h.update(rel.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
    return {"root": os.path.realpath(build_dir), "sha256": h.hexdigest(), "files": per_file}


def _read_str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_value(f, t):
    if t in _SCALAR:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return _read_str(f)
    if t == 9:
        (at,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        if at in _SCALAR:  # skip numeric arrays (token scores etc.) without decoding them
            f.seek(n * struct.calcsize(_SCALAR[at]), 1)
            return None
        vals = [_read_value(f, at) for _ in range(n)]
        return vals if n < 64 else None
    raise ValueError("unknown GGUF value type %d" % t)


def gguf_metadata(path, keys=("general.architecture", "general.name", "general.file_type")):
    """The wanted scalar keys of a GGUF header (stdlib only)."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file: " + path)
        (version,) = struct.unpack("<I", f.read(4))
        if version < 2:
            raise ValueError("GGUF version %d is not supported" % version)
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        for _ in range(n_kv):
            k = _read_str(f)
            (t,) = struct.unpack("<I", f.read(4))
            v = _read_value(f, t)
            if k in keys:
                out[k] = v
            if len(out) == len(keys):
                break
    return out


def is_revision(s):
    return isinstance(s, str) and len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower())


def model_identity(path):
    meta = gguf_metadata(path)
    name = meta.get("general.name")
    ft = meta.get("general.file_type")
    return {
        "path": os.path.abspath(path),
        "sha256": file_sha256(path),
        "architecture": meta.get("general.architecture"),
        "name": name,
        "file_type": ft,
        "weights": FILE_TYPES.get(ft, "f32" if ft is None else "ftype%d" % ft),
        "revision": name.lower() if is_revision(name) else None,
    }


def checkpoint_identity(ckpt_dir):
    """PyTorch reference checkpoint: the snapshot folder name is the HF revision."""
    d = os.path.realpath(ckpt_dir)
    rev = os.path.basename(d.rstrip("/"))
    cfg = os.path.join(d, "rl_agent_config.json")
    return {"path": d, "revision": rev.lower() if is_revision(rev) else None,
            "config_sha256": file_sha256(cfg) if os.path.isfile(cfg) else None, "weights": "f32"}


def sidecar_path(path):
    return path + ".identity.json"


def load_sidecar(path):
    p = sidecar_path(path)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def corpus_identity(path, subset="all", n_items=None):
    """The corpus of an input file; a reference file passes on the corpus it was made from."""
    parent = load_sidecar(path)
    if parent and parent.get("corpus"):
        c = dict(parent["corpus"])
        c["input_path"] = os.path.abspath(path)
        c["input_sha256"] = file_sha256(path)
        if subset != "all":
            c["subset"] = subset
        if n_items is not None:
            c["n_items"] = n_items
        return c
    return {"path": os.path.abspath(path), "sha256": file_sha256(path), "subset": subset, "n_items": n_items}


def cpu_brand():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return platform.processor() or None


def host_identity():
    return {"os": platform.platform(), "machine": platform.machine(), "cpu": cpu_brand(), "python": platform.python_version()}


def backend_of(device):
    """Backend family from a ggml device name: CPU, MTL0 / Metal, CUDA0, Vulkan0, ..."""
    d = (device or "cpu").lower()
    for prefix, name in (("cpu", "cpu"), ("mtl", "metal"), ("metal", "metal"), ("cuda", "cuda"), ("vulkan", "vulkan"),
                         ("rocm", "rocm"), ("hip", "rocm"), ("sycl", "sycl"), ("opencl", "opencl"), ("blas", "cpu")):
        if d.startswith(prefix):
            return name
    return d


def runtime_from_props(props):
    """Runtime block from /props (the whole reply or its "decision" object)."""
    d = props.get("decision", props) if isinstance(props, dict) else {}
    plan = d.get("plan") or {}
    device = d.get("device") or "cpu"
    return {"device": device, "backend": backend_of(device), "kernels": plan.get("kernels"), "plan": plan.get("name"),
            "n_threads": plan.get("n_threads"), "n_threads_blas": plan.get("n_threads_blas"),
            "precision": d.get("precision") or plan.get("precision") or "default", "props": d}


def runtime_from_bench(d):
    """Runtime block from a llama-decision-bench result JSON."""
    device = d.get("device") or "cpu"
    return {"device": device, "backend": backend_of(device), "device_description": d.get("device_description") or "",
            "kernels": d.get("kernels"), "plan": d.get("plan"), "n_threads": d.get("n_threads"),
            "n_threads_blas": d.get("n_threads_blas"), "precision": d.get("precision") or "default",
            "placement": d.get("placement")}


def _exe_names(i):
    root = (i.get("build") or {}).get("root") or ""
    names = set()
    for e in i.get("executables") or []:
        rp = os.path.realpath(e)
        rel = os.path.relpath(rp, root) if root and rp.startswith(root.rstrip("/") + "/") else rp
        names.add(rel.replace(os.sep, "/"))
    return names


def _device_description(rt):
    return rt.get("device_description") or (rt.get("props") or {}).get("device_description") or ""


def _kernels(rt):
    return rt.get("kernels_resolved") or rt.get("kernels")


def speed_identity_problems(run_id, parity_id):
    """Reasons why a parity record does not cover a speed run (empty list: it does).

    The same identity means: the same build tree (every file of the parity record's build, the
    executable that ran included, has the same sha256 in the speed run's build, and both trees have
    the same libraries and CMakeCache.txt; a speed run records llama-server and llama-laya-cli next
    to llama-decision-bench for this), the same GGUF sha256, and the same device: backend, device
    description (GPU model; the index does not matter) and precision, and on the CPU the kernels.
    """
    why = []
    if not run_id or not parity_id:
        return ["no identity record"]
    rb, pb = (run_id.get("build") or {}).get("files") or {}, (parity_id.get("build") or {}).get("files") or {}
    if not rb or not pb:
        why.append("no build record")
    else:
        diff = sorted(k for k, v in pb.items() if rb.get(k) != v)
        if diff:
            why.append("build differs: " + ", ".join(diff[:4]) + (" ..." if len(diff) > 4 else ""))
        r_libs = set(rb) - _exe_names(run_id)
        p_libs = set(pb) - _exe_names(parity_id)
        if r_libs != p_libs:
            why.append("build trees have different libraries: " + ", ".join(sorted(r_libs ^ p_libs)[:4]))
    rm, pm = (run_id.get("model") or {}).get("sha256"), (parity_id.get("model") or {}).get("sha256")
    if not rm or rm != pm:
        why.append("GGUF differs: %s vs %s" % ((rm or "-")[:12], (pm or "-")[:12]))
    rr, pr = run_id.get("runtime") or {}, parity_id.get("runtime") or {}
    if rr.get("backend", "cpu") != pr.get("backend", "cpu"):
        why.append("backend %s vs %s" % (rr.get("backend"), pr.get("backend")))
    elif rr.get("backend", "cpu") == "cpu":
        if _kernels(rr) != _kernels(pr):
            why.append("CPU kernels %s vs %s" % (_kernels(rr), _kernels(pr)))
    elif _device_description(rr) != _device_description(pr):
        why.append("device %r vs %r" % (_device_description(rr), _device_description(pr)))
    if rr.get("precision", "default") != pr.get("precision", "default"):
        why.append("precision %s vs %s" % (rr.get("precision", "default"), pr.get("precision", "default")))
    return why


def make_identity(kind, build=None, exes=(), model=None, corpus=None, subset="all", n_items=None, runtime=None, extra=None):
    ident = {"schema": SCHEMA, "kind": kind, "host": host_identity()}
    if build:
        ident["build"] = build_identity(build, exes)
        src = source_root_of(build)
        if src:
            ident["source"] = source_identity(src)
    if exes:
        ident["executables"] = [os.path.abspath(e) for e in exes]
    if model:
        ident["model"] = model_identity(model)
    if corpus:
        ident["corpus"] = corpus_identity(corpus, subset, n_items)
    if runtime:
        ident["runtime"] = runtime
    if extra:
        ident.update(extra)
    return ident


def write_sidecar(out_path, ident):
    with open(sidecar_path(out_path), "w", encoding="utf-8") as f:
        json.dump(ident, f, indent=1)
        f.write("\n")


def build_root_of(exe):
    d = os.path.dirname(os.path.abspath(exe))
    return os.path.dirname(d) if os.path.basename(d) in ("bin", "Release", "Debug", "RelWithDebInfo") else d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["write", "merge", "build-sha", "source-sha"])
    ap.add_argument("target", nargs="?", help="merge: the JSON result file; build-sha: the build dir")
    ap.add_argument("-o", "--out")
    ap.add_argument("--kind", default="run")
    ap.add_argument("--build")
    ap.add_argument("--exe", action="append", default=[])
    ap.add_argument("--model")
    ap.add_argument("--corpus")
    ap.add_argument("--subset", default="all")
    ap.add_argument("--props", help="a saved /props reply (JSON)")
    ap.add_argument("--kernels", help="runtime kernels when there is no /props (llama-decision-bench)")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()

    for p in a.exe + [x for x in (a.model, a.corpus, a.props) if x]:
        if not os.path.exists(p):
            sys.exit("missing: " + p)
    if a.cmd == "source-sha":
        root = a.target or (source_root_of(a.build) if a.build else None)
        if not root or not os.path.isdir(root):
            sys.exit("source-sha needs a source tree (or --build <configured build dir>)")
        src = source_identity(root)
        print(src["sha256"], flush=True)
        for top, sha in src["paths"].items():
            print("  %s  %s" % (sha, top), flush=True)
        return
    if a.cmd == "build-sha":
        if not a.target or not os.path.isdir(a.target):
            sys.exit("build-sha needs a build directory")
        b = build_identity(a.target, a.exe)
        print(b["sha256"], flush=True)
        for rel, sha in b["files"].items():
            print("  %s  %s" % (sha, rel), flush=True)
        return

    build = a.build or (build_root_of(a.exe[0]) if a.exe else None)
    runtime = None
    target = None
    if a.cmd == "merge":
        if not a.target or not os.path.isfile(a.target):
            sys.exit("merge needs an existing JSON file")
        with open(a.target, encoding="utf-8") as f:
            target = json.load(f)
        if not isinstance(target, dict):
            sys.exit("merge: %s is not a JSON object" % a.target)
    if target is not None and target.get("bench") == "llama-decision-bench":
        runtime = runtime_from_bench(target)  # the bench reports its own device, kernels and precision
    elif a.props:
        with open(a.props, encoding="utf-8") as f:
            runtime = runtime_from_props(json.load(f))
    elif a.kernels or a.threads:
        runtime = {"device": "cpu", "backend": "cpu", "kernels": a.kernels, "n_threads": a.threads, "precision": "default"}
    ident = make_identity(a.kind, build, a.exe, a.model, a.corpus, a.subset, None, runtime)

    if a.cmd == "write":
        if not a.out:
            sys.exit("write needs -o")
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(ident, f, indent=1)
            f.write("\n")
        print("wrote %s (build %s)" % (a.out, ident.get("build", {}).get("sha256", "-")[:12]), flush=True)
    else:
        d = target
        d["identity"] = ident
        with open(a.target, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
            f.write("\n")
        print("identity -> %s (build %s)" % (a.target, ident.get("build", {}).get("sha256", "-")[:12]), flush=True)


if __name__ == "__main__":
    main()
