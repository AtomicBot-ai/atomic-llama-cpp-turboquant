"""Unit tests of the identity records (tests/laya/parity/identity.py).

Modelled on laya.cpp tests/test_identity.py:14-23 (https://github.com/lkarlslund/laya.cpp,
MIT License, Copyright (c) 2026 Lars Karlslund; license text in parity_gate.py): a change of a
loaded library must change the identity. Here the libraries are found by walking the build tree.
"""

import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import identity as I  # noqa: E402


def _tree(root):
    (root / "bin").mkdir()
    exe = root / "bin" / "llama-server"
    exe.write_bytes(b"server v1")
    (root / "bin" / "libggml-base.0.dylib").write_bytes(b"ggml base v1")
    (root / "bin" / "libggml-cpu.so").write_bytes(b"ggml cpu v1")
    (root / "bin" / "llama.dll").write_bytes(b"llama dll v1")
    (root / "CMakeCache.txt").write_text("GGML_METAL:BOOL=ON\n")
    (root / "notes.txt").write_text("not part of the identity\n")
    (root / "CMakeFiles").mkdir()
    (root / "CMakeFiles" / "libggml-stale.so").write_bytes(b"object dir")
    return exe


def test_loaded_library_change_invalidates_identity(tmp_path):
    exe = _tree(tmp_path)
    first = I.build_identity(str(tmp_path), [str(exe)])
    assert first == I.build_identity(str(tmp_path), [str(exe)])
    assert sorted(first["files"]) == ["CMakeCache.txt", "bin/libggml-base.0.dylib", "bin/libggml-cpu.so", "bin/llama-server", "bin/llama.dll"]
    for name in ("bin/libggml-base.0.dylib", "bin/libggml-cpu.so", "bin/llama.dll", "bin/llama-server", "CMakeCache.txt"):
        p = tmp_path / name
        old = p.read_bytes()
        p.write_bytes(old + b" changed")
        assert I.build_identity(str(tmp_path), [str(exe)])["sha256"] != first["sha256"], name
        p.write_bytes(old)
    assert I.build_identity(str(tmp_path), [str(exe)])["sha256"] == first["sha256"]


def test_unrelated_files_do_not_change_identity(tmp_path):
    exe = _tree(tmp_path)
    first = I.build_identity(str(tmp_path), [str(exe)])["sha256"]
    (tmp_path / "notes.txt").write_text("edited\n")
    (tmp_path / "CMakeFiles" / "libggml-stale.so").write_bytes(b"rebuilt object dir")
    (tmp_path / "bin" / "llama-other-tool").write_bytes(b"another tool")
    assert I.build_identity(str(tmp_path), [str(exe)])["sha256"] == first


def test_new_backend_library_changes_identity(tmp_path):
    exe = _tree(tmp_path)
    first = I.build_identity(str(tmp_path), [str(exe)])["sha256"]
    (tmp_path / "bin" / "libggml-metal.0.dylib").write_bytes(b"metal backend")
    assert I.build_identity(str(tmp_path), [str(exe)])["sha256"] != first


def _gguf(path, kv):
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    out = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv))
    for k, (t, v) in kv.items():
        out += s(k) + struct.pack("<I", t)
        if t == 8:
            out += s(v)
        elif t == 4:
            out += struct.pack("<I", v)
        elif t == 9:  # array of float32
            out += struct.pack("<IQ", 6, len(v)) + b"".join(struct.pack("<f", x) for x in v)
    path.write_bytes(out)


def test_model_identity_reads_the_header(tmp_path):
    g = tmp_path / "m.gguf"
    _gguf(g, {"general.architecture": (8, "laya"), "tokenizer.scores": (9, [0.5] * 100),
              "general.name": (8, "E4E9Ddf21A7B1903B7Acffd8814Ad4307Bf63A67"), "general.file_type": (4, 7)})
    m = I.model_identity(str(g))
    assert m["architecture"] == "laya" and m["weights"] == "q8_0" and m["file_type"] == 7
    assert m["revision"] == "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67"
    _gguf(g, {"general.architecture": (8, "laya"), "general.name": (8, "tiny")})
    m2 = I.model_identity(str(g))
    assert m2["weights"] == "f32" and m2["revision"] is None and m2["sha256"] != m["sha256"]


def test_corpus_identity_passes_through_a_reference(tmp_path):
    items = tmp_path / "items.jsonl"
    items.write_text('{"id": "a"}\n')
    ref = tmp_path / "ref.jsonl"
    ref.write_text('{"id": "a", "per_question": {}}\n')
    I.write_sidecar(str(ref), {"corpus": I.corpus_identity(str(items), "english", 1)})
    c = I.corpus_identity(str(ref))
    assert c["sha256"] == I.file_sha256(str(items)) and c["subset"] == "english"
    assert c["input_sha256"] == I.file_sha256(str(ref))


def test_runtime_from_props():
    r = I.runtime_from_props({"decision": {"device": "MTL0", "plan": {"name": "sequential", "kernels": "cpu+blas", "n_threads": 8}}})
    assert r["backend"] == "metal" and r["kernels"] == "cpu+blas" and r["precision"] == "default"
    assert I.runtime_from_props({"device": "cpu", "plan": {}})["backend"] == "cpu"
    assert I.backend_of("CUDA0") == "cuda" and I.backend_of("Vulkan1") == "vulkan"


def test_merge_command(tmp_path):
    exe = _tree(tmp_path)
    res = tmp_path / "bench.json"
    res.write_text(json.dumps({"requests": []}))
    sys_argv = sys.argv
    sys.argv = ["identity.py", "merge", str(res), "--exe", str(exe), "--kernels", "auto", "--threads", "8"]
    try:
        I.main()
    finally:
        sys.argv = sys_argv
    d = json.loads(res.read_text())
    assert d["requests"] == [] and d["identity"]["build"]["sha256"] == I.build_identity(str(tmp_path), [str(exe)])["sha256"]
    assert d["identity"]["runtime"]["kernels"] == "auto"


def _speed_pair(tmp_path, device="CUDA0", desc="NVIDIA GeForce RTX 4090", kernels="cuda", precision="default"):
    exe = _tree(tmp_path)
    (tmp_path / "bin" / "llama-laya-cli").write_bytes(b"cli v1")
    (tmp_path / "bin" / "llama-decision-bench").write_bytes(b"bench v1")
    g = tmp_path / "m.gguf"
    _gguf(g, {"general.architecture": (8, "laya"), "general.file_type": (4, 1)})
    cli = str(tmp_path / "bin" / "llama-laya-cli")
    bench = str(tmp_path / "bin" / "llama-decision-bench")
    rt = {"device": device, "backend": I.backend_of(device), "kernels_resolved": kernels, "precision": precision, "device_description": desc}
    parity = I.make_identity("cli", build=str(tmp_path), exes=[cli], model=str(g), runtime=rt)
    run = I.make_identity("bench", build=str(tmp_path), exes=[bench, str(exe), cli], model=str(g),
                          runtime=I.runtime_from_bench({"device": "CUDA1" if device.startswith("CUDA") else device, "device_description": desc, "kernels": kernels, "precision": precision}))
    return run, parity, g


def test_speed_identity_covers_same_tree_model_device(tmp_path):
    run, parity, _ = _speed_pair(tmp_path)
    # another GPU index of the same model is the same device
    assert I.speed_identity_problems(run, parity) == []


def test_speed_identity_refuses_changes(tmp_path):
    run, parity, g = _speed_pair(tmp_path)
    other = json.loads(json.dumps(parity))
    other["build"]["files"]["bin/llama-laya-cli"] = "0" * 64  # the parity run used another binary
    assert any("build differs" in w for w in I.speed_identity_problems(run, other))
    other = json.loads(json.dumps(parity))
    other["model"]["sha256"] = "f" * 64
    assert any("GGUF differs" in w for w in I.speed_identity_problems(run, other))
    for key, val, word in (("backend", "vulkan", "backend"), ("device_description", "NVIDIA GeForce RTX 5060 Ti", "device"),
                           ("precision", "strict", "precision")):
        other = json.loads(json.dumps(parity))
        other["runtime"][key] = val
        assert any(w.startswith(word) for w in I.speed_identity_problems(run, other)), key
    assert I.speed_identity_problems(None, parity) == ["no identity record"]


def test_speed_identity_cpu_needs_same_kernels(tmp_path):
    run, parity, _ = _speed_pair(tmp_path, device="cpu", desc="", kernels="cpu+blas")
    assert I.speed_identity_problems(run, parity) == []
    parity["runtime"]["kernels_resolved"] = "cpu"
    assert I.speed_identity_problems(run, parity) == ["CPU kernels cpu+blas vs cpu"]


def test_merge_takes_the_runtime_of_a_bench_result(tmp_path):
    exe = _tree(tmp_path)
    res = tmp_path / "bench.json"
    res.write_text(json.dumps({"bench": "llama-decision-bench", "device": "MTL0", "device_description": "Apple M5",
                               "kernels": "metal", "precision": "strict", "n_threads": 8}))
    sys_argv = sys.argv
    sys.argv = ["identity.py", "merge", str(res), "--exe", str(exe)]
    try:
        I.main()
    finally:
        sys.argv = sys_argv
    rt = json.loads(res.read_text())["identity"]["runtime"]
    assert rt["backend"] == "metal" and rt["device_description"] == "Apple M5" and rt["precision"] == "strict"


def test_source_manifest_follows_the_laya_sources(tmp_path):
    src, build = tmp_path / "src", tmp_path / "build"
    (src / "tools" / "laya").mkdir(parents=True)
    (src / "tools" / "laya" / "laya.cpp").write_text("graph v1\n")
    (src / "tools" / "laya" / "__pycache__").mkdir()
    (src / "tools" / "laya" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    (src / "ggml" / "src" / "ggml-cuda").mkdir(parents=True)
    (src / "ggml" / "src" / "ggml-cuda" / "mmq.cu").write_text("kernel v1\n")
    (src / "README.md").write_text("not a laya source\n")
    build.mkdir()
    exe = _tree(build)
    with open(build / "CMakeCache.txt", "a") as f:
        f.write("CMAKE_HOME_DIRECTORY:INTERNAL=%s\n" % src)
    ident = I.make_identity("cli", build=str(build), exes=[str(exe)])
    first = ident["source"]
    assert sorted(first["files"]) == ["ggml/src/ggml-cuda/mmq.cu", "tools/laya/laya.cpp"]
    assert set(first["paths"]) == set(I.SOURCE_PATHS)
    (src / "README.md").write_text("edited\n")
    assert I.source_identity(str(src))["sha256"] == first["sha256"]
    (src / "tools" / "laya" / "laya.cpp").write_text("graph v2\n")
    second = I.source_identity(str(src))
    assert second["sha256"] != first["sha256"]
    assert second["paths"]["tools/laya"] != first["paths"]["tools/laya"]
    assert second["paths"]["ggml/src"] == first["paths"]["ggml/src"]
    # a build dir without a source tree: no source block
    os.remove(build / "CMakeCache.txt")
    assert "source" not in I.make_identity("cli", build=str(build), exes=[str(exe)])
