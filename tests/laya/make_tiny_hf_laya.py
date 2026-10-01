#!/usr/bin/env python3
"""Tiny laya HF checkpoints for the C++ converter tests (tools/laya/laya-convert.h).

usage:
  python make_tiny_hf_laya.py [--out tests/laya/convert]          write the fixture directories
  python make_tiny_hf_laya.py --golden                              ... and golden.sha256 (needs the Python converter)
  python make_tiny_hf_laya.py --check build/bin/llama-laya-convert  convert with Python and C++, compare
                                                                    (exit 77: no torch, or not transformers 5.17 / tokenizers 0.23)

Two checkpoints with the file layout of the real ones (rl_agent_config.json, encoder/config.json,
tokenizer/{tokenizer.json, tokenizer_config.json}, model*.safetensors, README.md):
  laya-tiny-ms-v0.1-8M       mmBERT family: metaspace BPE with byte fallback, F16 weights, one shard
  laya-bl-tiny-instruct-30K  ModernBERT family: byte-level BPE, F32 / F16 / BF16 / F64 weights, two shards
                             + index, NaN / inf / subnormal / f16 tie values where every outtype keeps them
The directory names exercise the gguf-py name heuristics (basename, version, finetune, size label).
Only stdlib + numpy; safetensors are written by hand (header keys and data out of name order, so the
readers must sort). The output is deterministic, the fixtures are committed.

golden.sha256 holds the sha256 of the Python converter's GGUF for each case (convert_hf_to_gguf.py with
this tree's gguf-py). test-laya-convert checks the C++ output against it without Python.
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

try:
    import numpy as np
except ImportError:
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# (fixture, outtype, --model-name or None)
CASES = [
    ("laya-tiny-ms-v0.1-8M", "f32", None),
    ("laya-tiny-ms-v0.1-8M", "f16", None),
    ("laya-tiny-ms-v0.1-8M", "q8_0", None),
    ("laya-tiny-ms-v0.1-8M", "f16", "tiny-ms"),
    ("laya-bl-tiny-instruct-30K", "f32", None),
    ("laya-bl-tiny-instruct-30K", "f16", None),
    ("laya-bl-tiny-instruct-30K", "q8_0", None),
]

MS = "\u2581"


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_text(path, text):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def to_bytes(arr, dtype):
    if dtype == "F32":
        return np.ascontiguousarray(arr, dtype="<f4").tobytes()
    if dtype == "F16":
        return np.ascontiguousarray(arr, dtype="<f2").tobytes()
    if dtype == "F64":
        return np.ascontiguousarray(arr, dtype="<f8").tobytes()
    if dtype == "BF16":
        # round to nearest even from f32
        u = np.ascontiguousarray(arr, dtype="<f4").view("<u4").astype(np.uint64)
        u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")
        return u.tobytes()
    raise ValueError(dtype)


def write_safetensors(path, tensors, meta=None):
    """tensors: list of (name, dtype, shape, raw bytes). Header keys and data are written in reverse name
    order (not the sorted order the converters use)."""
    items = sorted(tensors, key=lambda t: t[0], reverse=True)
    header = {}
    if meta is not None:
        header["__metadata__"] = meta
    off = 0
    blobs = []
    for name, dtype, shape, raw in items:
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    h = json.dumps(header, separators=(",", ":")).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def model_tensors(rng, d, n_ff, n_layer, n_head_layer, vocab, head_ff, act_hidden, n_act, dtype_of):
    """laya checkpoint tensors (encoder.* ModernBERT names + decision head), values ~ N(0, 0.05)."""
    shapes = [("encoder.embeddings.tok_embeddings.weight", (vocab, d)),
              ("encoder.embeddings.norm.weight", (d,)),
              ("encoder.final_norm.weight", (d,))]
    for i in range(n_layer):
        p = "encoder.layers.%d." % i
        if i > 0:
            shapes.append((p + "attn_norm.weight", (d,)))
        shapes += [(p + "attn.Wqkv.weight", (3 * d, d)), (p + "attn.Wo.weight", (d, d)),
                   (p + "mlp_norm.weight", (d,)), (p + "mlp.Wi.weight", (2 * n_ff, d)), (p + "mlp.Wo.weight", (d, n_ff))]
    for i in range(n_head_layer):
        p = "head.layers.%d." % i
        shapes += [(p + "self_attn.in_proj_weight", (3 * d, d)), (p + "self_attn.in_proj_bias", (3 * d,)),
                   (p + "self_attn.out_proj.weight", (d, d)), (p + "self_attn.out_proj.bias", (d,)),
                   (p + "linear1.weight", (head_ff, d)), (p + "linear1.bias", (head_ff,)),
                   (p + "linear2.weight", (d, head_ff)), (p + "linear2.bias", (d,)),
                   (p + "norm1.weight", (d,)), (p + "norm1.bias", (d,)), (p + "norm2.weight", (d,)), (p + "norm2.bias", (d,))]
    shapes += [("type_emb.weight", (3, d)),
               ("scorer.0.weight", (d,)), ("scorer.0.bias", (d,)), ("scorer.1.weight", (d, d)), ("scorer.1.bias", (d,)),
               ("scorer.3.weight", (1, d)), ("scorer.3.bias", (1,)),
               ("act_head.0.weight", (act_hidden, d + 4)), ("act_head.0.bias", (act_hidden,)),
               ("act_head.2.weight", (n_act, act_hidden)), ("act_head.2.bias", (n_act,))]
    out = {}
    for name, shape in shapes:
        a = (rng.standard_normal(shape) * 0.05).astype(np.float32)
        if name.endswith("norm.weight") or name.endswith("norm1.weight") or name.endswith("norm2.weight"):
            a = a + 1.0
        out[name] = [dtype_of(name), a]
    return out


def pack(tensors):
    return [(name, dt, arr.shape, to_bytes(arr, dt)) for name, (dt, arr) in tensors.items()]


def f32_bits(u):
    return np.array([u], dtype="<u4").view("<f4")[0]


# ---------------------------------------------------------------------------------------------
# laya-tiny-ms: mmBERT-like, metaspace BPE
# ---------------------------------------------------------------------------------------------

def make_ms(out):
    d = os.path.join(out, "laya-tiny-ms-v0.1-8M")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "encoder"))
    os.makedirs(os.path.join(d, "tokenizer"))
    write_text(os.path.join(d, "README.md"),
               "---\n"
               "license: apache-2.0\n"
               "library_name: transformers\n"
               "pipeline_tag: text-classification\n"
               "language: [multilingual, en, de, \"no\"]\n"
               "tags:\n"
               "- laya\n"
               "- tiny\n"
               "- no\n"
               "base_model: jhu-clsp/mmBERT-base\n"
               "model-index:\n"
               "  - name: tiny\n"
               "    results: []\n"
               "---\n"
               "# laya tiny (metaspace)\n\nTest fixture, random weights.\n")
    write_json(os.path.join(d, "rl_agent_config.json"), {
        "encoder": "jhu-clsp/mmBERT-base", "head_layers": 2, "max_len": 64, "head_max_len": 32, "max_prefixes": 6,
        "act_costs": {"escalate": 0.5}, "cost_wrong_act": 3.0, "amp_dtype": "bf16", "model_name": "rl-agent",
        "temperature": [1.0, 1.0, 1.0], "temperature_by_options": {},
        "training": {"updates": 1, "epochs_completed": 1, "hours": 0.01, "world_size": 1, "fine_tuned_from_checkpoint": False}})
    write_json(os.path.join(d, "encoder", "config.json"), {
        "architectures": ["ModernBertForMaskedLM"], "bos_token_id": 2, "cls_token_id": 1, "eos_token_id": 1,
        "global_attn_every_n_layers": 3, "hidden_activation": "gelu", "hidden_size": 32, "intermediate_size": 16,
        "layer_norm_eps": 1e-05, "local_attention": 16, "mask_token_id": 4, "max_position_embeddings": 128,
        "model_type": "modernbert", "norm_eps": 1e-05, "num_attention_heads": 4, "num_hidden_layers": 2, "pad_token_id": 0,
        "rope_parameters": {"full_attention": {"rope_theta": 160000, "rope_type": "default"},
                            "sliding_attention": {"rope_theta": 160000, "rope_type": "default"}},
        "sep_token_id": 1, "tie_word_embeddings": True, "vocab_size": 64})

    vocab_list = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<2mass>", "<unused0>", MS * 2, MS * 3, MS, "a", "b", "c",
                  MS + "a", "ab", MS + "ab", "\u00e9", "<0x41>", "<start_of_turn>", "<end_of_turn>", "x"]
    vocab = {t: i for i, t in enumerate(vocab_list)}

    def at(i, special, normalized=False, lstrip=False):
        return {"id": i, "content": vocab_list[i], "single_word": False, "lstrip": lstrip, "rstrip": False,
                "normalized": normalized, "special": special}
    added = [at(0, True), at(1, True), at(2, True), at(3, True), at(4, True, lstrip=True), at(5, False), at(6, False),
             at(7, False), at(8, False), at(18, False), at(19, False)]
    tj = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": MS},
        "pre_tokenizer": {"type": "Metaspace", "replacement": MS, "prepend_scheme": "always", "split": True},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                                      {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
                           "pair": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                                    {"SpecialToken": {"id": "<eos>", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}},
                                    {"SpecialToken": {"id": "<eos>", "type_id": 0}}],
                           "special_tokens": {"<bos>": {"id": "<bos>", "ids": [2], "tokens": ["<bos>"]},
                                              "<eos>": {"id": "<eos>", "ids": [1], "tokens": ["<eos>"]}}},
        "decoder": {"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": MS}, "content": " "},
                                                     {"type": "ByteFallback"}, {"type": "Fuse"}]},
        "model": {"type": "BPE", "dropout": None, "unk_token": "<unk>", "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": True, "byte_fallback": True, "ignore_merges": False,
                  "vocab": vocab, "merges": [[MS, "a"], ["a", "b"], [MS + "a", "b"]]},
    }
    write_json(os.path.join(d, "tokenizer", "tokenizer.json"), tj)
    write_json(os.path.join(d, "tokenizer", "tokenizer_config.json"), {
        "bos_token": "<bos>", "clean_up_tokenization_spaces": False, "cls_token": "<bos>", "eos_token": "<eos>",
        "extra_special_tokens": {"extra_0": "<start_of_turn>", "extra_1": "<end_of_turn>"}, "mask_token": "<mask>",
        "model_input_names": ["input_ids", "attention_mask"], "model_max_length": 128, "pad_token": "<pad>",
        "padding_side": "right", "sep_token": "<eos>", "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>"})

    rng = np.random.default_rng(20260930)
    t = model_tensors(rng, d=32, n_ff=16, n_layer=2, n_head_layer=2, vocab=64, head_ff=64, act_hidden=16, n_act=2,
                      dtype_of=lambda name: "F16")
    t["temperature"] = ["F32", np.ones(3, dtype=np.float32)]
    write_safetensors(os.path.join(d, "model.safetensors"), pack(t), meta={"format": "pt"})
    return d


# ---------------------------------------------------------------------------------------------
# laya-bl-tiny: ModernBERT-like, byte-level BPE, mixed dtypes, two shards
# ---------------------------------------------------------------------------------------------

def make_bl(out):
    d = os.path.join(out, "laya-bl-tiny-instruct-30K")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.join(d, "encoder"))
    os.makedirs(os.path.join(d, "tokenizer"))
    write_text(os.path.join(d, "README.md"),
               "---\n"
               "license: [apache-2.0, mit]\n"
               "model_name: Tiny Laya BL\n"
               "pipeline_tag: text-classification\n"
               "language:\n"
               "- en\n"
               "tags: [laya, system-one, 'quoted tag']   # comment\n"
               "datasets:\n"
               "- https://huggingface.co/datasets/org/ds\n"
               "description: \"A tiny \\u00e9 fixture\"\n"
               "---\n"
               "# laya tiny (byte-level)\n")
    write_json(os.path.join(d, "generation_config.json"), {"temperature": 0.7, "top_k": 40})
    write_json(os.path.join(d, "rl_agent_config.json"), {
        "encoder": "answerdotai/ModernBERT-large", "head_layers": 1, "max_len": 96, "head_max_len": 48,
        "act_costs": {"escalate": 0.5, "defer": 0.25},
        "temperature": [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
        "temperature_by_options": {"choice:3-5": 1.7601518630981445, "choice:6-10": 1.0000158548355103,
                                   "score:3-5": 1.2514300346374512, "noul:2": 1.983399510383606,
                                   "choice:11+": 0.10058280825614929, "choice:2": 1.9063563346862793}})
    write_json(os.path.join(d, "encoder", "config.json"), {
        "architectures": ["ModernBertForMaskedLM"], "bos_token_id": 23, "cls_token_id": 23, "eos_token_id": 24,
        "global_attn_every_n_layers": 3, "hidden_activation": "gelu", "hidden_size": 32, "intermediate_size": 16,
        "layer_norm_eps": 1e-05, "local_attention": 16, "max_position_embeddings": 128, "model_type": "modernbert",
        "norm_eps": 1e-05, "num_attention_heads": 4, "num_hidden_layers": 3, "pad_token_id": 25,
        "rope_parameters": {"full_attention": {"rope_theta": 160000.0, "rope_type": "default"},
                            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}},
        "sep_token_id": 24, "tie_word_embeddings": True, "vocab_size": 30})

    vocab_list = ["|||IP_ADDRESS|||", "<|padding|>", "!", "a", "b", "c", "\u0120", "\u0120a", "ab", "\u0120ab", "\u00c3", "\u00a9",
                  "\u00c3\u00a9", "\u010a", "\u010a\u010a", " ", " a", "x", "y", "z"]
    vocab = {t: i for i, t in enumerate(vocab_list)}
    extra = {20: "  ", 21: "\u0120xyz", 22: "[UNK]", 23: "[CLS]", 24: "[SEP]", 25: "[PAD]", 26: "[MASK]", 27: "<|endoftext|>"}

    def at(i, special, normalized, lstrip=False):
        content = vocab_list[i] if i < len(vocab_list) else extra[i]
        return {"id": i, "content": content, "single_word": False, "lstrip": lstrip, "rstrip": False,
                "normalized": normalized, "special": special}
    added = [at(0, False, True), at(1, False, False), at(20, False, True), at(21, False, False), at(22, True, False),
             at(23, True, False), at(24, True, False), at(25, True, False), at(26, True, False, lstrip=True), at(27, False, False)]
    tj = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "post_processor": {"type": "TemplateProcessing",
                           "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                                      {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
                           "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}}, {"Sequence": {"id": "A", "type_id": 0}},
                                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 0}},
                                    {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
                           "special_tokens": {"[CLS]": {"id": "[CLS]", "ids": [23], "tokens": ["[CLS]"]},
                                              "[SEP]": {"id": "[SEP]", "ids": [24], "tokens": ["[SEP]"]}}},
        "decoder": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab,
                  "merges": [["\u0120", "a"], ["a", "b"], ["\u0120a", "b"], ["\u00c3", "\u00a9"], ["\u010a", "\u010a"], [" ", "a"]]},
    }
    write_json(os.path.join(d, "tokenizer", "tokenizer.json"), tj)
    write_json(os.path.join(d, "tokenizer", "tokenizer_config.json"), {
        "clean_up_tokenization_spaces": True, "cls_token": "[CLS]", "mask_token": "[MASK]",
        "model_input_names": ["input_ids", "attention_mask"], "model_max_length": 128, "pad_token": "[PAD]",
        "sep_token": "[SEP]", "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "[UNK]"})

    def dtype_of(name):
        if name.endswith("tok_embeddings.weight") or "Wi" in name or "linear" in name or name.startswith("act_head.0") or name.startswith("scorer.3"):
            return "F32"
        if "Wqkv" in name or "in_proj" in name or name.endswith("norm1.weight"):
            return "BF16"
        if name.startswith("type_emb") or "mlp.Wo" in name:
            return "F64"
        return "F16"
    rng = np.random.default_rng(1234567)
    t = model_tensors(rng, d=32, n_ff=16, n_layer=3, n_head_layer=1, vocab=30, head_ff=64, act_hidden=16, n_act=3, dtype_of=dtype_of)

    # f32 -> f16 rounding cases in a q8_0-eligible tensor: finite only
    emb = t["encoder.embeddings.tok_embeddings.weight"][1]
    ties = [1.0 + 2.0 ** -11, 1.0 + 3 * 2.0 ** -11, 2.0 ** -25, 3 * 2.0 ** -26, 2.0 ** -24, 65520.0, 65519.0, -0.0,
            1e-10, -(2.0 ** -14) * 1.5, 6.1e-5, -65504.0]
    emb[5, :len(ties)] = np.array(ties, dtype=np.float32)
    # non-finite values where every outtype keeps them: 1-D tensors (F32 out) and rows that are not whole q8_0 blocks (F16)
    a0 = t["act_head.0.weight"][1]
    a0[0, :8] = np.array([f32_bits(0x7FC00000), f32_bits(0x7F800001), f32_bits(0xFFC00123), np.inf, -np.inf,
                          f32_bits(0x7F802000), 7.0e4, -1e-9], dtype=np.float32)
    n = t["encoder.final_norm.weight"][1]                    # F16 source, 1-D -> F32 out
    n[:4] = np.array([np.nan, np.inf, -0.0, 6e-8], dtype=np.float32)
    t["act_head.2.bias"][1][:] = np.array([np.nan, -np.inf, 1.0], dtype=np.float32)
    shards = [{}, {}]
    for i, name in enumerate(sorted(t)):
        shards[0 if name < "encoder.layers.1" else 1][name] = t[name]
    # f16 signaling NaN payload in a 1-D F16 tensor
    b = bytearray(to_bytes(t["scorer.0.bias"][1], "F16"))
    b[0:2] = struct.pack("<H", 0x7C01)
    b[2:4] = struct.pack("<H", 0xFE3F)
    for s in shards:
        if "scorer.0.bias" in s:
            s["scorer.0.bias"] = ["F16", np.frombuffer(bytes(b), dtype="<f2").copy()]
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    weight_map = {}
    for fname, s in zip(names, shards):
        write_safetensors(os.path.join(d, fname), pack(s))
        for name in s:
            weight_map[name] = fname
    write_json(os.path.join(d, "model.safetensors.index.json"), {"metadata": {"total_size": 0}, "weight_map": dict(sorted(weight_map.items()))})
    return d


# ---------------------------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# The C++ port reproduces AutoTokenizer as these versions behave (added-token round trip, special
# flags); another version is not a reference for it, so --check skips instead of failing.
REF_VERSIONS = {"transformers": "5.17.", "tokenizers": "0.23."}


def python_env():
    """ (interpreter, env, why-not): why-not is "" when the interpreter can be the reference """
    py = os.environ.get("LAYA_REF_PYTHON", sys.executable)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(ROOT, "gguf-py") + os.pathsep + env.get("PYTHONPATH", "")
    env["LC_ALL"] = "C"
    probe = ("import torch, transformers, tokenizers, numpy, yaml\n"
             "print(transformers.__version__, tokenizers.__version__)")
    r = subprocess.run([py, "-c", probe], env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if r.returncode != 0:
        return py, env, "torch / transformers / tokenizers / numpy / yaml not importable by %s (set LAYA_REF_PYTHON)" % py
    tv, kv = r.stdout.split()
    if not tv.startswith(REF_VERSIONS["transformers"]) or not kv.startswith(REF_VERSIONS["tokenizers"]):
        return py, env, "%s has transformers %s / tokenizers %s, the reference is transformers %sx / tokenizers %sx" % (
            py, tv, kv, REF_VERSIONS["transformers"], REF_VERSIONS["tokenizers"])
    return py, env, ""


def py_convert(py, env, ckpt, out, outtype, name):
    cmd = [py, os.path.join(ROOT, "convert_hf_to_gguf.py"), ckpt, "--outfile", out, "--outtype", outtype]
    if name is not None:
        cmd += ["--model-name", name]
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout)
        raise SystemExit("convert_hf_to_gguf.py failed for %s %s" % (ckpt, outtype))


def case_file(fixture, outtype, name):
    return "%s-%s%s.gguf" % (fixture, outtype, "" if name is None else "-name")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "convert"))
    ap.add_argument("--golden", action="store_true", help="also write golden.sha256 with the Python converter")
    ap.add_argument("--check", metavar="LLAMA_LAYA_CONVERT", help="compare the C++ converter with Python on fresh fixtures")
    args = ap.parse_args()

    # run under the reference interpreter when one is named
    ref = os.environ.get("LAYA_REF_PYTHON")
    if ref and os.path.realpath(ref) != os.path.realpath(sys.executable) and not os.environ.get("LAYA_REF_REEXEC"):
        env = dict(os.environ, LAYA_REF_REEXEC="1")
        return subprocess.run([ref, os.path.abspath(__file__)] + sys.argv[1:], env=env).returncode
    if np is None:
        print("skip: numpy is not importable by %s (set LAYA_REF_PYTHON)" % sys.executable, flush=True)
        return 77 if args.check else 1

    if args.check:
        py, env, why = python_env()
        if why:
            print("skip: " + why, flush=True)
            return 77
        tmp = tempfile.mkdtemp(prefix="laya-convert-check-")
        try:
            make_ms(tmp)
            make_bl(tmp)
            bad = 0
            for fixture, outtype, name in CASES:
                ref = os.path.join(tmp, "py-" + case_file(fixture, outtype, name))
                out = os.path.join(tmp, "cv-" + case_file(fixture, outtype, name))
                py_convert(py, env, os.path.join(tmp, fixture), ref, outtype, name)
                cmd = [args.check, os.path.join(tmp, fixture), "-o", out, "--outtype", outtype, "--verify-against", ref]
                if name is not None:
                    cmd += ["--model-name", name]
                r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                print("%-40s %s" % (case_file(fixture, outtype, name), r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no output)"), flush=True)
                bad += r.returncode != 0
            print("%d/%d cases identical" % (len(CASES) - bad, len(CASES)), flush=True)
            return 1 if bad else 0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    os.makedirs(args.out, exist_ok=True)
    dirs = [make_ms(args.out), make_bl(args.out)]
    for d in dirs:
        print("wrote", d, flush=True)
    if args.golden:
        py, env, why = python_env()
        if why:
            raise SystemExit("--golden: " + why)
        tmp = tempfile.mkdtemp(prefix="laya-convert-golden-")
        lines = ["# sha256 of convert_hf_to_gguf.py output (this tree's gguf-py): <file> <fixture> <outtype> <model-name|-> <bytes> <sha256>"]
        try:
            for fixture, outtype, name in CASES:
                f = os.path.join(tmp, case_file(fixture, outtype, name))
                py_convert(py, env, os.path.join(args.out, fixture), f, outtype, name)
                lines.append("%s %s %s %s %d %s" % (case_file(fixture, outtype, name), fixture, outtype, name or "-",
                                                    os.path.getsize(f), sha256_file(f)))
                print(lines[-1], flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        write_text(os.path.join(args.out, "golden.sha256"), "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
