# tiny random laya GGUF (< 1 MB) for offline smoke tests of llama-laya-cli and the decision server
# usage: python tests/decision/make_tiny_laya.py out.gguf [max_len [head_max_len]] [--english] [--marker-mismatch] [--q8]
# (the tokenizer has no merges, so every character is a token; the server tests use a longer max_len)
# --english: the bytelevel-bpe tokenizer of the English checkpoints (NFC, [CLS]/[SEP]/[MASK] with
# lstrip, no byte fallback) and their temperature_by_options buckets
# --marker-mismatch: laya.marker_token_id != tokenizer.ggml.mask_token_id (a GGUF that must not load)
# --q8: the encoder matmul weights in Q8_0 (the same random values), so the CPU repack buffers
# have something to repack (rows are multiples of 4: the NEON dotprod / i8mm Q8_0 repack)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

english = "--english" in sys.argv
marker_mismatch = "--marker-mismatch" in sys.argv
q8 = "--q8" in sys.argv
argv = [a for a in sys.argv if a not in ("--english", "--marker-mismatch", "--q8")]
out = argv[1]
max_len = int(argv[2]) if len(argv) > 2 else 128
head_max_len = int(argv[3]) if len(argv) > 3 else 64
rng = np.random.default_rng(0)

d, n_head, n_layer, n_ff, n_head_layers, n_act = 64, 4, 3, 32, 1, 2

# byte-level chars (gpt-2 byte_to_unicode) so every byte has a token
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
chars = sorted({chr(c) for c in cs}) + ["\u2581"]
byte_toks = [f"<0x{b:02X}>" for b in range(256)]
tokens = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"] + chars + byte_toks
ttype = [3, 3, 3, 3, 3] + [1] * len(chars) + [6] * len(byte_toks)
if english:
    # ModernBERT-like: special tokens at the end, two normalized added tokens (a space run, [unused0])
    chars = sorted({chr(c) for c in cs})
    tokens = chars + ["   ", "[unused0]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    ttype = [1] * len(chars) + [4, 4, 3, 3, 3, 3, 3]
n_vocab = len(tokens)


def w(*shape, dtype=np.float32):
    return (rng.standard_normal(shape) * 0.05).astype(dtype)


def ones(n):
    return np.ones(n, dtype=np.float32)


def add_matmul(name, x):
    # encoder matmul weight: F16, or Q8_0 of the same values with --q8
    if q8:
        wr.add_tensor(name, gguf.quants.quantize(x.astype(np.float32), gguf.GGMLQuantizationType.Q8_0),
                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    else:
        wr.add_tensor(name, x)


wr = gguf.GGUFWriter(out, "laya")
wr.add_context_length(max_len)
wr.add_embedding_length(d)
wr.add_block_count(n_layer)
wr.add_head_count(n_head)
wr.add_feed_forward_length(n_ff)
wr.add_vocab_size(n_vocab)
wr.add_layer_norm_rms_eps(1e-5)
wr.add_layer_norm_eps(1e-5)
wr.add_rope_freq_base(160000.0)
wr.add_sliding_window(8)
wr.add_uint32("laya.attention.sliding_window_pattern", 3)
ids = {t: i for i, t in enumerate(tokens)}
wr.add_head_layers(n_head_layers)
wr.add_n_qtype(3)
wr.add_marker_token_id((ids["[MASK]"] if english else 4) + (1 if marker_mismatch else 0))
wr.add_max_len(max_len)
wr.add_head_max_len(head_max_len)
# English: the base temperatures of convaiinnovations/laya (choice, score, noul)
wr.add_temperature([1.6369030475616455, 1.2514300346374512, 1.983399510383606] if english else [1.0, 1.5, 0.8])
wr.add_act_classes(n_act)
wr.add_tokenizer_model("gpt2")
wr.add_token_list(tokens)
wr.add_token_types(ttype)
wr.add_token_merges([])
if english:
    # the buckets of convaiinnovations/laya (choice:11+ sharpens: clamped to 0.5 when applied), except
    # noul:2 and score:3-5, which equal the base temperatures there: other values here, so a test can
    # tell a bucket lookup from the base value; score:2 is absent (falls back to the base 1.2514...)
    tbo = {"choice:3-5": 1.7601518630981445, "choice:6-10": 1.0000158548355103, "score:3-5": 1.375,
           "noul:2": 1.8125, "choice:11+": 0.10058280825614929, "choice:2": 1.9063563346862793}
    wr.add_array("laya.temperature_by_options.buckets", list(tbo.keys()))
    wr.add_array("laya.temperature_by_options.values", list(tbo.values()))
    wr.add_bos_token_id(ids["[CLS]"])
    wr.add_eos_token_id(ids["[SEP]"])
    wr.add_sep_token_id(ids["[SEP]"])
    wr.add_mask_token_id(ids["[MASK]"])
    wr.add_unk_token_id(ids["[UNK]"])
    wr.add_pad_token_id(ids["[PAD]"])
    wr.add_string("decision.laya.tokenizer", "bytelevel-bpe")
    wr.add_string("decision.laya.normalizer", "nfc")
    wr.add_array("decision.laya.added_tokens", [ids["   "], 2, ids["[unused0]"], 2, ids["[UNK]"], 0, ids["[CLS]"], 0,
                                                ids["[SEP]"], 0, ids["[PAD]"], 0, ids["[MASK]"], 1])
else:
    wr.add_bos_token_id(2)
    wr.add_eos_token_id(1)
    wr.add_sep_token_id(1)
    wr.add_mask_token_id(4)
    wr.add_unk_token_id(3)
    wr.add_pad_token_id(0)

wr.add_tensor("token_embd.weight", w(n_vocab, d, dtype=np.float16))
wr.add_tensor("token_embd_norm.weight", ones(d))
wr.add_tensor("output_norm.weight", ones(d))
for il in range(n_layer):
    p = f"blk.{il}."
    wr.add_tensor(p + "attn_norm.weight", ones(d))
    add_matmul(p + "attn_qkv.weight", w(3 * d, d, dtype=np.float16))
    add_matmul(p + "attn_output.weight", w(d, d, dtype=np.float16))
    add_matmul(p + "ffn_up.weight", w(2 * n_ff, d, dtype=np.float16))
    add_matmul(p + "ffn_down.weight", w(d, n_ff, dtype=np.float16))
    wr.add_tensor(p + "ffn_norm.weight", ones(d))
wr.add_tensor("type_emb.weight", w(3, d))
for il in range(n_head_layers):
    p = f"head.{il}."
    wr.add_tensor(p + "attn_norm.weight", ones(d))
    wr.add_tensor(p + "attn_norm.bias", w(d))
    wr.add_tensor(p + "attn_qkv.weight", w(3 * d, d, dtype=np.float16))
    wr.add_tensor(p + "attn_qkv.bias", w(3 * d))
    wr.add_tensor(p + "attn_output.weight", w(d, d, dtype=np.float16))
    wr.add_tensor(p + "attn_output.bias", w(d))
    wr.add_tensor(p + "ffn_norm.weight", ones(d))
    wr.add_tensor(p + "ffn_norm.bias", w(d))
    wr.add_tensor(p + "ffn_up.weight", w(4 * d, d, dtype=np.float16))
    wr.add_tensor(p + "ffn_up.bias", w(4 * d))
    wr.add_tensor(p + "ffn_down.weight", w(d, 4 * d, dtype=np.float16))
    wr.add_tensor(p + "ffn_down.bias", w(d))
wr.add_tensor("scorer.0.weight", ones(d))
wr.add_tensor("scorer.0.bias", w(d))
wr.add_tensor("scorer.1.weight", w(d, d, dtype=np.float16))
wr.add_tensor("scorer.1.bias", w(d))
wr.add_tensor("scorer.3.weight", w(1, d))
wr.add_tensor("scorer.3.bias", w(1))
wr.add_tensor("act_head.0.weight", w(256, d + 4))
wr.add_tensor("act_head.0.bias", w(256))
wr.add_tensor("act_head.2.weight", w(n_act, 256))
wr.add_tensor("act_head.2.bias", w(n_act))

wr.write_header_to_file()
wr.write_kv_data_to_file()
wr.write_tensors_to_file()
wr.close()
print("wrote", out, "n_vocab", n_vocab)
