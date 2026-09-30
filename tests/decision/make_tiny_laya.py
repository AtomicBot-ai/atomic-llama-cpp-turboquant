# tiny random laya GGUF (< 1 MB) for offline smoke tests of llama-laya-cli and the decision server
# usage: python tests/decision/make_tiny_laya.py out.gguf [max_len [head_max_len]]
# (the tokenizer has no merges, so every character is a token; the server tests use a longer max_len)
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

out = sys.argv[1]
max_len = int(sys.argv[2]) if len(sys.argv) > 2 else 128
head_max_len = int(sys.argv[3]) if len(sys.argv) > 3 else 64
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
n_vocab = len(tokens)


def w(*shape, dtype=np.float32):
    return (rng.standard_normal(shape) * 0.05).astype(dtype)


def ones(n):
    return np.ones(n, dtype=np.float32)


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
wr.add_head_layers(n_head_layers)
wr.add_n_qtype(3)
wr.add_marker_token_id(4)
wr.add_max_len(max_len)
wr.add_head_max_len(head_max_len)
wr.add_temperature([1.0, 1.5, 0.8])
wr.add_act_classes(n_act)
wr.add_tokenizer_model("gpt2")
wr.add_token_list(tokens)
wr.add_token_types(ttype)
wr.add_token_merges([])
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
    wr.add_tensor(p + "attn_qkv.weight", w(3 * d, d, dtype=np.float16))
    wr.add_tensor(p + "attn_output.weight", w(d, d, dtype=np.float16))
    wr.add_tensor(p + "ffn_up.weight", w(2 * n_ff, d, dtype=np.float16))
    wr.add_tensor(p + "ffn_down.weight", w(d, n_ff, dtype=np.float16))
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
