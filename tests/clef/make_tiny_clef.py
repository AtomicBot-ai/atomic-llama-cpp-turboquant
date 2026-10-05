#!/usr/bin/env python3
# tiny random Clef GGUF (< 1 MB) for offline tests of the clef engine of llama-server --decision
# usage: python tests/clef/make_tiny_clef.py out.gguf
# Qwen3.5 backbone of 4 blocks (3 gated delta net, 1 full attention) and a joint head of 2 routing + 2 joint
# blocks, the tensor names and keys of conversion/clef.py. The tokenizer has the special tokens of the prompt
# and byte-level chars without merges, so every character is a token (the prompt rules are tested with the
# real vocabulary by test-decision-clef).
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf  # noqa: E402

out = sys.argv[1]
rng = np.random.default_rng(0)

d, n_layer, n_ff = 64, 4, 128
n_head, n_head_kv, head_dim, rope_dim = 2, 1, 32, 16
conv_kernel, head_k, n_k_heads, n_v_heads = 4, 16, 2, 2
h, h_ff, h_heads, n_routing, n_joint = 32, 64, 2, 2, 2

# byte-level chars (gpt-2 byte_to_unicode)
bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
cs = bs[:]
n = 0
for b in range(256):
    if b not in bs:
        bs.append(b)
        cs.append(256 + n)
        n += 1
chars = [chr(c) for c in cs]
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
added = ["<think>", "</think>"]
tokens = chars + specials + added
ttype = [gguf.TokenType.NORMAL] * len(chars) + [gguf.TokenType.CONTROL] * len(specials) + [gguf.TokenType.USER_DEFINED] * len(added)
n_vocab = len(tokens)

w = gguf.GGUFWriter(out, "clef")
w.add_name("tiny-clef")
w.add_context_length(4096)
w.add_embedding_length(d)
w.add_block_count(n_layer)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_layer_norm_rms_eps(1e-6)
w.add_rope_dimension_count(rope_dim)
w.add_rope_dimension_sections([3, 3, 2, 0])
w.add_rope_freq_base(1e7)
w.add_ssm_conv_kernel(conv_kernel)
w.add_ssm_inner_size(head_k * n_v_heads)
w.add_ssm_state_size(head_k)
w.add_ssm_time_step_rank(n_v_heads)
w.add_ssm_group_count(n_k_heads)
w.add_full_attention_interval(4)
w.add_decision_type(gguf.DecisionType.CLEF)
w.add_decision_routing_block_count(n_routing)
w.add_decision_block_count(n_joint)
w.add_decision_head_count(h_heads)
w.add_layer_norm_eps(1e-5)

w.add_tokenizer_model("gpt2")
w.add_tokenizer_pre("qwen35")
w.add_token_list(tokens)
w.add_token_types(ttype)
w.add_token_merges(["\u0120 \u0120"])
w.add_eos_token_id(tokens.index("<|im_end|>"))
w.add_pad_token_id(tokens.index("<|endoftext|>"))
w.add_add_bos_token(False)


def t(name, *ne, scale=0.05, value=None):
    # ne in ggml order (ne0 first); numpy wants the reverse
    if value is not None:
        a = np.full(tuple(reversed(ne)), value, dtype=np.float32)
    else:
        a = (rng.standard_normal(tuple(reversed(ne))) * scale).astype(np.float32)
    w.add_tensor(name, a)


t("token_embd.weight", d, n_vocab, scale=0.5)
t("output_norm.weight", d, value=1.0)
t("output.weight", d, n_vocab, scale=0.5)
key_dim, value_dim = head_k * n_k_heads, head_k * n_v_heads
for il in range(n_layer):
    p = f"blk.{il}."
    t(p + "attn_norm.weight", d, value=1.0)
    t(p + "post_attention_norm.weight", d, value=1.0)
    if (il + 1) % 4 == 0:
        t(p + "attn_q.weight", d, head_dim * n_head * 2)
        t(p + "attn_k.weight", d, head_dim * n_head_kv)
        t(p + "attn_v.weight", d, head_dim * n_head_kv)
        t(p + "attn_output.weight", head_dim * n_head, d)
        t(p + "attn_q_norm.weight", head_dim, value=1.0)
        t(p + "attn_k_norm.weight", head_dim, value=1.0)
    else:
        t(p + "attn_qkv.weight", d, key_dim * 2 + value_dim)
        t(p + "attn_gate.weight", d, value_dim)
        t(p + "ssm_conv1d.weight", conv_kernel, key_dim * 2 + value_dim, scale=0.3)
        t(p + "ssm_dt.bias", n_v_heads, value=0.0)
        t(p + "ssm_a", n_v_heads, value=-1.0)
        t(p + "ssm_beta.weight", d, n_v_heads)
        t(p + "ssm_alpha.weight", d, n_v_heads)
        t(p + "ssm_norm.weight", head_k, value=1.0)
        t(p + "ssm_out.weight", value_dim, d)
    t(p + "ffn_gate.weight", d, n_ff)
    t(p + "ffn_up.weight", d, n_ff)
    t(p + "ffn_down.weight", n_ff, d)


def norm(name, size):
    t(name + ".weight", size, value=1.0)
    t(name + ".bias", size, value=0.0)


def attn(prefix):
    for x in ("q", "k", "v", "o"):
        t(f"{prefix}_{x}.weight", h, h, scale=0.2)
        t(f"{prefix}_{x}.bias", h, scale=0.02)


for il in range(n_routing + n_joint):
    p = f"dec.blk.{il}."
    if il < n_routing:
        norm(p + "cross_attn_norm_kv", h)
    else:
        norm(p + "attn_norm", h)
        attn(p + "attn")
    norm(p + "cross_attn_norm", h)
    attn(p + "cross_attn")
    norm(p + "ffn_norm", h)
    t(p + "ffn_up.weight", h, h_ff, scale=0.2)
    t(p + "ffn_up.bias", h_ff, scale=0.02)
    t(p + "ffn_down.weight", h_ff, h, scale=0.2)
    t(p + "ffn_down.bias", h, scale=0.02)

norm("decision.hidden_norm", d)
for name in ("option_summary_norm", "field_norm", "option_norm"):
    norm("decision." + name, h)
for name in ("memory", "question", "option_question", "global", "option_context", "option_lexical"):
    t(f"decision.proj_{name}.weight", d, h, scale=0.2)
w.add_tensor("decision.scales", np.array([2.0, 3.0, 0.5], dtype=np.float32))
t("token_types.weight", h, 3, scale=0.2)
t("decision.scorer.weight", 4 * h, h, scale=0.2)
t("decision.scorer.bias", h, scale=0.02)
t("decision.scorer_out.weight", h, 1, scale=0.5)
t("decision.scorer_out.bias", 1, value=0.0)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote {out}: vocab {n_vocab}")
