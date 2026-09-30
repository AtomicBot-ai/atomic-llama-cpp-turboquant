from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .bert import ModernBertModel


@ModelBase.register_hparams_loader(lambda dir_model: (dir_model / "rl_agent_config.json").is_file())
def _load_laya_hparams(dir_model: Path) -> dict[str, Any]:
    """Laya checkpoints ship no root config.json; the encoder config lives in
    encoder/config.json and the RL-agent hyperparameters in rl_agent_config.json."""
    with open(dir_model / "rl_agent_config.json", encoding="utf-8") as f:
        rl_cfg = json.load(f)

    enc_cfg: dict[str, Any] = {}
    enc_cfg_path = dir_model / "encoder" / "config.json"
    if enc_cfg_path.is_file():
        with open(enc_cfg_path, encoding="utf-8") as f:
            enc_cfg = json.load(f)

    hparams = {
        **enc_cfg,
        # force the laya architecture (root config is absent)
        "architectures": ["LayaModel"],
        "model_type": "laya",
    }
    # RL-agent decision head hyperparameters (defaults as in laya.agent)
    hparams["head_layers"]   = rl_cfg.get("head_layers", 2)
    hparams["max_len"]       = rl_cfg.get("max_len", 512)
    hparams["head_max_len"]  = rl_cfg.get("head_max_len", 192)
    hparams["n_qtype"]       = 3  # choice / score / noul
    hparams["temperature"]   = rl_cfg.get("temperature", [1.0, 1.0, 1.0])
    # "<qtype>:<bucket>" -> T, applied before the per-qtype temperature (laya.agent._decode_answers)
    hparams["temperature_by_options"] = rl_cfg.get("temperature_by_options", {})
    hparams["act_classes"]   = len(rl_cfg.get("act_costs", {})) + 1
    return hparams


@ModelBase.register("LayaModel")
class LayaModel(ModernBertModel):
    model_arch = gguf.MODEL_ARCH.LAYA

    # laya checkpoints keep the tokenizer files under a tokenizer/ subdirectory
    @property
    def _tokenizer_dir(self) -> Path:
        tok_dir = self.dir_model / "tokenizer"
        return tok_dir if tok_dir.is_dir() else self.dir_model

    def _tokenizer_json(self) -> dict[str, Any]:
        with open(self._tokenizer_dir / "tokenizer.json", encoding="utf-8") as f:
            return json.load(f)

    def _tokenizer_kind(self, tj: dict[str, Any]) -> str:
        """metaspace-bpe (laya-multilingual, mmBERT) or bytelevel-bpe (laya, laya-typed-decisions,
        ModernBERT / OLMo BPE): the two tokenizers tools/laya ports. Anything else is refused."""
        pre = tj.get("pre_tokenizer") or {}
        model = tj.get("model") or {}
        if model.get("type") != "BPE" or model.get("dropout") or model.get("continuing_subword_prefix") or model.get("end_of_word_suffix"):
            raise ValueError("laya: unsupported tokenizer model %r" % {k: v for k, v in model.items() if k not in ("vocab", "merges")})
        if pre.get("type") == "Metaspace":
            return "metaspace-bpe"
        if pre.get("type") == "ByteLevel":
            if pre.get("add_prefix_space") or not pre.get("use_regex", True):
                raise ValueError("laya: unsupported ByteLevel pre-tokenizer %r" % pre)
            if model.get("byte_fallback") or model.get("ignore_merges"):
                raise ValueError("laya: unsupported byte-level BPE options %r" % {k: v for k, v in model.items() if k not in ("vocab", "merges")})
            return "bytelevel-bpe"
        raise ValueError("laya: unsupported pre-tokenizer %r" % pre)

    def _token_id(self, tj: dict[str, Any], content: str) -> int:
        for at in tj.get("added_tokens", []):
            if at["content"] == content:
                return at["id"]
        if content in tj["model"]["vocab"]:
            return tj["model"]["vocab"][content]
        raise ValueError("laya: token %r is not in the vocabulary" % content)

    def set_vocab(self):
        tj = self._tokenizer_json()
        with open(self._tokenizer_dir / "tokenizer_config.json", encoding="utf-8") as f:
            tcfg = json.load(f)

        def special(name: str) -> int:
            tok = tcfg.get(name)
            if isinstance(tok, dict):
                tok = tok.get("content")
            if not isinstance(tok, str):
                raise ValueError("laya: tokenizer_config.json has no %s" % name)
            return self._token_id(tj, tok)

        # laya.common.build_sequence uses tok.cls_token_id, tok.sep_token_id and tok.mask_token_id
        cls_id, sep_id, mask_id = special("cls_token"), special("sep_token"), special("mask_token")

        self.gguf_writer.add_marker_token_id(mask_id)
        self._set_vocab_gpt2()

        # the special ids tools/laya reads: bos = [CLS], eos = sep = [SEP], mask, unk, pad
        ids = {"bos": cls_id, "eos": sep_id, "sep": sep_id, "mask": mask_id}
        if "unk_token" in tcfg:
            ids["unk"] = special("unk_token")
        if "pad_token" in tcfg:
            ids["pad"] = special("pad_token")
        self._special_vocab.special_token_ids.pop("cls", None)
        self._special_vocab.special_token_ids.update(ids)
        self._special_vocab.add_to_gguf(self.gguf_writer)

        kind = self._tokenizer_kind(tj)
        self.gguf_writer.add_string("decision.laya.tokenizer", kind)
        if kind == "bytelevel-bpe":
            norm = tj.get("normalizer")
            if norm is None:
                self.gguf_writer.add_string("decision.laya.normalizer", "none")
            elif norm.get("type") == "NFC":
                self.gguf_writer.add_string("decision.laya.normalizer", "nfc")
            else:
                raise ValueError("laya: unsupported normalizer %r" % norm)
            # HF AddedToken flags as (id, flags) pairs: bit 0 lstrip, bit 1 normalized
            flags: list[int] = []
            for at in tj.get("added_tokens", []):
                if at.get("rstrip") or at.get("single_word"):
                    raise ValueError("laya: added token %r uses rstrip / single_word" % at["content"])
                flags += [at["id"], (1 if at.get("lstrip") else 0) | (2 if at.get("normalized") else 0)]
            self.gguf_writer.add_array("decision.laya.added_tokens", flags)
        logger.info("laya: tokenizer %s, [CLS] %d, [SEP] %d, [MASK] %d", kind, cls_id, sep_id, mask_id)

    def get_vocab_base(self) -> tuple[list[str], list[int], str]:
        # tokenizer.json / tokenizer_config.json live in the tokenizer/ subdir
        saved = self.dir_model
        self.dir_model = self._tokenizer_dir
        try:
            return super().get_vocab_base()
        finally:
            self.dir_model = saved

    def _set_vocab_gpt2(self) -> None:
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # written by set_vocab once the special ids are fixed
        self._special_vocab = gguf.SpecialVocab(self._tokenizer_dir, load_merges=True)

        # the laya build_sequence always wraps with [CLS]/[SEP], so the llama.cpp
        # side must add bos/sep the same way; SpecialVocab may have turned sep off
        self._special_vocab.add_special_token.update({"bos": True, "eos": True, "sep": True})

    def get_vocab_base_pre(self, tokenizer) -> str:
        # "modern-bert" in llama.cpp is the English ModernBERT (OLMo BPE, GPT-2 regex). The multilingual
        # checkpoint (mmBERT, Metaspace) keeps the same name; decision.laya.tokenizer tells them apart.
        return "modern-bert"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # decision head hyperparameters
        self.gguf_writer.add_head_layers(self.hparams["head_layers"])
        self.gguf_writer.add_n_qtype(self.hparams["n_qtype"])
        self.gguf_writer.add_max_len(self.hparams["max_len"])
        self.gguf_writer.add_head_max_len(self.hparams["head_max_len"])
        self.gguf_writer.add_temperature(self.hparams["temperature"])
        self.gguf_writer.add_act_classes(self.hparams["act_classes"])
        tbo = self.hparams.get("temperature_by_options") or {}
        if tbo:
            arch = gguf.MODEL_ARCH_NAMES[self.model_arch]
            self.gguf_writer.add_array(f"{arch}.temperature_by_options.buckets", list(tbo.keys()))
            self.gguf_writer.add_array(f"{arch}.temperature_by_options.values", [float(v) for v in tbo.values()])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # the temperature buffer is persisted as a KV array, not as a tensor
        if name == "temperature":
            return None

        # encoder tensors reuse the modern-bert naming (strip the encoder. prefix)
        if name.startswith("encoder."):
            name = name[len("encoder."):]

        # PyTorch nn.MultiheadAttention uses in_proj_weight / in_proj_bias; the
        # laya decision head maps them onto the fused qkv gguf tensor.
        if name.endswith("self_attn.in_proj_weight"):
            name = name[:-len("in_proj_weight")] + "in_proj.weight"
        elif name.endswith("self_attn.in_proj_bias"):
            name = name[:-len("in_proj_bias")] + "in_proj.bias"

        return name, gen
