from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, logger
from .kimi_linear import KimiLinearModel


@ModelBase.register("KimiK3ForConditionalGeneration")
class KimiK3Model(KimiLinearModel):
    """Kimi K3: hybrid KDA + gated-MLA (NoPE) with Attention Residuals and Stable LatentMoE.

    Text config is `kimi_linear` with K3 extensions:
      - SiTU-GLU activation (soft-capped SiLU) in dense MLP, shared and routed experts
      - AttnRes: residual-stream snapshot bank every `attn_res_block_size` layers,
        softmax mixtures before attention, before MLP and at model output
      - Stable LatentMoE: routed experts run in a `routed_expert_hidden_size` latent
        space (down proj -> experts -> weighted sum -> RMSNorm -> up proj)
      - KDA safe gate: g_log = gate_lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
        with a full-rank output gate g_proj instead of the low-rank g_a/g_b pair
      - MLA output gate: attn = attn * sigmoid(g_proj(x)) before o_proj
    Routed expert weights are MXFP4 (compressed-tensors), dequantized in ModelBase.
    """
    model_arch = gguf.MODEL_ARCH.KIMI_K3

    def set_vocab(self):
        super().set_vocab()
        # KimiLinearModel.set_vocab forces the tokenizer's own eos, which for K3
        # is 163585 = [EOS], the document terminator. The config says 163586 =
        # <|end_of_msg|>, the chat turn terminator; keeping [EOS] means chat
        # generation never stops at the end of an assistant turn. Restore it.
        if (eos := self.hparams.get("eos_token_id")) is not None:
            self.gguf_writer.add_eos_token_id(eos)

        # Moonshot ships no chat_template in tokenizer_config.json (K3 is
        # API-first), so GGUFs come out template-less and chat tools refuse to
        # run. Embed the reference template from models/templates/Kimi-K3.jinja
        # unless the checkpoint provides one.
        has_template = (self.dir_model / "chat_template.jinja").is_file()
        if not has_template:
            try:
                with open(self.dir_model / "tokenizer_config.json", encoding="utf-8") as f:
                    has_template = "chat_template" in json.load(f)
            except OSError:
                pass
        if not has_template:
            tmpl = Path(__file__).resolve().parent.parent / "models" / "templates" / "Kimi-K3.jinja"
            if tmpl.is_file():
                logger.info("embedding reference chat template from models/templates/Kimi-K3.jinja")
                self.gguf_writer.add_chat_template(tmpl.read_text(encoding="utf-8"))

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # Stable LatentMoE
        self.gguf_writer.add_moe_latent_size(self.hparams["routed_expert_hidden_size"])

        # SiTU-GLU activation parameters
        self.gguf_writer.add_situ_beta(self.hparams["activation_situ_beta"])
        self.gguf_writer.add_situ_linear_beta(self.hparams["activation_situ_linear_beta"])

        # Attention residuals
        self.gguf_writer.add_attn_res_block_size(self.hparams["attn_res_block_size"])

        # KDA safe gate lower bound
        self.gguf_writer.add_kda_gate_lower_bound(self.hparams["linear_attn_config"]["gate_lower_bound"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # text-only conversion: vision tensors are handled by the mmproj path
        if name.startswith(("vision_tower.", "mm_projector.")):
            return

        name = name.removeprefix("language_model.")

        # K3 checkpoints store A_log as [head_dim] (128) but only the first
        # num_heads (96) entries are used. The safe-gate formula is
        #   g_log = gate_lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
        # so we store exp(A_log) directly (unlike Kimi-Linear's -exp(A_log)).
        if name.endswith(".A_log"):
            n_head = self.hparams["num_attention_heads"]
            data_torch = torch.exp(data_torch.float()[:n_head])
            # skip KimiLinearModel's -exp(A_log) handling
            yield from super(KimiLinearModel, self).modify_tensors(data_torch, name, bid)
            return

        # res projections are stored as [1, n_embd]: flatten to [n_embd]
        if name.endswith(("_res_proj.weight", "_res_norm.weight")):
            data_torch = data_torch.reshape(-1)

        yield from super().modify_tensors(data_torch, name, bid)
