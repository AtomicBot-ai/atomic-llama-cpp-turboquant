from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
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
