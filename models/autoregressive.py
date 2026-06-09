"""Autoregressive language model (AR) built on the shared Transformer backbone.

Provides the DDIT base class and the AR subclass with KV-cache inference support.
The AR model serves as both a standalone next-token predictor and the teacher for
push-forward distillation.

Based on https://github.com/kuleshov-group/mdlm
"""

import typing

import huggingface_hub
import omegaconf
import torch
import torch.nn as nn

from .transformer import (
    DDiTBlock,
    DDitFinalLayer,
    EmbeddingLayer,
    Rotary,
    bias_dropout_add_scale_fused_inference,
    bias_dropout_add_scale_fused_train,
)


class DDIT(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """Base causal Transformer with rotary embeddings and FlashAttention blocks."""

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size
        self.causal = (
            hasattr(config.model, "causal") and config.model.causal
        )
        assert self.causal

        self.vocab_embed = EmbeddingLayer(
            config.model.hidden_size, vocab_size
        )
        self.rotary_emb = Rotary(
            config.model.hidden_size // config.model.n_heads
        )

        blocks = []
        for _ in range(config.model.n_blocks):
            blocks.append(
                DDiTBlock(
                    config.model.hidden_size,
                    config.model.n_heads,
                    config.model.cond_dim,
                    dropout=config.model.dropout,
                    causal=self.causal,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.output_layer = DDitFinalLayer(
            config.model.hidden_size,
            vocab_size,
            config.model.cond_dim,
            causal=self.causal,
        )
        self.scale_by_sigma = config.model.scale_by_sigma

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference


class AR(DDIT):
    """Autoregressive next-token predictor with KV-cache support."""

    def __init__(self, config, vocab_size: int, mask_index: int):
        super().__init__(config, vocab_size)
        self.mask_index = mask_index
        self.neg_infinity = -1000.0

    def forward(
        self,
        xt: torch.Tensor,
        sigma: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Full-sequence forward pass returning log-probabilities."""
        x = self.vocab_embed(xt)
        rotary_cos_sin = self.rotary_emb(x)

        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, None, seqlens=None)
            output = self.output_layer(x, None)

        if self.mask_index is not None:
            output[:, :, self.mask_index] = self.neg_infinity

        if temperature is not None:
            if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
                temperature = temperature.view(-1, 1, 1)
            output = output.float() / (temperature + 1e-7)

        return output.float().log_softmax(-1)

    def infer_next_token(
        self,
        xt: torch.Tensor,
        kv_caches: typing.Optional[list] = None,
        offset: int = 0,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, list]:
        """Single-step forward with KV-cache for autoregressive decoding."""
        x = self.vocab_embed(xt)
        cos, sin = self.rotary_emb.forward_with_offset(x, offset)
        rotary_cos_sin = (cos, sin)

        new_kv_caches = []
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for i, block in enumerate(self.blocks):
                past_kv = kv_caches[i] if kv_caches is not None else None
                x, kv = block.forward_inference(x, rotary_cos_sin, kv_cache=past_kv)
                new_kv_caches.append(kv)

            logits = self.output_layer(x, None)

        if self.mask_index is not None:
            logits[:, :, self.mask_index] = self.neg_infinity

        if temperature is not None:
            if isinstance(temperature, torch.Tensor) and temperature.ndim == 1:
                temperature = temperature.view(-1, 1, 1)
            logits = logits.float() / (temperature + 1e-7)

        return logits.float().log_softmax(-1), new_kv_caches

    def forward_high_precision(
        self,
        xt: torch.Tensor,
        sigma: float = 0.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Full-sequence forward in float32 (no AMP), used by noise inversion."""
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
            x = self.vocab_embed(xt)
            rotary_cos_sin = self.rotary_emb(x)

            for i in range(len(self.blocks)):
                x, _ = self.blocks[i].forward_inference(x, rotary_cos_sin)
            output = self.output_layer(x, None)

            if self.mask_index is not None:
                output[:, :, self.mask_index] = self.neg_infinity

            return output
