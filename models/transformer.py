"""Shared Transformer backbone components for K-Forcing.

Provides the core building blocks used by both the autoregressive (AR) teacher
and the push-forward language model (PFLM) student: rotary embeddings,
layer normalization, attention blocks with FlashAttention and KV-cache support,
and embedding/output layers.

Based on https://github.com/kuleshov-group/mdlm
"""

import math
import typing

import flash_attn
import flash_attn.layers.rotary
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)


# ---------------------------------------------------------------------------
# Fused bias-dropout-scale helpers
# ---------------------------------------------------------------------------

def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool,
) -> torch.Tensor:
    """Fused bias + dropout + scale + residual add."""
    if bias is not None:
        out = scale * F.dropout(x + bias, p=prob, training=training)
    else:
        out = scale * F.dropout(x, p=prob, training=training)
    if residual is not None:
        out = residual + out
    return out


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
) -> torch.Tensor:
    return bias_dropout_add_scale(x, bias, scale, residual, prob, False)


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    qkv: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embeddings to packed QKV via flash_attn helper."""
    cos = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    return flash_attn.layers.rotary.apply_rotary_emb_qkv_(qkv, cos, sin)


class Rotary(nn.Module):
    """Rotary position embedding with offset support for KV-cache inference."""

    def __init__(self, dim: int, base: int = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(
        self, x: torch.Tensor, seq_dim: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
            # Identity transform on V
            self.cos_cached[:, :, 2, :, :].fill_(1.0)
            self.sin_cached[:, :, 2, :, :].fill_(0.0)
        return self.cos_cached, self.sin_cached

    def forward_with_offset(
        self, x: torch.Tensor, offset: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE starting from a position offset (for KV-cache decoding)."""
        seq_len = x.shape[1]
        t = torch.arange(offset, offset + seq_len, device=x.device).type_as(
            self.inv_freq
        )
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos = emb.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        sin = emb.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
        cos[:, :, 2, :, :].fill_(1.0)
        sin[:, :, 2, :, :].fill_(0.0)
        return cos, sin


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """Layer normalization that always computes in float32."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x = F.layer_norm(x.float(), [self.dim])
        return x * self.weight[None, None, :]


class EmbeddingLayer(nn.Module):
    """Learnable token embedding table."""

    def __init__(self, dim: int, vocab_dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding[x]


def residual_linear(
    x: torch.Tensor,
    W: torch.Tensor,
    x_skip: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale,
    ).view(*x.shape[:-1], dim_out)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class DDiTBlock(nn.Module):
    """Causal Transformer block with FlashAttention and KV-cache support.

    Training uses FlashAttention via ``forward()``.  Inference with KV-cache
    uses ``forward_inference()`` which falls back to PyTorch SDPA.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.causal = causal

        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        return bias_dropout_add_scale_fused_inference

    def forward(
        self,
        x: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        c: typing.Optional[torch.Tensor],
        seqlens: typing.Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass using FlashAttention (training)."""
        batch_size, seq_len = x.shape[0], x.shape[1]
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x_skip = x
        x = self.norm1(x)

        # float32 matmul to avoid cuBLAS non-determinism
        with torch.cuda.amp.autocast(enabled=False):
            qkv = self.attn_qkv(x.float())

        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if x.is_cuda and torch.is_autocast_enabled():
            qkv = qkv.to(torch.get_autocast_gpu_dtype())

        qkv = rearrange(qkv, "b s ... -> (b s) ...")
        if seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seq_len,
                step=seq_len,
                dtype=torch.int32,
                device=qkv.device,
            )
        else:
            cu_seqlens = seqlens.cumsum(-1)
        x = flash_attn.flash_attn_interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, 0.0, causal=self.causal
        )
        x = rearrange(x, "(b s) h d -> b s (h d)", b=batch_size)

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, scale, x_skip, self.dropout
        )

        with torch.cuda.amp.autocast(enabled=False):
            x_in = self.mlp(self.norm2(x))

        x = bias_dropout_scale_fn(x_in, None, scale, x, self.dropout)
        return x

    def forward_inference(
        self,
        x: torch.Tensor,
        rotary_cos_sin: tuple[torch.Tensor, torch.Tensor],
        kv_cache: typing.Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV-cache (inference)."""
        x_skip = x
        x = self.norm1(x)

        with torch.cuda.amp.autocast(enabled=False):
            qkv = self.attn_qkv(x.float())

        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads
        )
        cos, sin = rotary_cos_sin
        qkv = apply_rotary_pos_emb(qkv, cos, sin)

        if x.is_cuda and torch.is_autocast_enabled():
            qkv = qkv.to(torch.get_autocast_gpu_dtype())

        q, k, v = qkv.unbind(dim=2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)
        new_kv = (k, v)

        q = rearrange(q, "b s h d -> b h s d")
        k = rearrange(k, "b s h d -> b h s d")
        v = rearrange(v, "b s h d -> b h s d")

        attn_out = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal if q.shape[2] > 1 else False
        )
        attn_out = rearrange(attn_out, "b h s d -> b s (h d)")

        scale = torch.ones(1, device=x.device, dtype=x.dtype)
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        x = bias_dropout_scale_fn(
            self.attn_out(attn_out), None, scale, x_skip, self.dropout
        )
        with torch.cuda.amp.autocast(enabled=False):
            mlp_out = self.mlp(self.norm2(x))
        x = bias_dropout_scale_fn(mlp_out, None, scale, x, self.dropout)
        return x, new_kv


# ---------------------------------------------------------------------------
# Output layer
# ---------------------------------------------------------------------------

class DDitFinalLayer(nn.Module):
    """Final norm + linear projection for causal models."""

    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        cond_dim: int,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        assert causal

        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(
        self, x: torch.Tensor, c: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.linear(self.norm_final(x))
