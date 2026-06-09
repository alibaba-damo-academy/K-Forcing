"""Push-forward language model (PFLM) for joint multi-token decoding.

Provides FlexDDiTBlock (flexible attention mask support), PointwiseNoiseEncoder,
and the MTP class that maps k i.i.d. Uniform(0,1) noise variables to k future
tokens in a single forward pass.

Based on https://github.com/kuleshov-group/mdlm
"""

import math
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from einops import rearrange
from torch import amp
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint

from .autoregressive import AR, DDIT, DDiTBlock
from .transformer import LayerNorm, apply_rotary_pos_emb

class FlexDDiTBlock(DDiTBlock):
    """
    A flexible and FAITHFUL manual implementation of DDiTBlock that replicates
    all original operations while replacing FlashAttention with a standard
    attention mechanism that supports arbitrary masks.
    """
    def __init__(self, *args, **kwargs):
        # The 'causal' arg from DDiTBlock is no longer used by this forward pass,
        # but we accept it for signature compatibility during initialization.
        if 'causal' in kwargs:
            kwargs.pop('causal')
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, rotary_cos_sin, c, attn_mask: torch.Tensor = None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            rotary_cos_sin (tuple): Tuple of cosine and sine tensors for RoPE.
            c (torch.Tensor): Conditioning tensor. This is IGNORED by this block to strictly
                              match the provided original implementation but is kept for
                              signature compatibility.
            attn_mask (torch.Tensor, optional):
                Boolean mask where True indicates attention is allowed.
                Shape: (batch, seq_len, seq_len) or broadcastable.
        """
        # --- 1. Self-Attention Block ---
        x_skip = x

        # Pre-normalization. `c` is NOT used, strictly matching the original.
        x_norm = self.norm1(x)

        # QKV projection, executed in float32 to avoid bugs, as in original.
        with torch.cuda.amp.autocast(enabled=False):
            qkv = self.attn_qkv(x_norm.float())

        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        # Apply rotary positional embeddings, also in float32.
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            # You can swap this with `manual_apply_rotary_pos_emb` if you want
            # to remove all flash-attn dependencies.
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if x.is_cuda and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            qkv = qkv.to(target_dtype)

        q, k, v = qkv.unbind(dim=2)
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')
        
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)  # (B, S, S) -> (B, 1, S, S)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False  
        )

        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

        x = x_skip + self.dropout1(self.attn_out(attn_output))

        # --- 2. MLP Block ---
        x_skip = x

        # Pre-normalization for MLP. `c` is NOT used.
        x_norm = self.norm2(x)

        # Manual equivalent of `bias_dropout_add_scale_fn` for MLP
        # operation: residual + dropout(MLP(norm(x)))
        x = x_skip + self.dropout2(self.mlp(x_norm))

        return x

    def custom_forward_train(self, hidden_states_context: torch.Tensor, rotary_cos_sin_context: tuple, 
                            hidden_states_future: torch.Tensor, rotary_cos_sin_future: tuple,
                            attn_mask: torch.Tensor, return_kv: bool = False):
        """
        Args:
            ... existing args ...
            return_kv (bool): If True, returns the context KV tensors for caching.
        """
        B, N, H = hidden_states_context.shape
        _, Nk, _ = hidden_states_future.shape
        device = hidden_states_context.device

        # 1. Normalization
        context_norm = self.norm1(hidden_states_context)
        future_norm = self.norm1(hidden_states_future)

        # 2. QKV Projections
        with torch.cuda.amp.autocast(enabled=False):
            qkv_ctx = self.attn_qkv(context_norm.float())
            qkv_fut = self.attn_qkv(future_norm.float())
            
            qkv_ctx = rearrange(qkv_ctx, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
            qkv_fut = rearrange(qkv_fut, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

            # Apply RoPE
            cos_ctx, sin_ctx = rotary_cos_sin_context
            qkv_ctx = apply_rotary_pos_emb(qkv_ctx, cos_ctx.to(qkv_ctx.dtype), sin_ctx.to(qkv_ctx.dtype))
            
            cos_fut, sin_fut = rotary_cos_sin_future
            qkv_fut = apply_rotary_pos_emb(qkv_fut, cos_fut.to(qkv_fut.dtype), sin_fut.to(qkv_fut.dtype))

        if hidden_states_context.is_cuda and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            qkv_ctx, qkv_fut = qkv_ctx.to(target_dtype), qkv_fut.to(target_dtype)

        q_ctx, k_ctx, v_ctx = qkv_ctx.unbind(dim=2)
        q_fut, k_fut, v_fut = qkv_fut.unbind(dim=2)

        # 3. Concatenate for Attention
        # Heads dimension needs to be moved to dim 1 for SDPA
        q_all = torch.cat([rearrange(q_ctx, 'b s h d -> b h s d'), rearrange(q_fut, 'b s h d -> b h s d')], dim=2)
        k_all = torch.cat([rearrange(k_ctx, 'b s h d -> b h s d'), rearrange(k_fut, 'b s h d -> b h s d')], dim=2)
        v_all = torch.cat([rearrange(v_ctx, 'b s h d -> b h s d'), rearrange(v_fut, 'b s h d -> b h s d')], dim=2)

        # 4. SDPA
        attn_out = F.scaled_dot_product_attention(q_all, k_all, v_all, attn_mask=attn_mask)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        
        res_ctx = attn_out[:, :N, :]
        res_fut = attn_out[:, N:, :]

        # 5. Residual + MLP
        out_ctx = hidden_states_context + self.dropout1(self.attn_out(res_ctx))
        out_ctx = out_ctx + self.dropout2(self.mlp(self.norm2(out_ctx)))
        
        out_fut = hidden_states_future + self.dropout1(self.attn_out(res_fut))
        out_fut = out_fut + self.dropout2(self.mlp(self.norm2(out_fut)))

        if return_kv:
            # We return context KV in (B, H, S, D) format compatible with inference/double_forward
            # k_ctx is [B, N, H, D] -> rearrange to [B, H, N, D]
            return out_ctx, out_fut, (rearrange(k_ctx, 'b s h d -> b h s d'), rearrange(v_ctx, 'b s h d -> b h s d'))
        
        return out_ctx, out_fut



    def custom_forward_inference(self, x: torch.Tensor, rotary_cos_sin: tuple, 
                                N_new: int, kv_cache: typing.Optional[tuple] = None,
                                attn_mask: torch.Tensor = None):
        """
        x: [B, N_new + k_future, H] 
        rotary_cos_sin: (cos, sin) pre-computed for (N_new + k_future) positions
        """
        x_skip = x
        
        # 1. Fused Norm and Projection
        x_norm = self.norm1(x)
        with torch.cuda.amp.autocast(enabled=False):
            qkv = self.attn_qkv(x_norm.float())
        
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        # 2. Apply RoPE to the ENTIRE sequence (Context + Hints)
        # We assume rotary_cos_sin already contains the correct offsets
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))

        if x.is_cuda and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            qkv = qkv.to(target_dtype)

        # 3. Separate Q, K, V
        q, k, v = qkv.unbind(dim=2) # q, k, v are [B, S_total_new, H, D]

        # 4. KV Cache update (Only update with CONTEXT tokens, not Hints)
        # The hints (y1, y2) are transient and should NOT be stored in the long-term KV cache
        q_all = q
        k_context_new = k[:, :N_new]
        v_context_new = v[:, :N_new]
        
        # Hints K/V (used for current attention but not saved)
        k_hints = k[:, N_new:]
        v_hints = v[:, N_new:]

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k_context_full = torch.cat([prev_k, k_context_new], dim=1) if N_new > 0 else prev_k
            v_context_full = torch.cat([prev_v, v_context_new], dim=1) if N_new > 0 else prev_v
        else:
            k_context_full, v_context_full = k_context_new, v_context_new
        
        # The new cache only contains context
        new_kv_cache = (k_context_full, v_context_full)

        # 5. Attention: Query(Context+Hints) attends to Key(PastContext + NewContext + Hints)
        k_all = torch.cat([k_context_full, k_hints], dim=1)
        v_all = torch.cat([v_context_full, v_hints], dim=1)

        q_all_attn = rearrange(q_all, 'b s h d -> b h s d')
        k_all_attn = rearrange(k_all, 'b h s d -> b h s d') # wait, k_all is [B, S, H, D]
        k_all_attn = rearrange(k_all, 'b s h d -> b h s d')
        v_all_attn = rearrange(v_all, 'b s h d -> b h s d')

        attn_output = F.scaled_dot_product_attention(
            q_all_attn, k_all_attn, v_all_attn, 
            is_causal=False, attn_mask=attn_mask
        )
        attn_output = rearrange(attn_output, 'b h s d -> b s (h d)')

        # 6. Fused MLP and Residual
        x = x_skip + self.dropout1(self.attn_out(attn_output))
        x = x + self.dropout2(self.mlp(self.norm2(x)))

        return x, new_kv_cache
    
    def custom_forward_double(self, 
                              hidden_states_future_1: torch.Tensor, rotary_cos_sin_f1: tuple, 
                              hidden_states_future_2: torch.Tensor, rotary_cos_sin_f2: tuple, 
                              kv_cache: tuple, attn_mask: torch.Tensor):
        """
        Args:
            hidden_states_future_1: (B, N*k, H) - The real tokens from the first prediction.
            rotary_cos_sin_f1: (cos, sin) for positions [t+1 ... t+k]
            hidden_states_future_2: (B, N*k, H) - The noise hints for the second prediction.
            rotary_cos_sin_f2: (cos, sin) for positions [t+k+1 ... t+2k]
            kv_cache: (k_ctx, v_ctx) where k_ctx is (B, h, N, d)
            attn_mask: The global staircase mask (2Nk, N + 2Nk)
        """
        B, Nk, H = hidden_states_future_1.shape
        N = Nk // (self.n_heads if hasattr(self, 'n_heads') else 1) # This N is actually context len
        # Note: N is derived from kv_cache inside the calling function usually.
        
        k_ctx, v_ctx = kv_cache # Shapes: (B, h, N, d)

        # 1. Normalization
        f1_norm = self.norm1(hidden_states_future_1)
        f2_norm = self.norm1(hidden_states_future_2)

        # 2. QKV Projections
        with torch.cuda.amp.autocast(enabled=False):
            qkv_f1 = self.attn_qkv(f1_norm.float())
            qkv_f2 = self.attn_qkv(f2_norm.float())
            
            qkv_f1 = rearrange(qkv_f1, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
            qkv_f2 = rearrange(qkv_f2, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

            # Apply RoPE for each stream using their specific positions
            cos1, sin1 = rotary_cos_sin_f1
            qkv_f1 = apply_rotary_pos_emb(qkv_f1, cos1.to(qkv_f1.dtype), sin1.to(qkv_f1.dtype))
            
            cos2, sin2 = rotary_cos_sin_f2
            qkv_f2 = apply_rotary_pos_emb(qkv_f2, cos2.to(qkv_f2.dtype), sin2.to(qkv_f2.dtype))

        if hidden_states_future_1.is_cuda and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            qkv_f1, qkv_f2 = qkv_f1.to(target_dtype), qkv_f2.to(target_dtype)

        q1, k1, v1 = qkv_f1.unbind(dim=2)
        q2, k2, v2 = qkv_f2.unbind(dim=2)

        # 3. Concatenate for Global Attention
        # Queries: [Future_1 | Future_2] (B, h, 2*Nk, d)
        q_all = torch.cat([rearrange(q1, 'b s h d -> b h s d'), rearrange(q2, 'b s h d -> b h s d')], dim=2)
        
        # Keys/Values: [Context | Future_1 | Future_2] (B, h, N + 2*Nk, d)
        k_all = torch.cat([k_ctx, rearrange(k1, 'b s h d -> b h s d'), rearrange(k2, 'b s h d -> b h s d')], dim=2)
        v_all = torch.cat([v_ctx, rearrange(v1, 'b s h d -> b h s d'), rearrange(v2, 'b s h d -> b h s d')], dim=2)

        # 4. Attention
        attn_out = F.scaled_dot_product_attention(q_all, k_all, v_all, attn_mask=attn_mask)
        attn_out = rearrange(attn_out, 'b h s d -> b s (h d)')
        
        res_f1 = attn_out[:, :Nk, :]
        res_f2 = attn_out[:, Nk:, :]

        # 5. Residual + MLP
        out_f1 = hidden_states_future_1 + self.dropout1(self.attn_out(res_f1))
        out_f1 = out_f1 + self.dropout2(self.mlp(self.norm2(out_f1)))
        
        out_f2 = hidden_states_future_2 + self.dropout1(self.attn_out(res_f2))
        out_f2 = out_f2 + self.dropout2(self.mlp(self.norm2(out_f2)))

        return out_f1, out_f2
    
class PointwiseNoiseEncoder(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 1. Combined Projection (Reduces 3 layers into 1)
        # Takes concatenated noise & tau features (hidden_size * 2) -> hidden_size
        self.combined_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 2. Pointwise MLP (2 layers. Total layers in module = 3)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    @amp.autocast("cuda", enabled=False)
    def _high_precision_sinusoidal_noise(self, x, dim, theta=10000):
        """
        Calculates expansion for continuous [0, 1] noise in double precision.
        """
        assert dim % 2 == 0
        half = dim // 2
        x = x.to(torch.float64) 
        freqs = torch.pow(theta, -torch.arange(half, device=x.device).div(half))
        args = x * freqs * 2.0 * math.pi
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return embedding

    def forward(self, noise_sequence, tau_sequence):
        """
        noise_sequence: (B, K, 1)
        tau_sequence:   (B, K, 1) or (B, 1, 1)
        """
        dtype = self.combined_proj.weight.dtype 

        # --- A. High Precision Noise Expansion (Continuous Z and Tau) ---
        noise_features = self._high_precision_sinusoidal_noise(noise_sequence, self.hidden_size)
        noise_features = noise_features.to(dtype)
        
        tau_features = self._high_precision_sinusoidal_noise(tau_sequence, self.hidden_size)
        tau_features = tau_features.to(dtype)
        
        # --- B. Combine and Project ---
        # 1. Ensure tau_features matches K dimension of noise_features for concatenation
        if tau_features.size(1) == 1 and noise_features.size(1) > 1:
            tau_features = tau_features.expand(-1, noise_features.size(1), -1)
            
        # 2. Concatenate raw features -> shape: (B, K, hidden_size * 2)
        x_concat = torch.cat([noise_features, tau_features], dim=-1)
        
        # 3. Single projection down -> shape: (B, K, hidden_size)
        x = self.combined_proj(x_concat)
        
        # --- C. Pointwise Processing ---
        x = x + self.mlp(self.norm(x))
        
        return x


class DDitFinalLayer(nn.Module):
  def __init__(
    self, hidden_size, out_channels
  ):
    super().__init__()
    self.norm_final = LayerNorm(hidden_size)
    self.linear = nn.Linear(hidden_size, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()

  def forward(self, x):
    return self.linear(self.norm_final(x))

class MTP(DDIT):
  def __init__(self, config, vocab_size, mask_index, max_k):
    # --- INFER CONFIG AND INITIALIZE FROM BASE CLASS ---
    super().__init__(config, vocab_size)

    self.max_k = max_k
    self.mask_index = mask_index
    self.neg_infinity = -1000.0
    hidden_size = config.model.hidden_size

    self.gradient_checkpointing = config.model.get('gradient_checkpointing', False)

    # --- REPLACE TRANSFORMER BLOCKS WITH FLEXIBLE VERSION ---
    del self.blocks
    blocks = []
    for _ in range(config.model.n_blocks):
        blocks.append(
            FlexDDiTBlock(
                dim=hidden_size,
                n_heads=config.model.n_heads,
                cond_dim=config.model.cond_dim,
                dropout=config.model.dropout,
            )
        )
    self.blocks = nn.ModuleList(blocks)


    del self.output_layer
    self.noise_encoder = PointwiseNoiseEncoder(hidden_size)
    # Causal mode predicts 1 token per hidden state (it has k states)
    self.output_layer = DDitFinalLayer(hidden_size, vocab_size)

  def set_teacher_model(self, teacher_model, teacher_type):
    self.teacher_type = teacher_type

    if teacher_type == "ar":
        # AR is treated as a special MTP with k=1
        teacher_k = 1
        # assert self.max_k == 1, "If teacher is AR, student max_k must be 1."
    else:
        teacher_k = teacher_model.max_k
        assert self.max_k == teacher_k * 2

    teacher_state_dict = teacher_model.state_dict()
    mtp_state_dict = self.state_dict()

    # 1. Bucket 1: Fully Transferred (Exact Name and Shape Match)
    fully_transferred_keys = [
        k for k, v in teacher_state_dict.items() 
        if k in mtp_state_dict and v.shape == mtp_state_dict[k].shape
    ]
    
    filtered_teacher_dict = {k: teacher_state_dict[k] for k in fully_transferred_keys}
    self.load_state_dict(filtered_teacher_dict, strict=False)

    # 2. Bucket 2: Partially Transferred (Stitched for MTP expansion)
    partially_transferred_info = [] # Stores (key, method) for reporting

    # 3. Bucket 3: Newly Initialized (No teacher data used)
    partial_keys_set = {f"{p[0]}.weight" for p in partially_transferred_info} | \
                      {f"{p[0]}.bias" for p in partially_transferred_info}
    
    transferred_set = set(fully_transferred_keys) | partial_keys_set
    new_keys = [k for k in mtp_state_dict.keys() if k not in transferred_set and 'rotary_emb' not in k]

    # --- DETAILED REPORTING ---
    print(f"\n{'='*80}")
    print(f"MTP DISTILLATION INITIALIZATION SUMMARY")
    print(f"Teacher Type: {teacher_type.upper()} | Teacher k: {teacher_k} | Student k: {self.max_k}")
    print(f"{'='*80}")

    print(f"\n✅ FULLY TRANSFERRED ({len(fully_transferred_keys)} tensors):")
    print(f"   Includes: Embeddings, Transformer Blocks (Attention/MLP), and Normalization layers.")

    if partially_transferred_info:
        print(f"\n📈 PARTIALLY TRANSFERRED / STITCHED ({len(partially_transferred_info)} layers):")
        for layer, desc in partially_transferred_info:
            print(f"   • {layer:<25} | {desc}")
    else:
        print(f"\n📈 PARTIALLY TRANSFERRED: None (Direct AR-to-AR mapping)")

    print(f"\n✨ NEWLY INITIALIZED ({len(new_keys)} tensors):")
    for k in sorted(new_keys):
        # We explicitly show shapes for new weights to help debugging
        print(f"   • {k:<40} | Shape: {list(mtp_state_dict[k].shape)}")

    print(f"\n{'='*80}\n")


  def forward(self, context_tokens: torch.Tensor, noise_vectors: torch.Tensor, tau_vectors: torch.Tensor, mode: str = 'train', **kwargs) -> torch.Tensor:
    """
    Unified forward pass dispatcher for the MTP model.

    This method routes the input to the appropriate forward pass implementation
    (training or inference) based on the dimensionality of the `noise_tau_vectors`.

    Args:
        context_tokens (torch.Tensor): Input context token IDs. Shape: (B, N_ctx).
        noise_tau_vectors (torch.Tensor): The conditioning vector containing noise and tau.
            - For training: Shape (B, N_ctx, max_k + 1). A different hint is provided
              for each token position in the context.
            - For inference: Shape (B, max_k + 1). A single hint is used for the
              entire context to predict the next `k` tokens.

    Returns:
        torch.Tensor: The output log-probabilities.
            - Shape for training: (B, N_ctx, max_k, vocab_size)
            - Shape for inference: (B, N_ctx, max_k, vocab_size)
    """
    if mode == 'train':
        return self._forward_training(context_tokens, noise_vectors, tau_vectors, **kwargs)
    elif mode == 'inference':
        # Inference case: noise_tau_vectors is (B, max_k + 1)
        return self._forward_inference(context_tokens, noise_vectors, tau_vectors, **kwargs)
    else:
        raise NotImplementedError(f"Mode {mode} not implemented.")

  def _forward_inference(self, context_tokens: torch.Tensor, noise_vectors: torch.Tensor, tau_vectors: torch.Tensor, **kwargs) -> torch.Tensor:
    B, N_seq = context_tokens.shape
    device = context_tokens.device
    kv_caches = kwargs.get('kv_caches', None)
    
    # 1. Past length from cache
    N_past = kv_caches[0][0].shape[1] if (kv_caches is not None and kv_caches[0] is not None) else 0
    N_new = N_seq - N_past
    
    # 2. Get embeddings for ONLY the new context tokens
    new_context_tokens = context_tokens[:, N_past:]
    hidden_states_context = self.vocab_embed(new_context_tokens)
    
    # 3. Encode Noise hints
    # noise_vectors: (B, k, 1), tau_vectors: (B, 1, 1)
    hidden_states_hints = self.noise_encoder(noise_vectors, tau_vectors) # (B, k, H)
    k_fut = hidden_states_hints.shape[1]

    # 4. Concatenate: [New Context | Hints]
    x = torch.cat([hidden_states_context, hidden_states_hints], dim=1)
    
    # 5. Prepare RoPE for [New Context | Hints] with offset N_past
    # total_new_len = N_new + k_fut
    rotary_cos_sin = self.rotary_emb.forward_with_offset(x, offset=N_past)

    # 6. Prepare Causal Mask
    # Query length: N_new + k_fut
    # Key length: N_past + N_new + k_fut
    total_q = N_new + k_fut
    total_k = N_past + N_new + k_fut
    
    q_idx = torch.arange(N_past, N_past + total_q, device=device).unsqueeze(1)
    k_idx = torch.arange(total_k, device=device).unsqueeze(0)
    
    # Standard causal mask: Query at i sees Key at j if j <= i
    attn_mask = (q_idx >= k_idx).unsqueeze(0).unsqueeze(0) # (1, 1, total_q, total_k)

    # 7. Loop through blocks
    if kv_caches is None:
        kv_caches = [None] * len(self.blocks)
    
    new_kv_caches = []
    for i, block in enumerate(self.blocks):
        x, next_cache = block.custom_forward_inference(
            x, 
            rotary_cos_sin, 
            N_new, 
            kv_cache=kv_caches[i],
            attn_mask=attn_mask
        )
        new_kv_caches.append(next_cache)

    # 8. Extract logits for the hints only
    # x: [B, N_new + k_fut, H] -> last k_fut states
    hidden_states_future = x[:, N_new:]
    logits = self.output_layer(hidden_states_future) 

    if self.mask_index is not None:
        logits[:, :, self.mask_index] = self.neg_infinity

    return (logits, new_kv_caches) if kwargs.get('return_kv', False) else logits

  def _forward_training(self, context_tokens: torch.Tensor, noise_vectors: torch.Tensor, tau_vectors: torch.Tensor, **kwargs) -> torch.Tensor:
    B, N = context_tokens.shape
    k = noise_vectors.shape[2]
    device = context_tokens.device
    hidden_size = self.config.model.hidden_size
    return_kv = kwargs.get('return_kv', False) # Check for flag

    # 1. Encode Noise Hints
    noise_batched = noise_vectors.reshape(B * N, k, 1)
    tau_batched = tau_vectors.reshape(B * N, 1, 1)
    hidden_states_future = self.noise_encoder(noise_batched, tau_batched)
    hidden_states_future = hidden_states_future.view(B, N * k, hidden_size)

    # 2. Prepare Context Embeddings
    hidden_states_context = self.vocab_embed(context_tokens)

    # 3. Construct the Global Staircase Attention Mask
    total_len = N + (N * k)
    attn_mask = torch.zeros(total_len, total_len, device=device, dtype=torch.bool)
    attn_mask[:N, :N] = torch.tril(torch.ones(N, N, device=device, dtype=torch.bool))
    
    r_fut_idx = torch.arange(N * k, device=device)
    c_ctx_idx = torch.arange(N, device=device)
    c_fut_idx = torch.arange(N * k, device=device)

    attn_mask[N:, :N] = (r_fut_idx // k)[:, None] >= c_ctx_idx[None, :]
    same_window = (r_fut_idx // k)[:, None] == (c_fut_idx // k)[None, :]
    causal_in_window = (r_fut_idx % k)[:, None] >= (c_fut_idx % k)[None, :]
    attn_mask[N:, N:] = same_window & causal_in_window
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

    # 4. PREPARE ROPE
    cos_ctx, sin_ctx = self.rotary_emb(hidden_states_context)
    rotary_ctx = (cos_ctx, sin_ctx)

    t_idx = torch.arange(N, device=device).view(N, 1)
    i_idx = torch.arange(1, k + 1, device=device).view(1, k)
    fut_positions = (t_idx + i_idx).view(-1)
    
    max_pos = N + k + 1
    dummy = torch.empty(1, max_pos, 1, device=device)
    cos_all, sin_all = self.rotary_emb(dummy)
    
    cos_fut = cos_all[:, fut_positions, :, :, :]
    sin_fut = sin_all[:, fut_positions, :, :, :]
    rotary_fut = (cos_fut, sin_fut)

    # 5. Transformer Blocks Loop
    kv_caches = []
    for block in self.blocks:
        if self.gradient_checkpointing and not return_kv: # Usually don't checkpoint when returning KV
            hidden_states_context, hidden_states_future = checkpoint(
                block.custom_forward_train,
                hidden_states_context, rotary_ctx,
                hidden_states_future, rotary_fut,
                attn_mask,
                False, # return_kv
                use_reentrant=False
            )
        else:
            # Call with return_kv flag
            outputs = block.custom_forward_train(
                hidden_states_context, rotary_ctx,
                hidden_states_future, rotary_fut,
                attn_mask,
                return_kv=return_kv
            )
            
            if return_kv:
                hidden_states_context, hidden_states_future, cache = outputs
                kv_caches.append(cache)
            else:
                hidden_states_context, hidden_states_future = outputs
    
    # 6. Final Output
    logits = self.output_layer(hidden_states_future)
    all_logits = rearrange(logits, 'b (n k) v -> b n k v', n=N, k=k)

    if self.mask_index is not None:
        all_logits[:, :, :, self.mask_index] = self.neg_infinity

    # Return KV caches if requested
    if return_kv:
        return all_logits, kv_caches
    
    return all_logits 
  
  @torch.no_grad()
  def generate_ar_teacher_sequence(self, ar_teacher_model, context_tokens, noise, tau):
    """
    Teacher generation using Robust Direct Min-P Sampling.
    Uses .exp() on log_softmax outputs. No temperature scaling.
    """
    # 1. Get log_probs from the teacher (already log_softmax normalized)
    logits = ar_teacher_model.forward_high_precision(context_tokens, temperature=tau)
    B, N, V = logits.shape
    k = self.max_k

    logits = logits.double() # use double precision

    probs = torch.softmax(logits, dim=-1)
    probs_cumsum = torch.cumsum(probs, dim=-1) # (B, N, V)

    probs_cumsum_expanded = probs_cumsum.unsqueeze(2).expand(B, N, k, V)
    flat_cumsum = probs_cumsum_expanded.reshape(-1, V)
    flat_z = noise.double().reshape(-1, 1) # noise is z
    flat_z = flat_z
    
    teacher_sequences = torch.searchsorted(flat_cumsum, flat_z, right=False)
    teacher_sequences = torch.clamp(teacher_sequences, 0, V - 1)

    # 8. Return back to (B, N, k)
    return teacher_sequences.view(B, N, k)

  @torch.no_grad()
  def double_forward(self, context_kv_caches, future_tokens, noise_vectors, tau_vectors):
    """
    Args:
        context_kv_caches: List of (k, v) from the context pass.
        future_tokens: (B, N, k) - Tokens predicted in step 1.
        noise_vectors: (B, N, k, 1) - Noise for step 2.
        tau_vectors: (B, N, 1)
    """
    B, N, k = future_tokens.shape
    device = future_tokens.device
    hidden_size = self.config.model.hidden_size

    # 1. Embeddings and Noise Encoding
    # Future 1: Real tokens from first k steps
    h_f1 = self.vocab_embed(future_tokens).view(B, N * k, hidden_size)
    
    # Future 2: Noise hints for next k steps
    noise_batched = noise_vectors.view(B * N, k, 1)
    tau_batched = tau_vectors.view(B * N, 1, 1)
    h_f2 = self.noise_encoder(noise_batched, tau_batched).view(B, N * k, hidden_size)

    # 2. Prepare RoPE for both streams
    # F1 positions: t + 1 ... t + k
    # F2 positions: t + k + 1 ... t + 2k
    t_idx = torch.arange(N, device=device).view(N, 1)
    i_idx = torch.arange(1, k + 1, device=device).view(1, k)
    
    pos_f1 = (t_idx + i_idx).view(-1)
    pos_f2 = (t_idx + i_idx + k).view(-1)
    
    max_pos = N + (2 * k) + 1
    dummy = torch.empty(1, max_pos, 1, device=device)
    cos_all, sin_all = self.rotary_emb(dummy)
    
    rotary_f1 = (cos_all[:, pos_f1, :, :, :], sin_all[:, pos_f1, :, :, :])
    rotary_f2 = (cos_all[:, pos_f2, :, :, :], sin_all[:, pos_f2, :, :, :])

    # 3. Construct Double Staircase Mask
    # Query length: 2*Nk (F1 then F2)
    # Key length: N + 2*Nk (Context then F1 then F2)
    total_q = 2 * N * k
    total_k = N + 2 * N * k
    mask = torch.zeros(total_q, total_k, device=device, dtype=torch.bool)
    
    # Indices
    r_f1 = torch.arange(N * k, device=device)
    r_f2 = torch.arange(N * k, device=device) + (N * k)
    c_ctx = torch.arange(N, device=device)
    c_f1 = torch.arange(N * k, device=device) + N
    c_f2 = torch.arange(N * k, device=device) + (N + N * k)

    # --- Future 1 Constraints ---
    # F1 -> Context: Staircase
    mask[:N*k, :N] = (r_f1 // k)[:, None] >= c_ctx[None, :]
    # F1 -> F1: Causal in window
    same_win_f1 = (r_f1 // k)[:, None] == ((c_f1 - N) // k)[None, :]
    causal_f1 = (r_f1 % k)[:, None] >= ((c_f1 - N) % k)[None, :]
    mask[:N*k, N:N+N*k] = same_win_f1 & causal_f1

    # --- Future 2 Constraints ---
    # F2 -> Context: Staircase
    mask[N*k:, :N] = (r_f2 // k - N)[:, None] >= c_ctx[None, :]
    # F2 -> F1: Windowed (Full window visibility)
    mask[N*k:, N:N+N*k] = (r_f2 // k - N)[:, None] == ((c_f1 - N) // k)[None, :]
    # F2 -> F2: Causal in window
    same_win_f2 = (r_f2 // k - N)[:, None] == ((c_f2 - (N+N*k)) // k)[None, :]
    causal_f2 = (r_f2 % k)[:, None] >= ((c_f2 - (N+N*k)) % k)[None, :]
    mask[N*k:, N+N*k:] = same_win_f2 & causal_f2

    attn_mask = mask.unsqueeze(0).unsqueeze(0)

    # 4. Forward through blocks
    for block, kv_cache in zip(self.blocks, context_kv_caches):
        h_f1, h_f2 = block.custom_forward_double(
            h_f1, rotary_f1,
            h_f2, rotary_f2,
            kv_cache, attn_mask
        )

    # 5. Final Output (Logits for Future 2)
    logits = self.output_layer(h_f2)
    all_logits = rearrange(logits, 'b (n k) v -> b n k v', n=N, k=k)

    if self.mask_index is not None:
        all_logits[:, :, :, self.mask_index] = self.neg_infinity

    return all_logits

  @torch.no_grad()
  def generate_mtp_teacher_sequence(self, mtp_teacher_model, context_tokens, noise_vectors, tau_vectors):
    # The most complicated implementation here. 
    # context_tokens: (B, N_ctx)
    # noises: (B, N_ctx, max_k, 1)
    # tau: (B, N_ctx, 1)
    # mtp_teacher_model: the mtp models that support k=max_k / 2 sampling
    # In this code, it need to call mtp_teacher_model twice to sample the next two k tokens for each position in the context

    # noise splitting, split the noise into two parts with shape (B, N_ctx, max_k/2, 1)
    noise_vectors_1, noise_vectors_2 = noise_vectors.chunk(2, dim=-2)

    logits_1, context_kv_caches = mtp_teacher_model(context_tokens, noise_vectors_1, tau_vectors, mode='train', return_kv=True)

    future_tokens_1 = logits_1.float().argmax(dim=-1) # (B, N_ctx, max_k/2), the first half inferenced tokens
    del logits_1

    logits_2 = mtp_teacher_model.double_forward(context_kv_caches, future_tokens_1, noise_vectors_2, tau_vectors)

    future_tokens_2 = logits_2.float().argmax(dim=-1)
    del logits_2

    # merge the two halves
    teacher_sequences = torch.cat([future_tokens_1, future_tokens_2], dim=-1) # (B, N_ctx, max_k)

    return teacher_sequences 

  def compute_next_k_tokens_prediction_loss(self, teacher_model, batch_sequences: torch.Tensor, tau: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
    """
    Computes the training loss for predicting the next `k` tokens.

    This method orchestrates the full training step:
    1.  Performs noise inversion on the target sequences using a teacher model.
    2.  Creates sliding windows of contexts, targets, and noise vectors.
    3.  Passes the contexts and hints to the MTP model's forward pass.
    4.  Calculates the Negative Log-Likelihood (NLL) loss for each of the `k` predictions.

    Args:
        teacher_model: The teacher model for noise inversion.
        batch_sequences (torch.Tensor): Input sequences. Shape (B, N).
        tau (torch.Tensor): Temperature for each sample in the batch. Shape (B,).

    Returns:
        dict[str, torch.Tensor]: A dictionary of losses, including a total loss and
                                 individual losses for each predicted token step (loss_1, loss_2, ...).
    """
    B, N = batch_sequences.shape
    k = self.max_k
    if N <= k:
        return {}

    # === STEP 1: Data & Noise Preparation ===
    num_pred_steps = N - k
    # Sample 50D Gaussian noise and apply the diversity scale
    noise_vectors = torch.rand(B, num_pred_steps, k, 1, device=tau.device)

    context_tokens = batch_sequences[:, :num_pred_steps]
    
    tau_vectors = tau.view(B, 1, 1).expand(B, num_pred_steps, 1)

    # === STEP 2: Teacher Generation (Noise Inversion) ===
    if self.teacher_type == "ar":
        teacher_sequences = self.generate_ar_teacher_sequence(teacher_model, context_tokens, noise_vectors, tau)
    else:
        teacher_sequences = self.generate_mtp_teacher_sequence(teacher_model, context_tokens, noise_vectors, tau_vectors)

    targets = teacher_sequences

    # === STEP 4: Student Model Forward Pass ===
    logits = self.forward(context_tokens, noise_vectors, tau_vectors, mode='train')

    # === STEP 5: Loss Calculation ===
    loss_dict = {}
    all_correct_mask = None
    for i in range(k):
        # 1. Isolate the i-th prediction and flatten batch/sequence
        # Shape: (B * num_pred_steps, V)
        logits_i = logits[:, :, i, :].reshape(-1, self.vocab_size)
        targets_i = targets[:, :, i].reshape(-1)
        
        # 2. UPCAST HERE: Convert logits to float32 before the loss
        # F.cross_entropy is memory-efficient because it doesn't store the 
        # intermediate softmax results for the backward pass.
        loss = F.cross_entropy(
            logits_i.to(torch.float32), 
            targets_i,
            ignore_index=self.mask_index, # Standard way to handle mask tokens
            reduction='none'
        )
        
        # 3. Reshape back to (B, N) so the metric logic in diffusion.py stays the same
        loss_dict[f"loss_{i+1}"] = loss.view(B, num_pred_steps)

        with torch.no_grad():
            preds_i = logits_i.argmax(dim=-1)
            # 1 if correct, 0 if wrong
            correct_i = (preds_i == targets_i).float() 
            # Set mask tokens to 1 so they don't count as "wrong" in the 'all' check
            # (Though they are filtered out by weight later anyway)
            is_ignored = (targets_i == self.mask_index)
            correct_i_for_all = correct_i.clone()
            correct_i_for_all[is_ignored] = 1.0

            loss_dict[f"acc_{i+1}"] = correct_i.view(B, num_pred_steps)
            
            # Track if the whole sequence (1 to k) is correct
            if all_correct_mask is None:
                all_correct_mask = correct_i_for_all.view(B, num_pred_steps)
            else:
                all_correct_mask = all_correct_mask * correct_i_for_all.view(B, num_pred_steps)

    # Return the "Full Sequence Match" as acc_0
    loss_dict["acc_total"] = all_correct_mask
    
    return loss_dict
  
  @torch.no_grad()
  def noise_inversion(self, ar_teacher_model: AR, batch_sequences: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Calculates the noise `u` that would generate a given sequence using a teacher model.

    This function performs the inverse of the sampling process. For each token in
    the sequence, it computes the teacher model's predicted probability distribution,
    finds the CDF, and then determines what noise `u` from U(0, 1) would have
    resulted in that token being sampled.

    `u = CDF(token-1) + (CDF(token) - CDF(token-1)) * rand`

    Args:
        ar_teacher_model (AR): The autoregressive teacher model for providing probabilities.
        batch_sequences (torch.Tensor): The ground-truth sequences. Shape (B, N).
        tau (torch.Tensor): The temperature used for generation. Shape (B,).

    Returns:
        torch.Tensor: The inverted noise `u` for each token (except the first).
                      Shape (B, N-1).
    """
    ar_inputs = batch_sequences[:, :-1]
    ar_targets = batch_sequences[:, 1:]
    log_probs = ar_teacher_model(ar_inputs, temperature=tau)
    probs = log_probs.exp().to(dtype=torch.float64)

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)

    cdf_truncated = sorted_probs.cumsum(dim=-1)

    rank = sorted_indices.argsort(dim=-1)
    target_rank = torch.gather(rank, -1, ar_targets.unsqueeze(-1)).squeeze(-1)

    current_cdf = torch.gather(cdf_truncated, -1, target_rank.unsqueeze(-1)).squeeze(-1)

    prev_rank = (target_rank - 1).clamp(min=0)
    prev_cdf = torch.gather(cdf_truncated, -1, prev_rank.unsqueeze(-1)).squeeze(-1)
    prev_cdf[target_rank == 0] = 0.0

    u = prev_cdf + (current_cdf - prev_cdf) * torch.rand_like(prev_cdf)
    u = u.clamp(min=0.0, max=1.0).to(dtype=log_probs.dtype) # Clamp to ensure u is in the valid [0, 1] range.
    
    return u
  
  def compute_noise_inversion_next_k_tokens_prediction_loss(self, ar_teacher_model: AR, batch_sequences: torch.Tensor, tau: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
    """
    Computes the training loss for predicting the next `k` tokens.

    This method orchestrates the full training step:
    1.  Performs noise inversion on the target sequences using a teacher model.
    2.  Creates sliding windows of contexts, targets, and noise vectors.
    3.  Passes the contexts and hints to the MTP model's forward pass.
    4.  Calculates the Negative Log-Likelihood (NLL) loss for each of the `k` predictions.

    Args:
        ar_teacher_model (AR): The autoregressive teacher model for noise inversion.
        batch_sequences (torch.Tensor): Input sequences. Shape (B, N).
        tau (torch.Tensor): Temperature for each sample in the batch. Shape (B,).

    Returns:
        dict[str, torch.Tensor]: A dictionary of losses, including a total loss and
                                 individual losses for each predicted token step (loss_1, loss_2, ...).
    """
    B, N = batch_sequences.shape
    k = self.max_k
    if N <= k:
        return {}
    all_noises = self.noise_inversion(ar_teacher_model, batch_sequences, tau)
    num_pred_steps = N - k
    context_tokens = batch_sequences[:, :num_pred_steps]
    targets = batch_sequences[:, 1:].as_strided(
        size=(B, num_pred_steps, k),
        stride=(batch_sequences.stride(0), batch_sequences.stride(1), batch_sequences.stride(1)),
    ).contiguous()
    noise_vectors = all_noises.as_strided(
        size=(B, num_pred_steps, k),
        stride=(all_noises.stride(0), all_noises.stride(1), all_noises.stride(1)),
    ).contiguous()
    tau_vectors = tau.view(B, 1, 1).expand(B, num_pred_steps, 1)
    logits = self.forward(context_tokens, noise_vectors, tau_vectors, mode='train')
    
    loss_dict = {}
    all_correct_mask = None
    for i in range(k):
        logits_i = logits[:, :, i, :].reshape(-1, self.vocab_size)
        targets_i = targets[:, :, i].reshape(-1)

        loss = F.cross_entropy(
            logits_i.to(torch.float32),
            targets_i,
            ignore_index=self.mask_index,
            reduction='none'
        )
        loss_dict[f"loss_{i+1}"] = loss.view(B, num_pred_steps)

        with torch.no_grad():
            preds_i = logits_i.argmax(dim=-1)
            # 1 if correct, 0 if wrong
            correct_i = (preds_i == targets_i).float() 
            # Set mask tokens to 1 so they don't count as "wrong" in the 'all' check
            # (Though they are filtered out by weight later anyway)
            is_ignored = (targets_i == self.mask_index)
            correct_i_for_all = correct_i.clone()
            correct_i_for_all[is_ignored] = 1.0

            loss_dict[f"acc_{i+1}"] = correct_i.view(B, num_pred_steps)
            
            # Track if the whole sequence (1 to k) is correct
            if all_correct_mask is None:
                all_correct_mask = correct_i_for_all.view(B, num_pred_steps)
            else:
                all_correct_mask = all_correct_mask * correct_i_for_all.view(B, num_pred_steps)

    # Return the "Full Sequence Match" as acc_0
    loss_dict["acc_total"] = all_correct_mask
    
    return loss_dict

  @torch.no_grad()
  def sample_next_k_tokens(self,
                           context_tokens: torch.Tensor,
                           tau: torch.Tensor,
                           num_tokens_to_generate: int,
                           k: typing.Optional[int] = None):
    """
    Autoregressively generates text by predicting `k` tokens at a time.
    """
    self.eval()
    if k is None: k = self.max_k
    if not isinstance(k, int) or not (1 <= k <= self.max_k):
        raise ValueError(f"k must be an integer between 1 and {self.max_k}, but got {k}.")
    B, N_ctx = context_tokens.shape
    device = context_tokens.device
    generated_sequence = torch.cat([
        context_tokens,
        torch.full((B, num_tokens_to_generate), self.mask_index, dtype=torch.long, device=device)
    ], dim=1)
    current_length = N_ctx
    for i in tqdm(range(0, num_tokens_to_generate, k), desc="Generating text"):
        tokens_in_this_step = min(k, num_tokens_to_generate - i)
        if tokens_in_this_step == 0: break
        current_context = generated_sequence[:, :current_length]
        noise_vectors = torch.rand(B, self.max_k, 1, device=tau.device)
        tau_vectors = tau.view(B, 1, 1)
        logits = self.forward(current_context, noise_vectors, tau_vectors, mode='inference')
        last_step_log_probs = logits[:, :, :]
        log_probs_to_sample = last_step_log_probs[:, :tokens_in_this_step, :]
        predicted_tokens = log_probs_to_sample.float().argmax(dim=-1)
        del log_probs_to_sample, logits, last_step_log_probs
        start_idx = current_length
        end_idx = current_length + tokens_in_this_step
        generated_sequence[:, start_idx:end_idx] = predicted_tokens
        current_length = end_idx
    return generated_sequence

  @torch.no_grad()
  def sample_next_k_tokens_with_kv_caches(self,
                           context_tokens: torch.Tensor,
                           tau: torch.Tensor,
                           num_tokens_to_generate: int,
                           k: typing.Optional[int] = None):
    """
    Autoregressively generates text by predicting `k` tokens at a time.
    """
    self.eval()
    if k is None: k = self.max_k
    if not isinstance(k, int) or not (1 <= k <= self.max_k):
        raise ValueError(f"k must be an integer between 1 and {self.max_k}, but got {k}.")
    B, N_ctx = context_tokens.shape
    device = context_tokens.device
    generated_sequence = torch.cat([
        context_tokens,
        torch.full((B, num_tokens_to_generate), self.mask_index, dtype=torch.long, device=device)
    ], dim=1)
    current_length = N_ctx

    kv_caches = None

    for i in tqdm(range(0, num_tokens_to_generate, k)):
        tokens_in_this_step = min(k, num_tokens_to_generate - i)
        if tokens_in_this_step == 0: break
        current_context = generated_sequence[:, :current_length]
        noise_vectors = torch.rand(B, self.max_k, 1, device=tau.device)
        tau_vectors = tau.view(B, 1, 1)
        logits, kv_caches = self.forward(current_context, noise_vectors, tau_vectors, mode='inference', kv_caches=kv_caches, return_kv=True)
        last_step_log_probs = logits[:, :, :]
        log_probs_to_sample = last_step_log_probs[:, :tokens_in_this_step, :]
        predicted_tokens = log_probs_to_sample.float().argmax(dim=-1)
        del log_probs_to_sample, logits, last_step_log_probs
        start_idx = current_length
        end_idx = current_length + tokens_in_this_step
        generated_sequence[:, start_idx:end_idx] = predicted_tokens
        current_length = end_idx
    return generated_sequence
