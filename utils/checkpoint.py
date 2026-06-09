"""Checkpoint loading with automatic config inference from state_dict.

Based on https://github.com/kuleshov-group/mdlm
"""

import re

import omegaconf
import torch
from loguru import logger

from models.autoregressive import AR
from models.pflm import MTP


def _infer_base_config(state_dict: dict) -> tuple[int, int, int, int]:
    """Infer (vocab_size, hidden_size, n_heads, n_blocks) from state_dict keys."""
    vocab_weight = state_dict["vocab_embed.embedding"]
    vocab_size, hidden_size = vocab_weight.shape

    block_indices = [
        int(re.search(r"blocks\.(\d+)\.", k).group(1))
        for k in state_dict.keys()
        if k.startswith("blocks.")
    ]
    n_blocks = max(block_indices) + 1

    inv_freq = state_dict["rotary_emb.inv_freq"]
    head_dim = inv_freq.shape[0] * 2
    n_heads = hidden_size // head_dim

    return vocab_size, hidden_size, n_heads, n_blocks


def load_ar_model(ckpt_path: str, device: str, mask_index: int) -> AR:
    """Load an AR checkpoint, auto-inferring config from the state_dict."""
    logger.info(f"Loading AR checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    vocab_size, hidden_size, n_heads, n_blocks = _infer_base_config(state_dict)

    config = omegaconf.OmegaConf.create(
        {
            "model": {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "n_heads": n_heads,
                "n_blocks": n_blocks,
                "cond_dim": 1024,
                "dropout": 0.0,
                "causal": True,
                "scale_by_sigma": False,
            }
        }
    )
    model = AR(config, vocab_size=vocab_size, mask_index=mask_index)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    logger.info(
        f"  vocab_size={vocab_size}, hidden={hidden_size}, "
        f"n_heads={n_heads}, n_blocks={n_blocks}"
    )
    return model


def load_pflm_model(ckpt_path: str, device: str, mask_index: int) -> MTP:
    """Load a PFLM (MTP) checkpoint, auto-inferring config from the state_dict."""
    logger.info(f"Loading PFLM checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    vocab_size, hidden_size, n_heads, n_blocks = _infer_base_config(state_dict)

    # Infer max_k from output layer weight shape
    output_weight = state_dict["output_layer.linear.weight"]
    max_k = output_weight.shape[0] // vocab_size
    if max_k < 1:
        max_k = 1

    logger.info(
        f"  Inferred: max_k={max_k}, hidden={hidden_size}, "
        f"blocks={n_blocks}, heads={n_heads}"
    )

    config = omegaconf.OmegaConf.create(
        {
            "model": {
                "hidden_size": hidden_size,
                "vocab_size": vocab_size,
                "n_heads": n_heads,
                "n_blocks": n_blocks,
                "cond_dim": 1024,
                "dropout": 0.0,
                "causal": True,
                "scale_by_sigma": False,
            }
        }
    )

    model = MTP(config, vocab_size=vocab_size, mask_index=mask_index, max_k=max_k)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model
