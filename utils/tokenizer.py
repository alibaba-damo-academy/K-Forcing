"""Tokenizer setup, mask_index resolution, and BOS resolution."""

import transformers


def get_tokenizer(name: str) -> transformers.PreTrainedTokenizer:
    """Load a HuggingFace tokenizer by name or path."""
    return transformers.AutoTokenizer.from_pretrained(name)


def resolve_mask_index(
    tokenizer: transformers.PreTrainedTokenizer,
) -> tuple[int, int]:
    """Determine the mask token index and effective vocabulary size.

    If the tokenizer has no mask_token, the mask index is set to vocab_size
    (one past the last real token) and the effective vocab grows by 1.
    Otherwise the existing mask_token_id is used.

    Returns:
        (mask_index, effective_vocab_size)
    """
    base_vocab = tokenizer.vocab_size
    if (not hasattr(tokenizer, "mask_token")) or tokenizer.mask_token is None:
        mask_index = base_vocab
        effective_vocab = base_vocab + 1
    else:
        mask_index = tokenizer.mask_token_id
        effective_vocab = base_vocab
    return mask_index, effective_vocab


def get_bos_id(tokenizer: transformers.PreTrainedTokenizer) -> int:
    """Return the beginning-of-sequence token id.

    Falls back to cls_token_id (for BERT-style tokenizers) then 0.
    """
    if tokenizer.bos_token_id is not None:
        return tokenizer.bos_token_id
    if tokenizer.cls_token_id is not None:
        return tokenizer.cls_token_id
    return 0
