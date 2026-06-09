# K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling

## Description

K-Forcing is a push-forward language modeling paradigm for **joint next-k-token decoding**. It distills an existing autoregressive (AR) model into a conditional push-forward mapping that transforms independent uniform noise variables into a joint sample of multiple future tokens in a single forward pass. This design preserves fixed-length outputs, reuses the AR backbone architecture, and enables significant inference speedup under high-load batch serving — the scenario most critical for industrial-scale deployment.

<p align="center">
  <img src="assets/paradigm4.png" width="100%" />
</p>

**Comparison of four language-model inference paradigms within one forward evaluation.**
**(a) K-Forcing (ours)** uses a push-forward language model to map i.i.d. uniform noise tokens to a fixed-length block of future tokens, modeling their joint distribution.
**(b) AR** predicts one next token from the current context, leading to memory-bound decoding.
**(c) Speculative decoding** drafts a token block and verifies it with the target AR model, yielding a variable number of accepted tokens that breaks regular batching.
**(d) MDLM** predicts masked positions in parallel from per-position marginals, rather than their joint distribution.

## Setup

```bash
# Install dependencies (requires uv)
uv pip install -e .

# flash-attn must match your CUDA/torch version
uv pip install flash-attn --no-build-isolation
```

## Checkpoints

| Model | Dataset | HuggingFace | Local path |
|-------|---------|-------------|------------|
| AR    | OWT     | TBD         | `checkpoints/ar_owt.ckpt` |
| AR    | LM1B    | TBD         | `checkpoints/ar_lm1b.ckpt` |
| PFLM (k=4) | OWT | TBD     | `checkpoints/pflm_owt_k4.ckpt` |
| PFLM (k=4) | LM1B | TBD   | `checkpoints/pflm_lm1b_k4.ckpt` |
| MDLM  | OWT     | [kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt) | Auto-downloaded |

```bash
# Download AR/PFLM checkpoints (URLs TBD until public release)
mkdir -p checkpoints
# huggingface-cli download <repo> --local-dir checkpoints/
```

## Usage

```bash
# AR inference on OWT
python batch_inference_with_prefix.py \
    --model ar --task owt \
    --ckpt_path checkpoints/ar_owt.ckpt \
    --batch_size 4 --num_samples 16

# PFLM inference on OWT (k=4 tokens per forward pass)
python batch_inference_with_prefix.py \
    --model pflm --task owt \
    --ckpt_path checkpoints/pflm_owt_k4.ckpt \
    --batch_size 4 --num_samples 16 --num_tokens 4

# MDLM inference on OWT (downloads from HF automatically)
python batch_inference_with_prefix.py \
    --model mdlm --task owt \
    --batch_size 4 --num_samples 16 --num_tokens 4

# Show all options
python batch_inference_with_prefix.py --help
```

## TODO

- [ ] Arxiv paper release
- [ ] Checkpoint release on HuggingFace
- [ ] Training recipe (progressive self-forcing distillation)
