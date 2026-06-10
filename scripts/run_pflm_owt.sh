#!/bin/bash
# PFLM inference on OpenWebText (seq_len=1024, GPT-2 tokenizer).
# Decodes K tokens per forward pass via push-forward inverse-CDF mapping.
#
# --freq_penalty: penalizes repeated tokens (PFLM only). Recommended values for OWT:
#   K=2: 0.3 
#   K=3: 0.5 
#   K=4: 0.5
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model pflm \
    --task owt \
    --batch_size 4 \
    --n_per_prefix 1 \
    --K 2 \
    --freq_penalty 0.3 \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --output_dir outputs/pflm_owt \
    --warmup_steps 1
