#!/bin/bash
# AR baseline inference on LM1B (seq_len=128, BERT tokenizer).
# Generates one completion per prefix using temperature-1 sampling with KV-cache.
# This is the teacher model baseline -- compare its outputs against PFLM.
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model ar \
    --task lm1b \
    --batch_size 4 \
    --n_per_prefix 1 \
    --prefix_file assets/prefix_lm1b_examples.jsonl \
    --output_dir outputs/ar_lm1b \
    --warmup_steps 1
