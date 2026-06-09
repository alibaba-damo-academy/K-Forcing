#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model pflm \
    --task lm1b \
    --ckpt_path /tmp/pflm_models/pflm_lm1b_k4.ckpt \
    --batch_size 4 \
    --num_samples 1 \
    --K 4 \
    --prefix_file assets/prefix_lm1b_examples.jsonl \
    --output_dir outputs/pflm_lm1b \
    --warmup_steps 1
