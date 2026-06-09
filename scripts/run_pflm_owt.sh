#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model pflm \
    --task owt \
    --ckpt_path /tmp/pflm_models/pflm_owt_k4.ckpt \
    --batch_size 4 \
    --num_samples 1 \
    --num_tokens 4 \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --output_dir outputs/pflm_owt \
    --warmup_steps 1
