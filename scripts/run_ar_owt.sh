#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model ar \
    --task owt \
    --ckpt_path /tmp/pflm_models/ar_openwebtxt.ckpt \
    --batch_size 16 \
    --num_samples 1 \
    --prefix_file assets/prefix_examples.jsonl \
    --output_dir outputs/ar_owt \
    --warmup_steps 1
