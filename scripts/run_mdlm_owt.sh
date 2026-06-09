#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model mdlm \
    --task owt \
    --batch_size 16 \
    --num_samples 1 \
    --num_tokens 4 \
    --prefix_file assets/prefix_examples.jsonl \
    --output_dir outputs/mdlm_owt \
    --warmup_steps 1
