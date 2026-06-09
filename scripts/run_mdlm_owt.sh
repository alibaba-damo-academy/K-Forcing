#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model mdlm \
    --task owt \
    --batch_size 4 \
    --n_per_prefix 1 \
    --K 2 \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --output_dir outputs/mdlm_owt \
    --warmup_steps 1
