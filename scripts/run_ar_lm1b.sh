#!/bin/bash
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model ar \
    --task lm1b \
    --ckpt_path /tmp/pflm_models/ar_best_lm1b.ckpt `# change to your path` \
    --batch_size 4 \
    --n_per_prefix 1 \
    --prefix_file assets/prefix_lm1b_examples.jsonl \
    --output_dir outputs/ar_lm1b \
    --warmup_steps 1
