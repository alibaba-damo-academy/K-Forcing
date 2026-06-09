#!/bin/bash
# AR baseline inference on OpenWebText (seq_len=1024, GPT-2 tokenizer).
# Generates one completion per prefix using temperature-1 sampling with KV-cache.
# This is the teacher model baseline -- compare its outputs against PFLM.
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model ar \
    --task owt \
    --ckpt_path /tmp/pflm_models/ar_openwebtxt.ckpt `# change to your path` \
    --batch_size 4 \
    --n_per_prefix 1 \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --output_dir outputs/ar_owt \
    --warmup_steps 1
