#!/bin/bash
# MDLM baseline inference on OpenWebText (seq_len=1024, GPT-2 tokenizer).
# Uses top-k-by-confidence iterative unmasking (unmask K most confident tokens per step).
# Loads the HuggingFace checkpoint kuleshov-group/mdlm-owt (no local ckpt needed).
#
# --mdlm_greedy: greedy decoding (argmax) vs default temperature-1 sampling.
#   Greedy collapses into repetition loops; kept here for completeness.
#
# Note: MDLM throughput is NOT directly comparable to AR/PFLM because MDLM uses
# full-sequence bidirectional attention each step (no KV-cache).
set -euo pipefail

uv run python batch_inference_with_prefix.py \
    --model mdlm \
    --task owt \
    --batch_size 4 \
    --n_per_prefix 1 \
    --K 2 \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --output_dir outputs/mdlm_owt_greedy \
    --warmup_steps 1
    # Add --mdlm_greedy for fully greedy decoding
