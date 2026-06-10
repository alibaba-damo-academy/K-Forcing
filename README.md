<h1 align="center">K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling</h1>

<p align="center"><em>A new language modeling paradigm that decodes multiple tokens jointly in one forward pass, enabling batch-friendly inference speedup.</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.10820"><img src="https://img.shields.io/badge/arXiv-2606.10820-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/zwave/K-Forcing"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow" alt="HuggingFace"></a>
</p>

## TODO

- [x] ~~Arxiv paper release~~ &ensp; <img src="https://img.shields.io/badge/-done-brightgreen" height="16">
- [x] ~~Checkpoints release~~ &ensp; <img src="https://img.shields.io/badge/-done-brightgreen" height="16">
- [ ] Blog post
- [ ] Training recipe
- [ ] Future Direction

## Introduction

K-Forcing distills an autoregressive (AR) language model into a **push-forward language model (PFLM)** that generates **k tokens in one forward pass**. It takes k independent uniform noise variables as input and maps them to k future tokens jointly. The output length is always fixed, and the AR backbone is reused as-is, making it a new paradigm especially suitable for batch serving.

<p align="center">
  <img src="assets/paradigm4.png" width="100%" />
</p>

**(a) K-Forcing (ours)**: maps k noise tokens to k future tokens in one pass, modeling their joint distribution.

**(b) AR**: generates one token per step — simple but memory-bound.

**(c) Speculative decoding**: drafts multiple tokens then verifies — output length varies, breaking regular batching.

**(d) MDLM**: predicts masked positions in parallel but independently (per-position marginals, not joint).

## Venv Setup

```bash
# 1. Download flash-attn wheel
mkdir -p wheels
wget -P wheels https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.6/flash_attn-2.5.6+cu122torch2.2cxx11abiFALSE-cp39-cp39-linux_x86_64.whl

# 2. Install
uv sync
```

## Checkpoints

All checkpoints are hosted at [<img src="https://img.shields.io/badge/%F0%9F%A4%97-zwave/K--Forcing-yellow" height="18">](https://huggingface.co/zwave/K-Forcing). 

The MDLM baseline uses the checkpoint from [<img src="https://img.shields.io/badge/%F0%9F%A4%97-kuleshov--group/mdlm--owt-yellow" height="18">](https://huggingface.co/kuleshov-group/mdlm-owt).

| Model | Dataset | Filename |
|-------|---------|----------|
| AR    | OWT     | `ar_openwebtxt.ckpt` |
| AR    | LM1B    | `ar_best_lm1b.ckpt` |
| PFLM (k=4) | OWT | `pflm_owt_k4.ckpt` |
| PFLM (k=4) | LM1B | `pflm_lm1b_k4.ckpt` |

## Inference

`batch_inference_with_prefix.py` supports AR, PFLM, and MDLM inference with batched generation from text prefixes all in one script:

- **AR**: temperature-1 sampling with KV-cache.
- **PFLM**: push-forward sampling with KV-cache, arbitrary K (up to 4), with optional frequency penalty.
- **MDLM**: iterative unmasking with arbitrary K, supporting both top-k-by-confidence and fully greedy decoding.

Checkpoints are auto-downloaded from HuggingFace [<img src="https://img.shields.io/badge/%F0%9F%A4%97-zwave/K--Forcing-yellow" height="18">](https://huggingface.co/zwave/K-Forcing) when `--ckpt_path` is omitted. 

Run `python batch_inference_with_prefix.py -h` for the full list of arguments. Example usages:

```bash
# AR (auto-downloads from HuggingFace if --ckpt_path is omitted)
python batch_inference_with_prefix.py \
    --model ar --task owt \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --batch_size 4 --n_per_prefix 1

# PFLM (K=2 tokens per forward pass)
python batch_inference_with_prefix.py \
    --model pflm --task owt \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --batch_size 4 --n_per_prefix 1 --K 2 --freq_penalty 0.3

# MDLM
python batch_inference_with_prefix.py \
    --model mdlm --task owt \
    --prefix_file assets/prefix_owt_examples.jsonl \
    --batch_size 4 --n_per_prefix 1 --K 2
```

See `scripts/` for complete inference scripts with instructions.

## Acknowledgements

A large portion of this codebase is built upon [<img src="https://img.shields.io/badge/GitHub-MDLM-blue?logo=github" height="18">](https://github.com/kuleshov-group/mdlm). We thank the authors for open-sourcing their code.

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{tang2026kforcingjointnextktokendecoding,
      title={K-Forcing: Joint Next-K-Token Decoding via Push-Forward Language Modeling}, 
      author={Zhiwei Tang and Yuanyu He and Yizheng Han and Wangbo Zhao and Jiasheng Tang and Fan Wang and Bohan Zhuang},
      year={2026},
      eprint={2606.10820},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.10820}, 
}
```
