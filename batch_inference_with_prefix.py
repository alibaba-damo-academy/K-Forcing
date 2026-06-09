"""Unified batch inference CLI for AR, MDLM, and PFLM models.

Supports prefix-based completion with throughput measurement.

Based on https://github.com/kuleshov-group/mdlm
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
import transformers
from loguru import logger
from tqdm import tqdm

from utils.tokenizer import get_bos_id, get_tokenizer, resolve_mask_index
from utils.checkpoint import load_ar_model, load_pflm_model

# ------------------------------------------------------------
# Task configs
# ------------------------------------------------------------
TASK_CONFIGS = {
    "owt": {"tokenizer": "gpt2", "seq_len": 1024, "prefix_len": 6, "eos_token": "<|endoftext|>"},
    "lm1b": {"tokenizer": "bert-base-uncased", "seq_len": 128, "prefix_len": 6, "eos_token": "[SEP]"},
}

# ------------------------------------------------------------
# Prefix loading
# ------------------------------------------------------------
def load_prefixes(prefix_file, tokenizer, prefix_len):
    prompts = []
    with open(prefix_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("prefix", "")
            except json.JSONDecodeError:
                text = line
            ids = tokenizer.encode(text, add_special_tokens=False)[:prefix_len]
            if len(ids) == prefix_len:
                prompts.append(ids)
    return prompts

# ------------------------------------------------------------
# EOS truncation
# ------------------------------------------------------------
def truncate_at_eos(ids, eos_token_id, prefix_len):
    for i in range(prefix_len, len(ids)):
        if ids[i] == eos_token_id:
            return ids[: i + 1]
    return ids

# ------------------------------------------------------------
# AR generation (KV-cache)
# ------------------------------------------------------------
@torch.no_grad()
def sample_temperature_one(logits):
    probs = logits.exp()
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)

@torch.no_grad()
def generate_ar_cached(model, prompt_ids, max_new_tokens):
    generated = prompt_ids
    log_probs, kv_caches = model.infer_next_token(
        prompt_ids, kv_caches=None, offset=0, temperature=1.0
    )
    curr_token = sample_temperature_one(log_probs[:, -1, :])
    generated = torch.cat([generated, curr_token], dim=1)

    for _ in range(max_new_tokens - 1):
        offset = generated.shape[1] - 1
        log_probs, kv_caches = model.infer_next_token(
            curr_token, kv_caches=kv_caches, offset=offset, temperature=1.0
        )
        curr_token = sample_temperature_one(log_probs[:, -1, :])
        generated = torch.cat([generated, curr_token], dim=1)

    return generated

# ------------------------------------------------------------
# MDLM sampling (greedy-by-confidence)
# ------------------------------------------------------------
NEG_INF = -1000000.0

def subs_parameterization(logits, xt, mask_index):
    logits[:, :, mask_index] += NEG_INF
    logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    unmasked_indices = (xt != mask_index)
    logits[unmasked_indices] = NEG_INF
    logits[unmasked_indices, xt[unmasked_indices]] = 0
    return logits

@torch.no_grad()
def mdlm_sample_topk_confidence(model, x, mask_index, k, temperature=1.0):
    B, L = x.shape
    device = x.device

    while True:
        mask_pos = (x == mask_index)
        n_masked_per_row = mask_pos.sum(dim=1)
        if n_masked_per_row.max().item() == 0:
            break

        sigma = torch.zeros(B, device=device)
        out = model(x, sigma)

        if isinstance(out, tuple):
            logits = out[0]
        elif hasattr(out, "logits"):
            logits = out.logits
        else:
            logits = out

        logits = subs_parameterization(logits, x, mask_index)

        if temperature != 1.0:
            logits_t = logits / temperature
        else:
            logits_t = logits
        probs = F.softmax(logits_t, dim=-1)

        conf, _ = probs.max(dim=-1)

        flat_probs = probs.view(B * L, -1)
        sampled_tok = torch.multinomial(flat_probs, num_samples=1).view(B, L)

        conf_masked = conf.masked_fill(~mask_pos, -float("inf"))

        new_x = x.clone()
        for b in range(B):
            n_left = int(n_masked_per_row[b].item())
            if n_left == 0:
                continue
            kk = min(k, n_left)
            _, idx = torch.topk(conf_masked[b], kk)
            new_x[b, idx] = sampled_tok[b, idx]
        x = new_x

    return x

def build_mdlm_initial_batch(prompts, seq_len, mask_index, device):
    B = len(prompts)
    x = torch.full((B, seq_len), mask_index, dtype=torch.long, device=device)
    for i, p in enumerate(prompts):
        L = min(len(p), seq_len)
        x[i, :L] = torch.tensor(p[:L], dtype=torch.long, device=device)
    return x


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Unified batch inference for AR / MDLM / PFLM"
    )
    parser.add_argument("--model", choices=["ar", "mdlm", "pflm"], required=True)
    parser.add_argument("--task", choices=["owt", "lm1b"], required=True)
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Local checkpoint path (required for ar/pflm)")
    parser.add_argument("--K", type=int, default=4,
                        help="Tokens per forward pass for mdlm/pflm")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Completions per prefix")
    parser.add_argument("--prefix_file", type=str, default="assets/prefix_examples.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--warmup_steps", type=int, default=1,
                        help="Warmup batches to discard")
    args = parser.parse_args()

    # --- Validate ---
    if args.model in ("ar", "pflm") and args.ckpt_path is None:
        parser.error(f"--ckpt_path is required for --model {args.model}")
    if args.model == "mdlm" and args.task != "owt":
        parser.error("MDLM only supports --task owt (the only released HF checkpoint)")

    task_cfg = TASK_CONFIGS[args.task]
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Tokenizer ---
    tokenizer = get_tokenizer(task_cfg["tokenizer"])
    mask_index, effective_vocab = resolve_mask_index(tokenizer)
    logger.info(
        f"Tokenizer: vocab={tokenizer.vocab_size}, "
        f"mask_index={mask_index}, effective_vocab={effective_vocab}"
    )

    # --- EOS token for truncation ---
    eos_token_id = tokenizer.convert_tokens_to_ids(task_cfg["eos_token"])

    # --- Model ---
    if args.model == "ar":
        model = load_ar_model(args.ckpt_path, device, mask_index)
    elif args.model == "mdlm":
        logger.info("Loading MDLM from HuggingFace: kuleshov-group/mdlm-owt")
        model = transformers.AutoModelForMaskedLM.from_pretrained(
            "kuleshov-group/mdlm-owt", trust_remote_code=True
        )
        model.to(device).eval()
    else:
        model = load_pflm_model(args.ckpt_path, device, mask_index)

    # --- Prefixes ---
    prefixes = load_prefixes(
        args.prefix_file, tokenizer, task_cfg["prefix_len"]
    )
    if not prefixes:
        logger.error(f"No prefixes found for task={args.task} in {args.prefix_file}")
        return

    # Expand by num_samples
    prefixes = prefixes * args.num_samples
    total = len(prefixes)

    # Truncate to clean batches
    n_full = (total // args.batch_size) * args.batch_size
    if n_full == 0:
        logger.error(f"Not enough prefixes ({total}) for batch_size={args.batch_size}")
        return
    if n_full != total:
        logger.info(f"Truncating {total} -> {n_full} for clean batching")
        prefixes = prefixes[:n_full]
        total = n_full
    n_batches = total // args.batch_size

    # --- Compute generation length ---
    if args.model == "ar":
        max_new_tokens = task_cfg["seq_len"] - task_cfg["prefix_len"]
        nfe_per_sample = max_new_tokens
    elif args.model == "mdlm":
        n_to_fill = task_cfg["seq_len"] - task_cfg["prefix_len"]
        est_steps = math.ceil(n_to_fill / args.K)
        nfe_per_sample = est_steps
        max_new_tokens = n_to_fill
    else:
        max_new_tokens = task_cfg["seq_len"] - task_cfg["prefix_len"]
        nfe_per_sample = math.ceil(max_new_tokens / args.K)

    logger.info(
        f"Model={args.model}, Task={args.task}, "
        f"Batch={args.batch_size}, Samples={total}, "
        f"NewTokens={max_new_tokens}"
    )

    # --- Warmup ---
    logger.info(f"Warmup: {args.warmup_steps} batch(es)")
    for b in range(min(args.warmup_steps, n_batches)):
        batch = prefixes[b * args.batch_size : (b + 1) * args.batch_size]
        batch_t = torch.tensor(batch, dtype=torch.long, device=device)
        if args.model == "ar":
            _ = generate_ar_cached(model, batch_t, max_new_tokens)
        elif args.model == "mdlm":
            x0 = build_mdlm_initial_batch(batch, task_cfg["seq_len"], mask_index, device)
            _ = mdlm_sample_topk_confidence(model, x0, mask_index, k=args.K)
        else:
            tau = torch.ones(args.batch_size, device=device)
            _ = model.sample_next_k_tokens_with_kv_caches(
                batch_t, tau, max_new_tokens, k=args.K
            )
    if device == "cuda":
        torch.cuda.synchronize()

    # --- Timed generation ---
    logger.info(f"Timed run: {n_batches} batch(es)")
    samples_path = os.path.join(args.output_dir, "samples.jsonl")
    per_batch_times = []

    if device == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with open(samples_path, "w", encoding="utf-8") as f_out:
        for b in tqdm(range(n_batches), desc="Batches"):
            batch = prefixes[b * args.batch_size : (b + 1) * args.batch_size]
            batch_t = torch.tensor(batch, dtype=torch.long, device=device)

            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            if args.model == "ar":
                out_ids = generate_ar_cached(model, batch_t, max_new_tokens)
            elif args.model == "mdlm":
                x0 = build_mdlm_initial_batch(
                    batch, task_cfg["seq_len"], mask_index, device
                )
                out_ids = mdlm_sample_topk_confidence(
                    model, x0, mask_index, k=args.K
                )
            else:
                tau = torch.ones(args.batch_size, device=device)
                out_ids = model.sample_next_k_tokens_with_kv_caches(
                    batch_t, tau, max_new_tokens, k=args.K
                )

            if device == "cuda":
                torch.cuda.synchronize()
            per_batch_times.append(time.perf_counter() - t0)

            for i in range(args.batch_size):
                ids = out_ids[i].tolist()
                ids_trunc = truncate_at_eos(ids, eos_token_id, task_cfg["prefix_len"])
                prefix_text = tokenizer.decode(
                    ids_trunc[: task_cfg["prefix_len"]], skip_special_tokens=False
                )
                completion_text = tokenizer.decode(
                    ids_trunc[task_cfg["prefix_len"]:], skip_special_tokens=False
                )
                f_out.write(
                    json.dumps(
                        {"prefix": prefix_text, "completion": completion_text},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            f_out.flush()

    if device == "cuda":
        torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - t_start

    # --- Summary ---
    new_tokens_total = total * max_new_tokens
    throughput = new_tokens_total / total_elapsed
    avg_batch_time = sum(per_batch_times) / len(per_batch_times)
    batch_throughput = (args.batch_size * max_new_tokens) / avg_batch_time
    total_nfe = total * nfe_per_sample
    nfe_per_token = nfe_per_sample / max_new_tokens

    stats = {
        "model": args.model,
        "task": args.task,
        "batch_size": args.batch_size,
        "num_samples": total,
        "max_new_tokens": max_new_tokens,
        "K_per_step": args.K if args.model != "ar" else 1,
        "prefix_len": task_cfg["prefix_len"],
        "n_batches": n_batches,
        "n_warmup_batches": args.warmup_steps,
        "total_elapsed_sec": total_elapsed,
        "avg_batch_time_sec": avg_batch_time,
        "throughput_tok_per_sec_total": throughput,
        "throughput_tok_per_sec_avg_batch": batch_throughput,
        "total_nfe": total_nfe,
        "nfe_per_token": nfe_per_token,
        "device": device,
    }

    stats_path = os.path.join(args.output_dir, "throughput.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n========== Summary ==========")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nSamples: {samples_path}")
    print(f"Stats:   {stats_path}")

    if args.model == "mdlm":
        print(
            "\nNote: MDLM throughput is NOT directly comparable to AR/PFLM. "
            "MDLM uses a bidirectional transformer (full-sequence attention each "
            "step) vs. causal KV-cache for AR/PFLM."
        )


if __name__ == "__main__":
    main()
