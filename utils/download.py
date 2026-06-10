"""Auto-download checkpoints from HuggingFace."""

from huggingface_hub import hf_hub_download

REPO_ID = "zwave/K-Forcing"

HF_FILENAMES = {
    ("ar", "owt"): "ar_openwebtxt.ckpt",
    ("ar", "lm1b"): "ar_best_lm1b.ckpt",
    ("pflm", "owt"): "pflm_owt_k4.ckpt",
    ("pflm", "lm1b"): "pflm_lm1b_k4.ckpt",
}


def get_checkpoint(model: str, task: str) -> str:
    """Download a checkpoint from HuggingFace and return the local path."""
    key = (model, task)
    if key not in HF_FILENAMES:
        raise ValueError(f"No checkpoint for model={model}, task={task}")
    return hf_hub_download(repo_id=REPO_ID, filename=HF_FILENAMES[key])
