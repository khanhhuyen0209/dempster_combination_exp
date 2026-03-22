# =============================================================================
# experiment/ckpt.py — Checkpoint save/load
# =============================================================================
import os
import torch
from config import OUTPUT_DIR

CKPT_BASE = os.path.join(OUTPUT_DIR, 'checkpoints')
os.makedirs(CKPT_BASE, exist_ok=True)


def ckpt_dir(fusion: str, run_id: int) -> str:
    d = os.path.join(CKPT_BASE, fusion, f"run{run_id}")
    os.makedirs(d, exist_ok=True)
    return d


def save_checkpoint(model, optimizer, epoch, metrics_log, ckpt_d, tag):
    torch.save({
        'epoch':       epoch,
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'metrics_log': metrics_log,
    }, os.path.join(ckpt_d, f"{tag}.pt"))


def load_checkpoint(model, optimizer, ckpt_d, tag):
    path = os.path.join(ckpt_d, f"{tag}.pt")
    if not os.path.exists(path):
        return None, None, None
    device = next(model.parameters()).device
    ckpt   = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, ckpt['metrics_log']
