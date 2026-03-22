# =============================================================================
# edl_core/loss.py — Re-EDL MSE loss (Eq. 12)
# No Dirichlet variance term — deliberately removed by Re-EDL.
# =============================================================================
import torch
from .opinion import compute_opinion


def re_edl_mse_loss(logits, y_onehot, lam, num_classes):
    """L = (1/N) Σₙ Σₖ (yₖ - Pₖ)²"""
    _, _, P, _, _ = compute_opinion(logits, lam, num_classes)
    return ((y_onehot - P) ** 2).sum(dim=1).mean()
