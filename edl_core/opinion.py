# =============================================================================
# edl_core/opinion.py — Re-EDL opinion computation
# =============================================================================
import torch
import torch.nn.functional as F


def compute_opinion(logits: torch.Tensor, lam: float,
                    num_classes: int) -> tuple:
    """
    Map raw logits to a Dirichlet opinion via Re-EDL formulation.

    e = softplus(logits)     evidence
    α = e + λ               concentration
    S = Σ αₖ                strength
    P = α / S               projected probability
    b = e / S               belief mass
    u = K·λ / S             vacuity
    """
    e     = F.softplus(logits)
    alpha = e + lam
    S     = alpha.sum(dim=1, keepdim=True)
    P     = alpha / S
    b     = e / S
    u     = (num_classes * lam) / S.squeeze(1)
    return alpha, S, P, b, u


def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels, num_classes).float()
