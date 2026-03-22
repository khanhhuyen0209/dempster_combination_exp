# =============================================================================
# architecture/dempster.py — Two Dirichlet fusion methods
# =============================================================================
#
# Method 1 — Original CBF:
#   Standard Josang Cumulative Belief Fusion across all N heads.
#   Self-weighting: uncertain heads are automatically downweighted.
#
# Method 2 — Selective CBF:
#   Fuse only heads that agree with the majority prediction.
#   Fallback: when no consensus exists (OOD-like inputs), use top-K
#   high-λ (cautious) heads only.
#   Rationale: on OOD inputs heads disagree → fewer heads fuse →
#   higher residual u_fused → stronger OOD signal.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from edl_core import compute_opinion


class TemperatureNet(nn.Module):
    """
    Direction 1 — Input-adaptive Dempster.
    Learns per-sample per-head temperature to scale vacuity before CBF.
    A high temperature on head i increases u_i → that head gets downweighted.

    Input : raw logits (N, K)
    Output: temperatures (N, num_heads) — positive via softplus
    Architecture: Linear(K, 32) → ReLU → Linear(32, num_heads) → Softplus
    Params: K*32 + 32*M ≈ tiny vs backbone
    """
    def __init__(self, num_classes: int, num_heads: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_classes, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
            nn.Softplus(),   # ensures positive temperatures
        )

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns (N, num_heads) positive temperature scalars."""
        # Detach logits so temperature net doesn't interfere with backbone grad
        return self.net(logits.detach()) + 0.1   # minimum temperature = 0.1


class DempsterEDL(nn.Module):
    """
    Shared backbone + N Re-EDL heads fused by Dirichlet combination.

    Three fusion methods:
      'original'  — Josang CBF across all heads
      'selective' — CBF on agreeing heads; high-λ fallback on OOD
      'adaptive'  — CBF with input-adaptive temperature scaling (Direction 1)

    Parameters
    ----------
    backbone     : shared feature extractor outputting raw logits
    num_classes  : K
    num_heads    : N (hyperparameter, range 6-10, default 7)
    lambda_min   : smallest prior weight (default 0.1)
    lambda_max   : largest prior weight (default 3.5)
    fusion       : 'original' | 'selective' | 'adaptive'
    fallback_k   : high-λ fallback heads for selective mode
    """

    def __init__(self, backbone: nn.Module, num_classes: int,
                 num_heads: int = 7, lambda_min: float = 0.1,
                 lambda_max: float = 3.5, fusion: str = 'original',
                 fallback_k: int = 3):
        super().__init__()
        self.backbone    = backbone
        self.num_classes = num_classes
        self.num_heads   = num_heads
        self.fusion      = fusion
        self.fallback_k  = fallback_k

        # Log-uniform λ spacing
        self.prior_lambdas = torch.logspace(
            math.log10(lambda_min),
            math.log10(lambda_max),
            num_heads).tolist()

        # Direction 1: temperature network for adaptive fusion
        if fusion == 'adaptive':
            self.temp_net = TemperatureNet(num_classes, num_heads)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        P_fused  : (N, K)
        u_fused  : (N,)
        b_fused  : (N, K)
        alphas   : list of (N, K) per-head alpha tensors
        opinions : list of (P_i, u_i) tuples
        n_fused  : (N,) number of heads actually fused per sample
        """
        logits = self.backbone(x)   # single forward pass

        alphas   = []
        opinions = []
        for lam in self.prior_lambdas:
            alpha, _, P, b, u = compute_opinion(logits, lam, self.num_classes)
            alphas.append(alpha)
            opinions.append((P, u))

        if self.fusion == 'original':
            alpha_fused, n_fused = _cbf_all(alphas, self.prior_lambdas,
                                             self.num_classes)
        elif self.fusion == 'selective':
            alpha_fused, n_fused = _cbf_selective(
                alphas, opinions, self.prior_lambdas,
                self.num_classes, self.fallback_k)
        else:  # adaptive
            # Temperature scaling: scale each head's u before CBF
            # High temperature → higher effective u → head downweighted
            temps = self.temp_net(logits)   # (N, M) positive scalars
            scaled_alphas = []
            scaled_lambdas = []
            for i, (alpha, lam) in enumerate(zip(alphas, self.prior_lambdas)):
                # Scale evidence by temperature: higher temp = more uncertain
                tau   = temps[:, i:i+1]                # (N, 1)
                S     = alpha.sum(dim=1, keepdim=True) # (N, 1)
                e     = alpha - lam
                # Apply temperature to evidence (shrink evidence = more uncertain)
                e_scaled   = e / tau
                alpha_scaled = e_scaled + lam
                lam_eff    = float(lam)
                scaled_alphas.append(alpha_scaled.clamp(min=lam_eff))
                scaled_lambdas.append(lam_eff)
            alpha_fused, n_fused = _cbf_all(scaled_alphas, scaled_lambdas,
                                             self.num_classes)

        # Derive fused opinion
        S_fused = alpha_fused.sum(dim=1, keepdim=True)
        P_fused = alpha_fused / S_fused
        # Effective λ = weighted mean of fused head λ values (approx mean)
        lam_eff = sum(self.prior_lambdas) / self.num_heads
        u_fused = (self.num_classes * lam_eff) / S_fused.squeeze(1)
        b_fused = (alpha_fused - lam_eff) / S_fused

        return P_fused, u_fused, b_fused, alphas, opinions, n_fused


# ── CBF helpers ───────────────────────────────────────────────────────────────

def _cbf_pair(alpha_A, lam_A, alpha_B, lam_B, num_classes):
    """
    Pairwise Josang CBF for two Dirichlet opinions.
    harmony = u_A + u_B - u_A·u_B
    b_fused = (b_A·u_B + b_B·u_A) / harmony
    u_fused = (u_A·u_B) / harmony
    """
    S_A    = alpha_A.sum(dim=1, keepdim=True)
    S_B    = alpha_B.sum(dim=1, keepdim=True)
    u_A    = (num_classes * lam_A) / S_A.squeeze(1)
    u_B    = (num_classes * lam_B) / S_B.squeeze(1)
    b_A    = (alpha_A - lam_A) / S_A
    b_B    = (alpha_B - lam_B) / S_B

    harmony = (u_A + u_B - u_A * u_B).clamp(min=1e-10)   # (N,)
    b_fused = (b_A * u_B.unsqueeze(1) +
               b_B * u_A.unsqueeze(1)) / harmony.unsqueeze(1)
    u_fused = (u_A * u_B) / harmony

    # Convert back to alpha: α = b·S + λ
    # Use mean λ as effective prior for fused opinion
    lam_eff  = (lam_A + lam_B) / 2
    S_fused  = (num_classes * lam_eff) / (u_fused + 1e-10)
    alpha_out = b_fused * S_fused.unsqueeze(1) + lam_eff
    return alpha_out.clamp(min=lam_eff)


def _cbf_all(alphas, lambdas, num_classes):
    """Iterative pairwise CBF across all heads."""
    result = alphas[0]
    lam_eff = lambdas[0]
    for i in range(1, len(alphas)):
        result  = _cbf_pair(result, lam_eff, alphas[i], lambdas[i], num_classes)
        lam_eff = (lam_eff + lambdas[i]) / 2
    n_fused = torch.full((alphas[0].shape[0],), len(alphas),
                          dtype=torch.float, device=alphas[0].device)
    return result, n_fused


def _cbf_selective(alphas, opinions, lambdas, num_classes, fallback_k):
    """
    Selective CBF — fuse only heads agreeing with majority prediction.

    Fallback B: when no consensus, fuse top-K high-λ (cautious) heads.
    High-λ heads are cautious by design → high u_fused → strong OOD signal.
    """
    N      = alphas[0].shape[0]
    device = alphas[0].device

    # Per-sample per-head predictions
    preds = torch.stack([op[0].argmax(dim=1) for op in opinions],
                        dim=1)   # (N, M)

    # Majority vote per sample
    majority = torch.mode(preds, dim=1).values   # (N,)

    # Agreement mask (N, M)
    agree = (preds == majority.unsqueeze(1))   # (N, M) bool

    # Fallback heads: top-K highest λ (last fallback_k in sorted list)
    fallback_idx = list(range(len(lambdas) - fallback_k, len(lambdas)))

    results  = []
    n_fused  = torch.zeros(N, device=device)

    for n in range(N):
        agree_idx = agree[n].nonzero(as_tuple=True)[0].tolist()

        if len(agree_idx) >= 2:
            # Normal: fuse agreeing heads
            sel_alphas  = [alphas[i][n:n+1] for i in agree_idx]
            sel_lambdas = [lambdas[i]        for i in agree_idx]
        else:
            # Fallback B: no consensus → use cautious high-λ heads
            sel_alphas  = [alphas[i][n:n+1] for i in fallback_idx]
            sel_lambdas = [lambdas[i]        for i in fallback_idx]
            agree_idx   = fallback_idx

        # Fuse selected heads
        fused = sel_alphas[0]
        lam   = sel_lambdas[0]
        for i in range(1, len(sel_alphas)):
            fused = _cbf_pair(fused, lam, sel_alphas[i],
                              sel_lambdas[i], num_classes)
            lam   = (lam + sel_lambdas[i]) / 2

        results.append(fused)
        n_fused[n] = len(agree_idx)

    return torch.cat(results, dim=0), n_fused
