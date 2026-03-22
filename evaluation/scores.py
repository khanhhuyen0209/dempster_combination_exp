# =============================================================================
# evaluation/scores.py
# -----------------------------------------------------------------------------
# All per-sample uncertainty score functions and helpfulness metrics.
#
# Four uncertainty measures (matching paper Table 3 / Table 4):
#   MP  — Max Projected Probability: -max_k(P_k), negated so higher = more OOD
#   UM  — Uncertainty Mass: K·λ / S, the vacuity score from subjective logic
#   DE  — Differential Entropy of the Dirichlet distribution
#   MI  — Mutual Information I(y; p | x) = H[E[p]] − E[H[p]]
#
# Helpfulness metrics used to prove the opinions are useful, not just diverse:
#   OOD detection AUPR / AUROC (ID vs OOD using each uncertainty score)
#   Misclassification detection AUPR / AUROC (correct vs wrong on val set)
#   ECE — Expected Calibration Error (confidence reliability)
#   AURC — Area Under Risk-Coverage curve (selective classification)
# =============================================================================

import numpy as np
import torch
from scipy.special  import digamma, gammaln
from sklearn.metrics import average_precision_score, roc_auc_score

from edl_core  import compute_opinion
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ── Uncertainty score extraction ─────────────────────────────────────────────

@torch.no_grad()
def collect_all_scores(
    model:       torch.nn.Module,
    loader:      torch.utils.data.DataLoader,
    lam:         float,
    num_classes: int,
) -> dict:
    """
    Run a DataLoader through the model and compute all four uncertainty scores.

    Returns dict of (N,) numpy arrays, one per measure:
      'MP' — Max Projected Probability (negated: higher = more uncertain)
      'UM' — Uncertainty Mass (vacuity)
      'DE' — Differential Entropy of Dir(α)
      'MI' — Mutual Information I(y; p | x)
    """
    MP_list, UM_list, DE_list, MI_list = [], [], [], []
    model.eval()

    for inputs, _ in loader:
        inputs   = inputs.to(device)
        alpha, S, P, b, u = compute_opinion(model(inputs), lam, num_classes)

        alpha_np = alpha.cpu().numpy()          # (N, K)
        P_np     = P.cpu().numpy()              # (N, K)
        u_np     = u.cpu().numpy()              # (N,)
        alpha0   = alpha_np.sum(axis=1)         # (N,) = S squeezed

        # MP: negative max projected probability (higher = less confident = more OOD)
        mp = -P_np.max(axis=1)

        # UM: vacuity = K·λ / S (already computed as u)
        um = u_np

        # DE: differential entropy of Dir(α)
        #   log B(α) + (α₀ − K)·ψ(α₀) − Σ_k (αₖ − 1)·ψ(αₖ)
        #   where log B(α) = Σ log Γ(αₖ) − log Γ(α₀)
        log_B = gammaln(alpha_np).sum(axis=1) - gammaln(alpha0)
        de    = (log_B
                 + (alpha0 - num_classes) * digamma(alpha0)
                 - ((alpha_np - 1) * digamma(alpha_np)).sum(axis=1))

        # MI: mutual information I(y; p | x) = H[E[p]] − E[H[p]]
        #   H[E[p]] = -Σ_k p̄_k log p̄_k   (entropy of mean prediction)
        #   E[H[p]] = Σ_k (αₖ/α₀)(ψ(αₖ+1) − ψ(α₀+1))
        p_bar  = alpha_np / alpha0[:, None]
        H_mean = -(p_bar * np.log(p_bar + 1e-10)).sum(axis=1)
        E_Hp   = (p_bar * (digamma(alpha_np + 1)
                           - digamma(alpha0 + 1)[:, None])).sum(axis=1)
        mi     = H_mean - E_Hp

        MP_list.append(mp); UM_list.append(um)
        DE_list.append(de); MI_list.append(mi)

    return {
        'MP': np.concatenate(MP_list),
        'UM': np.concatenate(UM_list),
        'DE': np.concatenate(DE_list),
        'MI': np.concatenate(MI_list),
    }


# ── OOD detection ─────────────────────────────────────────────────────────────

def ood_scores_all_measures(id_scores: dict, ood_scores: dict) -> dict:
    """
    Compute AUPR and AUROC for each of the four uncertainty measures.
    Label convention: ID = 0, OOD = 1 (higher score → OOD).

    Returns nested dict: result[measure] = {'aupr': float, 'auroc': float}
    """
    results = {}
    for measure in ['MP', 'UM', 'DE', 'MI']:
        y_score = np.concatenate([id_scores[measure], ood_scores[measure]])
        y_true  = np.concatenate([
            np.zeros(len(id_scores[measure])),
            np.ones(len(ood_scores[measure])),
        ])
        results[measure] = {
            'aupr':  average_precision_score(y_true, y_score) * 100,
            'auroc': roc_auc_score(y_true, y_score) * 100,
        }
    return results


def ood_scores_um(id_u: np.ndarray, ood_u: np.ndarray) -> tuple:
    """
    AUPR and AUROC using UM (uncertainty mass) only.
    Convenience wrapper kept for fusion evaluation where we average UM across heads.
    """
    y_score = np.concatenate([id_u, ood_u])
    y_true  = np.concatenate([np.zeros(len(id_u)), np.ones(len(ood_u))])
    return (average_precision_score(y_true, y_score) * 100,
            roc_auc_score(y_true, y_score) * 100)


# ── Helpfulness metrics ───────────────────────────────────────────────────────

def expected_calibration_error(
    P_all:  np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    ECE = Σ_b (|B_b| / N) |acc(B_b) − conf(B_b)|

    Groups samples into confidence bins; measures how well confidence
    (max projected probability) tracks actual accuracy in each bin.
    Lower ECE = better calibrated = more trustworthy prediction scores.
    """
    confidences = P_all.max(axis=1)
    predictions = P_all.argmax(axis=1)
    correctness = (predictions == labels).astype(float)
    bins        = np.linspace(0, 1, n_bins + 1)
    ece         = 0.0
    N           = len(labels)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correctness[mask].mean()
        conf = confidences[mask].mean()
        ece += (mask.sum() / N) * abs(acc - conf)

    return ece


def compute_aurc(P_all: np.ndarray, labels: np.ndarray) -> float:
    """
    Area Under Risk-Coverage curve (selective classification).

    Sorts samples by max-probability confidence (highest first), then sweeps
    coverage from 100% → 0%, computing error rate at each threshold.
    AURC = area under the resulting risk-coverage curve.

    Lower AURC = better selective classification — the model correctly
    abstains on hard samples while remaining confident on easy ones.
    """
    confidences       = P_all.max(axis=1)
    errors            = (P_all.argmax(axis=1) != labels).astype(float)
    order             = np.argsort(-confidences)        # descending confidence
    errors_ord        = errors[order]
    N                 = len(errors)
    cumulative_errors = np.cumsum(errors_ord)
    coverages         = np.arange(1, N + 1) / N
    risks             = cumulative_errors / np.arange(1, N + 1)
    return float(np.trapezoid(risks, coverages))


def misclassification_detection(P_all: np.ndarray, u_all: np.ndarray,
                                 labels: np.ndarray) -> dict:
    """
    Misclassification detection using uncertainty mass as the score.
    Higher uncertainty should predict wrong predictions.

    Returns dict with 'aupr' and 'auroc' (both in %).
    Returns nan values if all samples are correct or all are wrong.
    """
    mis = (P_all.argmax(axis=1) != labels).astype(int)
    if mis.any() and not mis.all():
        return {
            'aupr':  average_precision_score(mis, u_all) * 100,
            'auroc': roc_auc_score(mis, u_all) * 100,
        }
    return {'aupr': float('nan'), 'auroc': float('nan')}
