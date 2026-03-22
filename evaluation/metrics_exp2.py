# =============================================================================
# evaluation/metrics_exp2.py  —  Experiment 2
# -----------------------------------------------------------------------------
# Additional metrics needed to compare directly with Deep Ensemble and
# Bayesian uncertainty methods. All metrics use the same conventions as
# the benchmark papers (Lakshminarayanan 2017, Ovadia 2019).
#
# Metrics added beyond Experiment 1:
#   NLL      — Negative Log Likelihood (proper scoring rule)
#   Brier    — Brier Score = Σ(y - P)² (same form as Re-EDL loss on test set)
#   FPR@95   — False Positive Rate at 95% True Positive Rate (OOD)
#
# These three + ECE + AUROC cover all metrics reported in Deep Ensemble
# and MC Dropout papers, enabling direct comparison.
# =============================================================================

import numpy as np
from sklearn.metrics import roc_curve


def negative_log_likelihood(P_all: np.ndarray, labels: np.ndarray) -> float:
    """
    NLL = -(1/N) Σ log P[yₙ]
    where P[yₙ] is the fused projected probability for the true class.

    Lower = better. A perfectly calibrated model achieves NLL = 0.
    Sensitive to overconfidence — a model that says P=0.99 on a wrong
    prediction is penalised heavily.

    Uses fused projected probability P_fused, not softmax output.
    Clip at 1e-10 to avoid log(0).
    """
    N = len(labels)
    true_class_probs = P_all[np.arange(N), labels]   # (N,) prob of true class
    true_class_probs = np.clip(true_class_probs, 1e-10, 1.0)
    return float(-np.log(true_class_probs).mean())


def brier_score(P_all: np.ndarray, labels: np.ndarray) -> float:
    """
    Brier Score = (1/N) Σ Σ_k (yₖ - Pₖ)²
    Same form as Re-EDL's MSE loss evaluated on the test set.
    Lower = better. Measures both calibration and accuracy jointly.

    Ranges from 0 (perfect) to 2 (worst).
    For a uniform predictor: BS = 1 - 1/K.
    """
    N, K   = P_all.shape
    y_onehot = np.zeros_like(P_all)
    y_onehot[np.arange(N), labels] = 1.0
    return float(((y_onehot - P_all) ** 2).sum(axis=1).mean())


def fpr_at_95_tpr(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    FPR@95TPR — False Positive Rate when True Positive Rate = 95%.
    Standard OOD detection metric in deep ensemble / MC Dropout papers.

    Convention: ID = negative (label 0), OOD = positive (label 1).
    Score = uncertainty (higher = more likely OOD).

    Lower FPR@95 = better OOD detection.
    A random detector achieves FPR@95 ≈ 0.95.
    """
    y_score = np.concatenate([id_scores,  ood_scores])
    y_true  = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores))
    ])
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # Find FPR at closest TPR to 0.95
    idx = np.argmin(np.abs(tpr - 0.95))
    return float(fpr[idx])


def compute_all_metrics_exp2(
    P_fused:    np.ndarray,
    u_fused:    np.ndarray,
    labels:     np.ndarray,
    id_scores:  dict,          # {'MP':..., 'UM':..., 'DE':..., 'MI':...}
    ood_scores: dict,          # {ood_name: {'MP':..., 'UM':..., 'DE':..., 'MI':...}}
    ood_names:  list,
) -> dict:
    """
    Compute the full metric set for Experiment 2, covering both:
      Table A: Re-EDL paper format (cls_acc, mis_detect AUPR, OOD AUPR)
      Table B: Deep Ensemble format (NLL, Brier, ECE, OOD AUROC, FPR@95)

    Parameters
    ----------
    P_fused    : (N, K) fused projected probabilities
    u_fused    : (N,)   fused vacuity
    labels     : (N,)   ground-truth class indices
    id_scores  : dict of (N,) uncertainty scores on ID val set
    ood_scores : {ood_name: dict of (N,) scores}

    Returns
    -------
    Flat dict of all scalar metrics
    """
    from evaluation.scores import (expected_calibration_error, compute_aurc,
                                    misclassification_detection)
    from sklearn.metrics import average_precision_score, roc_auc_score

    m = {}

    # ── Classification ────────────────────────────────────────────────────────
    preds          = P_fused.argmax(axis=1)
    m['cls_acc']   = float((preds == labels).mean() * 100)

    # ── Calibration ──────────────────────────────────────────────────────────
    m['ece']       = float(expected_calibration_error(P_fused, labels))
    m['nll']       = negative_log_likelihood(P_fused, labels)
    m['brier']     = brier_score(P_fused, labels)
    m['aurc']      = float(compute_aurc(P_fused, labels))

    # ── Misclassification detection (aleatoric) ───────────────────────────────
    # Paper convention: correct=1, confidence=max(P) — matches Re-EDL Table 2
    correct        = (preds == labels).astype(int)
    conf           = P_fused.max(axis=1)
    if correct.any() and not correct.all():
        m['mis_detect_aupr']  = float(
            average_precision_score(correct, conf) * 100)
        m['mis_detect_auroc'] = float(
            roc_auc_score(correct, conf) * 100)
    else:
        m['mis_detect_aupr'] = m['mis_detect_auroc'] = float('nan')

    # ── OOD detection (epistemic) per dataset, all four measures ─────────────
    for ood_name in ood_names:
        ood_sc = ood_scores[ood_name]
        for meas in ['MP', 'UM', 'DE', 'MI']:
            id_s  = id_scores[meas]
            ood_s = ood_sc[meas]
            y_score = np.concatenate([id_s,  ood_s])
            y_true  = np.concatenate([
                np.zeros(len(id_s)), np.ones(len(ood_s))])
            m[f'{meas}_aupr_{ood_name}']  = float(
                average_precision_score(y_true, y_score) * 100)
            m[f'{meas}_auroc_{ood_name}'] = float(
                roc_auc_score(y_true, y_score) * 100)

            # FPR@95TPR (UM score only — matches DE paper convention)
            if meas == 'UM':
                m[f'fpr95_{ood_name}'] = fpr_at_95_tpr(id_s, ood_s)

        # Shortcut keys for UM (primary OOD measure)
        m[f'aupr_{ood_name}']  = m[f'UM_aupr_{ood_name}']
        m[f'auroc_{ood_name}'] = m[f'UM_auroc_{ood_name}']

    m['mean_ood_aupr']  = float(np.mean([m[f'aupr_{n}']  for n in ood_names]))
    m['mean_ood_auroc'] = float(np.mean([m[f'auroc_{n}'] for n in ood_names]))
    m['mean_fpr95']     = float(np.mean([m[f'fpr95_{n}'] for n in ood_names]))

    return m
