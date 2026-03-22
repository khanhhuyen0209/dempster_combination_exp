# =============================================================================
# evaluation/cifar10c_eval.py — CIFAR-10-C distribution shift evaluation
# -----------------------------------------------------------------------------
# Following F-EDL paper (arXiv:2510.18322):
#   - AUPR per severity level (C=1..5)
#   - Averaged across 19 corruption types
#   - Uses aleatoric uncertainty (correct=1, max(P)) — mis-detect convention
# =============================================================================

import numpy as np
import torch
from sklearn.metrics import average_precision_score


@torch.no_grad()
def evaluate_cifar10c(model, cifar10c_loaders: dict, device) -> dict:
    """
    Evaluate DempsterEDL on CIFAR-10-C across all severities.

    Parameters
    ----------
    model            : DempsterEDL (best checkpoint loaded)
    cifar10c_loaders : {severity: DataLoader} from get_cifar10c_loaders()
    device           : torch device

    Returns
    -------
    dict with keys:
        'aupr_sev_{s}'   : AUPR at severity s (aleatoric, correct=1, max(P))
        'aupr_mean'      : mean AUPR across severities
        'acc_sev_{s}'    : accuracy at severity s
        'acc_mean'       : mean accuracy across severities
    """
    model.eval()
    results = {}

    for sev, loader in cifar10c_loaders.items():
        all_P, all_labels = [], []
        for x, y in loader:
            x = x.to(device)
            P_fused, _, _, _, _, _ = model(x)
            all_P.append(P_fused.cpu().numpy())
            all_labels.append(y.numpy())

        P_np   = np.concatenate(all_P)
        labs   = np.concatenate(all_labels)
        preds  = P_np.argmax(axis=1)

        # Accuracy
        acc = float((preds == labs).mean() * 100)
        results[f'acc_sev_{sev}'] = acc

        # AUPR — aleatoric convention: correct=1, confidence=max(P)
        correct = (preds == labs).astype(int)
        conf    = P_np.max(axis=1)
        if correct.any() and not correct.all():
            aupr = float(average_precision_score(correct, conf) * 100)
        else:
            aupr = float('nan')
        results[f'aupr_sev_{sev}'] = aupr

        print(f"  CIFAR-10-C severity {sev}: "
              f"acc={acc:.2f}%  AUPR={aupr:.2f}%")

    severities = list(cifar10c_loaders.keys())
    results['aupr_mean'] = float(np.nanmean(
        [results[f'aupr_sev_{s}'] for s in severities]))
    results['acc_mean']  = float(np.mean(
        [results[f'acc_sev_{s}'] for s in severities]))

    print(f"  CIFAR-10-C mean: acc={results['acc_mean']:.2f}%  "
          f"AUPR={results['aupr_mean']:.2f}%")
    return results
