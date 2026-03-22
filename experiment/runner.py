# =============================================================================
# experiment/runner.py — Training and evaluation for one DempsterEDL run
# =============================================================================

import os, time, copy
import numpy as np
import torch
import torch.optim as optim

from config import (NUM_EPOCHS, BATCH_SIZE, SEED, TRACK_DIVERSITY_EVERY, CIFAR10C_SEVERITIES,
                    ID_DATASET, NUM_HEADS, LAMBDA_MIN, LAMBDA_MAX,
                    SELECTIVE_FALLBACK_K)
from models       import get_backbone
from architecture import DempsterEDL
from edl_core     import one_hot, compute_opinion
from evaluation.metrics_exp2 import compute_all_metrics_exp2
from evaluation.cifar10c_eval import evaluate_cifar10c
from data import get_cifar10c_loaders
from experiment.ckpt import ckpt_dir, save_checkpoint, load_checkpoint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── Diversity tracking ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_diversity(model, loader):
    """One backbone pass → Δb and Var(P̂) across all λ heads."""
    model.eval()
    all_b = {i: [] for i in range(model.num_heads)}
    all_P = {i: [] for i in range(model.num_heads)}

    for x, _ in loader:
        logits = model.backbone(x.to(device))
        for i, lam in enumerate(model.prior_lambdas):
            _, _, P, b, _ = compute_opinion(logits, lam, model.num_classes)
            all_b[i].append(b.cpu().numpy())
            all_P[i].append(P.cpu().numpy())

    b_all = [np.concatenate(all_b[i]) for i in range(model.num_heads)]
    P_all = [np.concatenate(all_P[i]) for i in range(model.num_heads)]

    # Δb: mean pairwise L1 belief distance
    pairs, delta_b = 0, 0.0
    for i in range(model.num_heads):
        for j in range(i+1, model.num_heads):
            delta_b += np.abs(b_all[i] - b_all[j]).sum(axis=1).mean()
            pairs   += 1
    delta_b /= max(pairs, 1)

    # Var(P̂): mean predictive variance across heads
    P_stack = np.stack(P_all, axis=0)   # (M, N, K)
    var_P   = P_stack.var(axis=0).mean()

    return {'delta_b': float(delta_b), 'var_P': float(var_P)}


# ── Efficiency ────────────────────────────────────────────────────────────────

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


@torch.no_grad()
def measure_inference_time(model, loader, n_batches=50):
    model.eval()
    for x, _ in loader:
        model(x.to(device)); break   # warmup
    times = []
    for i, (x, _) in enumerate(loader):
        if i >= n_batches: break
        x = x.to(device)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        times.append((time.perf_counter()-t0) / x.size(0) * 1000)
    return float(np.mean(times))


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataloaders, ood_names):
    model.eval()
    all_P, all_u, all_labels, all_nfused = [], [], [], []

    for x, y in dataloaders['val']:
        x = x.to(device)
        P_fused, u_fused, _, _, _, n_fused = model(x)
        all_P.append(P_fused.cpu().numpy())
        all_u.append(u_fused.cpu().numpy())
        all_labels.append(y.numpy())
        all_nfused.append(n_fused.cpu().numpy())

    P_np  = np.concatenate(all_P)
    u_np  = np.concatenate(all_u)
    labs  = np.concatenate(all_labels)
    nf_np = np.concatenate(all_nfused)

    # Uncertainty scores for OOD
    from scipy.special import digamma, gammaln
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    def _scores(loader):
        MP, UM, DE, MI = [], [], [], []
        for x, _ in loader:
            x = x.to(device)
            P_f, u_f, _, _, _, _ = model(x)
            P_np_  = P_f.cpu().numpy()
            u_np_  = u_f.cpu().numpy()
            K      = P_np_.shape[1]
            lam_e  = sum(model.prior_lambdas)/model.num_heads
            S_np_  = (K * lam_e) / (u_np_ + 1e-10)
            a_np_  = P_np_ * S_np_[:, None]
            a0     = a_np_.sum(axis=1)
            MP.append(-P_np_.max(axis=1))
            UM.append(u_np_)
            log_B  = gammaln(a_np_).sum(axis=1) - gammaln(a0)
            de     = (log_B + (a0-K)*digamma(a0)
                      - ((a_np_-1)*digamma(a_np_)).sum(axis=1))
            DE.append(de)
            p_bar  = a_np_ / a0[:, None]
            H_mean = -(p_bar * np.log(p_bar+1e-10)).sum(axis=1)
            E_Hp   = (p_bar*(digamma(a_np_+1)-digamma(a0+1)[:,None])).sum(axis=1)
            MI.append(H_mean - E_Hp)
        return {k: np.concatenate(v) for k, v in
                zip(['MP','UM','DE','MI'], [MP,UM,DE,MI])}

    id_scores  = _scores(dataloaders['val'])
    ood_scores = {n: _scores(dataloaders['ood'][n]) for n in ood_names}

    metrics = compute_all_metrics_exp2(P_np, u_np, labs,
                                        id_scores, ood_scores, ood_names)

    # Mean heads fused (selective mode only)
    metrics['mean_heads_fused'] = float(nf_np.mean())
    metrics['min_heads_fused']  = float(nf_np.min())

    return metrics


# ── Main training function ────────────────────────────────────────────────────

def run_one(fusion: str, dataloaders: dict, num_classes: int,
            ood_names: list, run_id: int = 0,
            num_heads: int = None) -> dict:
    """
    Train and evaluate one DempsterEDL configuration.

    Parameters
    ----------
    fusion      : 'original' | 'selective'
    dataloaders : {'train', 'val', 'ood': {name: loader}}
    num_classes : K
    ood_names   : list of OOD dataset names
    run_id      : seed offset
    """
    print(f"\n{'='*65}")
    print(f"  DempsterEDL — {fusion.upper()}  (run {run_id+1})")
    print(f"  λ heads: {NUM_HEADS}, log-uniform [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print(f"{'='*65}")

    n_heads = num_heads if num_heads is not None else NUM_HEADS
    set_seed(SEED + run_id * 100)
    ckpt_d = ckpt_dir(fusion, run_id)

    # Build model
    backbone = get_backbone(num_classes=num_classes)
    model    = DempsterEDL(
        backbone, num_classes,
        num_heads=n_heads, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        fusion=fusion, fallback_k=SELECTIVE_FALLBACK_K).to(device)

    params = count_params(model)
    print(f"  Params : {params['total']:,}  |  "
          f"λ = {[round(l,3) for l in model.prior_lambdas]}")

    # Optimizer
    lr        = 1e-3 if ID_DATASET == 'mnist' else 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = (optim.lr_scheduler.StepLR(optimizer, 15, 0.1)
                 if ID_DATASET == 'mnist' else
                 optim.lr_scheduler.MultiStepLR(optimizer, [100,150], 0.1))

    # Resume
    _m, _o, metrics_log = load_checkpoint(model, optimizer, ckpt_d, 'latest')
    start_epoch = 0
    if metrics_log is not None:
        model, optimizer = _m, _o
        start_epoch = metrics_log.get('last_epoch', 0) + 1
        print(f"  Resumed from epoch {start_epoch}")
    else:
        metrics_log = {'train_loss':[], 'val_loss':[], 'val_acc':[],
                       'diversity':[], 'last_epoch':-1}

    best_val_acc = 0.0
    best_wts     = copy.deepcopy(model.state_dict())
    t0           = time.time()

    for epoch in range(start_epoch, NUM_EPOCHS):

        # Diversity tracking
        if epoch % TRACK_DIVERSITY_EVERY == 0:
            div = compute_diversity(model, dataloaders['val'])
            div['epoch'] = epoch
            metrics_log['diversity'].append(div)
            print(f"  [ep {epoch:3d}] Δb={div['delta_b']:.5f}  "
                  f"Var(P̂)={div['var_P']:.5f}")

        # Train
        model.train()
        run_loss = 0.0
        for x, y in dataloaders['train']:
            x, y = x.to(device), y.to(device)
            y_oh = one_hot(y, num_classes)
            optimizer.zero_grad()
            P_fused, _, _, _, _, _ = model(x)
            loss = ((y_oh - P_fused)**2).sum(dim=1).mean()
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * x.size(0)
        metrics_log['train_loss'].append(
            run_loss / len(dataloaders['train'].dataset))

        # Validate
        model.eval()
        run_loss = run_corr = 0
        with torch.no_grad():
            for x, y in dataloaders['val']:
                x, y = x.to(device), y.to(device)
                y_oh = one_hot(y, num_classes)
                P_fused, _, _, _, _, _ = model(x)
                run_loss += ((y_oh-P_fused)**2).sum(dim=1).mean().item()*x.size(0)
                run_corr += (P_fused.argmax(1)==y).sum().item()

        val_acc  = run_corr / len(dataloaders['val'].dataset)
        val_loss = run_loss / len(dataloaders['val'].dataset)
        metrics_log['val_loss'].append(val_loss)
        metrics_log['val_acc'].append(val_acc)
        metrics_log['last_epoch'] = epoch

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_wts     = copy.deepcopy(model.state_dict())
            save_checkpoint(model, optimizer, epoch, metrics_log, ckpt_d, 'best')

        scheduler.step()
        if (epoch+1) % max(1, NUM_EPOCHS//5) == 0 or epoch == NUM_EPOCHS-1:
            print(f"  ep {epoch+1:3d}/{NUM_EPOCHS}  "
                  f"val_acc={val_acc:.4f}  val_loss={val_loss:.5f}")
        save_checkpoint(model, optimizer, epoch, metrics_log, ckpt_d, 'latest')

    train_time = time.time() - t0
    print(f"\n  Training done in {train_time:.0f}s")

    # Final evaluation on best checkpoint
    model.load_state_dict(best_wts)
    model.eval()
    final_div = compute_diversity(model, dataloaders['val'])
    metrics   = evaluate(model, dataloaders, ood_names)
    metrics['param_count']      = params['total']
    metrics['train_time_s']     = train_time
    metrics['inference_ms']     = measure_inference_time(model, dataloaders['val'])
    metrics['best_val_acc']     = best_val_acc * 100
    # Note: CIFAR-10-C evaluation is NOT run here.
    # Call eval_cifar10c_best() in main.py after all training completes.

    print(f"\n  ── {fusion.upper()} run {run_id+1} ──")
    print(f"  cls_acc={metrics['cls_acc']:.2f}%  ECE={metrics['ece']:.4f}  "
          f"NLL={metrics['nll']:.4f}")
    print(f"  Mis Detect={metrics['mis_detect_aupr']:.2f}%")
    for n in ood_names:
        print(f"  OOD ({n}): AUPR={metrics[f'aupr_{n}']:.2f}%  "
              f"AUROC={metrics[f'auroc_{n}']:.2f}%  "
              f"FPR95={metrics[f'fpr95_{n}']:.4f}")
    print(f"  Δb={final_div['delta_b']:.5f}  Var(P̂)={final_div['var_P']:.5f}")
    if fusion == 'selective':
        print(f"  Mean heads fused={metrics['mean_heads_fused']:.2f}/"
              f"{NUM_HEADS}  Min={metrics['min_heads_fused']:.0f}")

    return {
        'metrics':     metrics,
        'diversity':   final_div,
        'div_history': metrics_log['diversity'],
        'train_log':   metrics_log,
        'fusion':      fusion,
        'run_id':      run_id,
    }


# ── CIFAR-10-C evaluation on best checkpoint ──────────────────────────────────

def eval_cifar10c_best(fusion: str, num_classes: int,
                       num_heads: int = None,
                       num_runs: int = None) -> dict:
    """
    Load the best checkpoint across ALL runs for a fusion method and
    evaluate on CIFAR-10-C. Called once per method after training.

    Selects the run with the highest best_val_acc from its best.pt —
    this is the strongest model for that method, most representative
    for distribution shift evaluation.

    Parameters
    ----------
    fusion      : 'original' | 'selective' | 'adaptive'
    num_classes : K
    num_heads   : override NUM_HEADS from config
    num_runs    : how many runs to search (default: NUM_RUNS from config)
    """
    from config import NUM_RUNS as _NUM_RUNS
    n_heads  = num_heads if num_heads is not None else NUM_HEADS
    n_runs   = num_runs  if num_runs  is not None else _NUM_RUNS

    # Find best run by val_acc stored in checkpoint
    best_val_acc = -1.0
    best_ckpt_path = None

    for run_id in range(n_runs):
        ckpt_d    = ckpt_dir(fusion, run_id)
        best_path = os.path.join(ckpt_d, 'best.pt')
        if not os.path.exists(best_path):
            continue
        try:
            ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            val_accs = ckpt.get('metrics_log', {}).get('val_acc', [])
            run_best = max(val_accs) if val_accs else -1.0
            if run_best > best_val_acc:
                best_val_acc   = run_best
                best_ckpt_path = best_path
                best_run_id    = run_id
        except Exception as e:
            print(f"  ⚠️  Could not read {best_path}: {e}")
            continue

    if best_ckpt_path is None:
        print(f"  ⚠️  No checkpoints found for {fusion} — skipping CIFAR-10-C")
        return {}

    print(f"  Using run {best_run_id} (val_acc={best_val_acc*100:.2f}%)")

    # Build model and load best weights
    backbone = get_backbone(num_classes=num_classes)
    model    = DempsterEDL(
        backbone, num_classes,
        num_heads=n_heads, lambda_min=LAMBDA_MIN, lambda_max=LAMBDA_MAX,
        fusion=fusion, fallback_k=SELECTIVE_FALLBACK_K).to(device)

    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"\n  CIFAR-10-C evaluation — {fusion.upper()} "
          f"(best of {n_runs} runs, run {best_run_id})")
    cifar10c_loaders = get_cifar10c_loaders(CIFAR10C_SEVERITIES)
    c10c_metrics     = evaluate_cifar10c(model, cifar10c_loaders, device)
    c10c_metrics['best_run_id']  = best_run_id
    c10c_metrics['best_val_acc'] = best_val_acc * 100

    return c10c_metrics
