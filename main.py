#!/usr/bin/env python3
# =============================================================================
# main.py — Dempster-EDL
# -----------------------------------------------------------------------------
# Usage:
#   python main.py                          # both methods, config defaults
#   python main.py --fusion original        # original CBF only
#   python main.py --fusion selective       # selective CBF only
#   python main.py --runs 1                 # smoke test
#   python main.py --exp cifar100           # CIFAR-100 as ID
# =============================================================================

import os, pickle, argparse
import numpy as np
import pandas as pd
import torch

from config import (ID_DATASET, NUM_RUNS, NUM_EPOCHS, FUSION_METHODS,
                    NUM_HEADS, LAMBDA_MIN, LAMBDA_MAX,
                    OUTPUT_DIR, out)
from data        import get_dataloaders
from experiment  import run_one, eval_cifar10c_best
from experiment.ckpt import ckpt_dir as _ckpt_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Dempster-EDL')
    parser.add_argument('--eval-c10c', action='store_true',
                        help='Run CIFAR-10-C evaluation on best checkpoints only.\n'
                             'Does not train. Run after training completes.\n'
                             'Uses run0/best.pt per fusion method.')
    parser.add_argument('--fusion', nargs='+',
                        choices=['original', 'selective', 'adaptive'],
                        default=FUSION_METHODS,
                        help='Fusion method(s): original | selective | adaptive')
    parser.add_argument('--heads', type=int, default=None,
                        help=f'Number of λ heads (range 6-10, default: {NUM_HEADS})')
    parser.add_argument('--runs',  type=int, default=None,
                        help='Number of runs (default: NUM_RUNS from config)')
    parser.add_argument('--exp',
                        choices=['classical', 'cifar100', 'lt_mild', 'lt_heavy'],
                        default='classical')
    return parser.parse_args()


args   = parse_args()
N_RUNS  = args.runs  if args.runs  is not None else NUM_RUNS
N_HEADS = args.heads if args.heads is not None else NUM_HEADS

# Validate heads range
if not (6 <= N_HEADS <= 10):
    print(f"⚠️  --heads {N_HEADS} outside recommended range [6,10]")

EXP_DATASET = {
    'classical': 'cifar10',
    'cifar100':  'cifar100',
    'lt_mild':   'cifar10_lt',
    'lt_heavy':  'cifar10_lt',
}[args.exp]

EXP_IMBALANCE = {'lt_mild': 0.1, 'lt_heavy': 0.01}.get(args.exp, None)

print(f"\n{'='*65}")
print(f"  Dempster-EDL  |  {EXP_DATASET}  |  {args.exp}")
print(f"  Methods : {args.fusion}")
print(f"  Heads   : {N_HEADS}  λ=[{LAMBDA_MIN}, {LAMBDA_MAX}] log-uniform")
print(f"  Runs    : {N_RUNS}  |  Epochs: {NUM_EPOCHS}")
print(f"{'='*65}")

# ── Data ──────────────────────────────────────────────────────────────────────
dataloaders, num_classes, _ = get_dataloaders(EXP_DATASET, EXP_IMBALANCE)
ood_names = list(dataloaders['ood'].keys())
print(f"  OOD sets: {ood_names}\n")

# ── CIFAR-10-C only mode ──────────────────────────────────────────────────────
if args.eval_c10c:
    if EXP_DATASET != 'cifar10':
        print("❌ --eval-c10c only applies to cifar10 setting")
        exit(1)
    print(f"\n{'='*65}")
    print(f"  CIFAR-10-C evaluation — best checkpoints only")
    print(f"  Methods: {args.fusion}")
    print(f"{'='*65}")

    c10c_results = {}
    for fusion in args.fusion:
        c10c_results[fusion] = eval_cifar10c_best(
            fusion, num_classes, num_heads=N_HEADS, num_runs=N_RUNS)

    # Print and save CIFAR-10-C table
    import pandas as pd
    rows = []
    for fusion, metrics in c10c_results.items():
        if not metrics: continue
        row = {'Method': fusion}
        for sev in [1, 2, 3, 4, 5]:
            row[f'AUPR C={sev}'] = f"{metrics.get(f'aupr_sev_{sev}', float('nan')):.2f}%"
            row[f'Acc C={sev}']  = f"{metrics.get(f'acc_sev_{sev}',  float('nan')):.2f}%"
        row['AUPR mean'] = f"{metrics.get('aupr_mean', float('nan')):.2f}%"
        row['Acc mean']  = f"{metrics.get('acc_mean',  float('nan')):.2f}%"
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        path = out(f'cifar10c_{args.exp}_results.csv')
        df.to_csv(path, index=False)
        print(f"\nCIFAR-10-C Results:")
        print(df.to_string(index=False))
        print(f"\n✅ Saved to {path}")
    exit(0)

# ── Run ───────────────────────────────────────────────────────────────────────
all_results = {}

for fusion in args.fusion:
    all_results[fusion] = []
    for run_id in range(N_RUNS):

        # Check if already complete
        _last = os.path.join(_ckpt_dir(fusion, run_id), 'latest.pt')
        if os.path.exists(_last):
            try:
                _s = torch.load(_last, map_location='cpu', weights_only=False)
                if _s.get('epoch', -1) >= NUM_EPOCHS - 1:
                    print(f"  ✓ {fusion} run {run_id+1} complete — eval only")
            except Exception: pass

        result = run_one(fusion, dataloaders, num_classes, ood_names,
                         run_id, num_heads=N_HEADS)
        all_results[fusion].append(result)

# ── Aggregate ─────────────────────────────────────────────────────────────────

def agg(results, key):
    vals = [r['metrics'][key] for r in results
            if not np.isnan(float(r['metrics'].get(key, float('nan'))))]
    return (np.mean(vals), np.std(vals)) if vals else (float('nan'), float('nan'))

def fmt(m, s): return f"{m:.2f}±{s:.2f}"

# ── Tables ────────────────────────────────────────────────────────────────────
print(f"\n{'═'*80}")
print(f"RESULTS — {EXP_DATASET.upper()}  ({args.exp})")
print(f"{'═'*80}")

# Table A — Re-EDL format
rows_A = []
for fusion, results in all_results.items():
    row = {'Method': fusion}
    row['Cls Acc']    = fmt(*agg(results, 'cls_acc'))
    row['Mis Detect'] = fmt(*agg(results, 'mis_detect_aupr'))
    for n in ood_names:
        row[f'OOD ({n})'] = fmt(*agg(results, f'aupr_{n}'))
    rows_A.append(row)

df_A = pd.DataFrame(rows_A)
path_A = out(f'dempster_{EXP_DATASET}_{args.exp}_table_A.csv')
df_A.to_csv(path_A, index=False)
print(f"\nTable A (Re-EDL format):")
print(df_A.to_string(index=False))

# Table B — Deep Ensemble format
rows_B = []
for fusion, results in all_results.items():
    row = {'Method': fusion}
    for k, label in [('cls_acc','Cls Acc'), ('nll','NLL'), ('brier','Brier'),
                     ('ece','ECE'), ('aurc','AURC')]:
        row[label] = fmt(*agg(results, k))
    for n in ood_names:
        row[f'AUROC ({n})'] = fmt(*agg(results, f'auroc_{n}'))
        row[f'FPR95 ({n})'] = fmt(*agg(results, f'fpr95_{n}'))
    rows_B.append(row)

df_B = pd.DataFrame(rows_B)
path_B = out(f'dempster_{EXP_DATASET}_{args.exp}_table_B.csv')
df_B.to_csv(path_B, index=False)
print(f"\nTable B (Deep Ensemble format):")
print(df_B.to_string(index=False))

# Table C — Efficiency + Diversity
rows_C = []
for fusion, results in all_results.items():
    r0  = results[0]['metrics']
    div = results[-1]['diversity']
    row = {
        'Method':    fusion,
        'Params(M)': f"{r0['param_count']/1e6:.2f}",
        'Train(s)':  f"{r0['train_time_s']:.0f}",
        'Inf(ms)':   f"{r0['inference_ms']:.3f}",
        'Final Δb':  f"{div['delta_b']:.5f}",
        'Final Var': f"{div['var_P']:.5f}",
    }
    if fusion == 'selective':
        row['Mean heads'] = fmt(*agg(results, 'mean_heads_fused'))
    rows_C.append(row)

df_C = pd.DataFrame(rows_C)
path_C = out(f'dempster_{EXP_DATASET}_{args.exp}_efficiency.csv')
df_C.to_csv(path_C, index=False)
print(f"\nTable C (Efficiency + Diversity):")
print(df_C.to_string(index=False))

# Save pickle
pkl = out(f'dempster_{EXP_DATASET}_{args.exp}_results.pkl')
with open(pkl, 'wb') as f:
    pickle.dump({
        fusion: [{'metrics': r['metrics'], 'diversity': r['diversity'],
                  'div_history': r['div_history']}
                 for r in results]
        for fusion, results in all_results.items()
    }, f)

print(f"\n✅  All outputs saved to {OUTPUT_DIR}")
