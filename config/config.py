# =============================================================================
# config/config.py — Dempster-EDL
# -----------------------------------------------------------------------------
# Single source of truth. Only file you need to edit.
# =============================================================================

import os

# ── Dataset ───────────────────────────────────────────────────────────────────
ID_DATASET = 'cifar10'   # 'cifar10' | 'cifar100'

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 200
NUM_RUNS   = 5
BATCH_SIZE = 64
SEED       = 42

# ── λ head configuration ──────────────────────────────────────────────────────
# Log-uniform spacing informed by Exp 1 characterisation study.
# Range [0.1, 3.5] covers three distinct operating zones:
#   [0.1, 0.3]  — accuracy zone
#   [0.5, 1.0]  — best OOD zone
#   [1.5, 3.5]  — uncertainty/mis-detect zone
NUM_HEADS  = 7
LAMBDA_MIN = 0.1
LAMBDA_MAX = 3.5

# ── Fusion methods ────────────────────────────────────────────────────────────
# 'original'  — Josang CBF across all heads
# 'selective' — CBF on agreeing heads; fallback to high-λ heads on OOD
# 'adaptive'  — CBF with input-adaptive temperature scaling (Direction 1)
FUSION_METHODS       = ['original', 'selective', 'adaptive']
SELECTIVE_FALLBACK_K = 3   # high-λ fallback heads for selective mode

# ── CIFAR-10-C ────────────────────────────────────────────────────────────────
# 19 corruption types × 5 severity levels
# Auto-downloaded from https://zenodo.org/record/2535967
CIFAR10C_SEVERITIES = [1, 2, 3, 4, 5]

# ── Diversity tracking ────────────────────────────────────────────────────────
TRACK_DIVERSITY_EVERY = 5

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = './data'
OUTPUT_DIR = './outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def out(filename: str) -> str:
    return os.path.join(OUTPUT_DIR, filename)
