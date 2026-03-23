# Dempster-EDL

Uncertainty estimation via Dirichlet opinion fusion using Josang's Cumulative
Belief Fusion (CBF) — the same theoretical framework underlying Re-EDL.

One shared backbone. N λ heads applied post-hoc to the same logits. Two fusion
methods compared: standard CBF and selective CBF with fallback.

---

## Core idea

Re-EDL maps logits to a Dirichlet opinion via:
```
e = softplus(logits)   α = e + λ   P = α/S   u = K·λ/S
```

Different λ values produce analytically distinct opinions from the same logits.
We fuse N such opinions using Josang's CBF rule:
```
harmony  = u_A + u_B - u_A·u_B
b_fused  = (b_A·u_B + b_B·u_A) / harmony
u_fused  = (u_A·u_B) / harmony
```

Uncertain heads are automatically downweighted. No learnable parameters.

---

## Two fusion methods

### Original CBF
Standard iterative pairwise CBF across all N heads.
Self-weighting by vacuity. Treats all inputs the same.

### Selective CBF - Inference Time
Fuse only heads agreeing with the majority prediction.
- On easy ID inputs: all heads agree → full fusion
- On hard ID inputs: partial agreement → partial fusion → higher uncertainty
- On OOD inputs: no consensus → **Fallback B**: fuse top-K high-λ (cautious)
  heads only → high u_fused → strong OOD signal

**Fallback B rationale:** high-λ heads are conservative by design. When heads
completely disagree (OOD-like), deferring to cautious heads gives maximum
uncertainty signal rather than averaging conflicting evidence.

### Input-adaptive CBF
Instead of straight CBF, scale the vacuity before fusion based on input features:
python# Learn a small temperature network
τ_i = temperature_net(logits)   # (N, 11) — per-sample per-head temperature
u_i_scaled = u_i * τ_i          # sharpen or flatten each head's uncertainty

**Then apply CBF with scaled vacuities**
The temperature network is tiny (logits→11 outputs) and learns to amplify the vacuity of heads that are historically unreliable for this type of input. This preserves the theoretical CBF structure while adding input-dependence.

---

## λ head configuration

Log-uniform spacing in [0.1, 3.5], 7 heads (configurable):
```
λ = [0.10, 0.17, 0.29, 0.50, 0.85, 1.46, 3.50]
```
Three operating zones from Exp 1 characterisation:
- [0.1, 0.3]  — accuracy zone
- [0.5, 1.0]  — best OOD zone
- [1.5, 3.5]  — uncertainty/mis-detect zone

---

## File map

```
dempster_edl/
├── main.py                  ← entry point
├── README.md
├── config/config.py         ← all hyperparameters
├── models/                  ← VGG16-BN, ResNet-18
├── architecture/
│   └── dempster.py          ← DempsterEDL, _cbf_pair, _cbf_all, _cbf_selective
├── edl_core/                ← compute_opinion, re_edl_mse_loss
├── data/loaders.py          ← CIFAR-10, CIFAR-100, LT, TinyImageNet
├── evaluation/              ← all metrics (NLL, Brier, ECE, FPR95, etc.)
└── experiment/
    ├── runner.py            ← training loop + evaluation
    └── ckpt.py              ← checkpoint save/load (per fusion, per run)
```

---

## How to run

```bash
python main.py                          # both methods, 5 runs
python main.py --fusion original        # original CBF only
python main.py --fusion selective       # selective CBF only
python main.py --runs 1                 # smoke test
python main.py --exp cifar100           # CIFAR-100 as ID, ResNet-18
python main.py --exp lt_mild            # long-tailed ρ=0.1
python main.py --exp lt_heavy           # long-tailed ρ=0.01
```

---

## Config (`config/config.py`)

| Parameter | Default | Description |
|---|---|---|
| `ID_DATASET` | `'cifar10'` | `'cifar10'` \| `'cifar100'` |
| `NUM_EPOCHS` | `200` | 10=smoke, 200=paper |
| `NUM_RUNS` | `5` | runs for mean±std |
| `NUM_HEADS` | `7` | number of λ heads (range 6-10) |
| `LAMBDA_MIN` | `0.1` | smallest λ value |
| `LAMBDA_MAX` | `3.5` | largest λ value |
| `FUSION_METHODS` | `['original','selective']` | methods to run |
| `SELECTIVE_FALLBACK_K` | `3` | fallback heads in selective mode |
| `TRACK_DIVERSITY_EVERY` | `5` | diversity logging frequency |

---

## Checkpoints

```
outputs/checkpoints/
  original/run0/  best.pt  latest.pt
  original/run1/  ...
  selective/run0/ best.pt  latest.pt
  selective/run1/ ...
```

Delete checkpoint folders to force full re-run.

---

## Paper reference

Chen et al., Re-EDL, TPAMI 2025. arXiv:2410.00393
Josang, Subjective Logic, Springer 2016 (CBF: Chapter 12)
