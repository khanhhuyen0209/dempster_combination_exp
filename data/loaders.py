# =============================================================================
# data/loaders.py
# =============================================================================

import os, urllib.request, zipfile
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN, ImageFolder

from config import DATA_DIR, BATCH_SIZE

_PIN = torch.cuda.is_available()


# ── TinyImageNet auto-download ────────────────────────────────────────────────

def _ensure_tinyimagenet():
    tin_root = os.path.join(DATA_DIR, 'tiny-imagenet-200')
    tin_val  = os.path.join(tin_root, 'val', 'images')
    zip_path = os.path.join(DATA_DIR, 'tiny-imagenet-200.zip')

    if os.path.exists(os.path.join(tin_root, 'val', 'n01443537')):
        return tin_root

    if not os.path.exists(zip_path):
        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        print(f"  Downloading TinyImageNet (~235 MB) ...")
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path,
            reporthook=lambda b, bs, t: print(
                f"\r  {min(b*bs,t)/1e6:.1f}/{t/1e6:.1f} MB", end=''))
        print()

    if not os.path.exists(tin_root):
        print("  Extracting TinyImageNet ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)

    annotations = os.path.join(tin_root, 'val', 'val_annotations.txt')
    if os.path.exists(annotations) and os.path.exists(tin_val):
        print("  Fixing TinyImageNet val/ structure ...")
        with open(annotations) as f:
            for line in f:
                parts = line.strip().split('\t')
                fname, cls = parts[0], parts[1]
                cls_dir = os.path.join(tin_root, 'val', cls)
                os.makedirs(cls_dir, exist_ok=True)
                src = os.path.join(tin_val, fname)
                dst = os.path.join(cls_dir, fname)
                if os.path.exists(src):
                    os.rename(src, dst)
        try: os.rmdir(tin_val)
        except OSError: pass
        print("  TinyImageNet val/ fixed ✓")

    return tin_root


# ── Long-tailed subset ────────────────────────────────────────────────────────

def _make_lt_subset(dataset, imbalance_ratio: float):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    K       = len(classes)
    n_max   = int(np.bincount(targets).max())
    selected = []
    for k in classes:
        n_k     = int(n_max * (imbalance_ratio ** (k / (K - 1))))
        n_k     = max(n_k, 1)
        indices = np.where(targets == k)[0]
        np.random.shuffle(indices)
        selected.extend(indices[:n_k].tolist())
    print(f"  LT subset: {len(selected)} samples (ρ={imbalance_ratio})")
    return Subset(dataset, selected)


# ── Public API ────────────────────────────────────────────────────────────────

def get_dataloaders(dataset: str = None,
                    imbalance_ratio: float = None) -> tuple:
    """
    Returns (dataloaders, num_classes, in_channels).
    dataset: 'mnist' | 'cifar10' | 'cifar100' | 'cifar10_lt'
    """
    from config import ID_DATASET
    ds = dataset if dataset is not None else ID_DATASET

    if ds == 'mnist':
        return _mnist_loaders()
    elif ds == 'cifar10':
        return _cifar10_loaders()
    elif ds == 'cifar100':
        return _cifar100_id_loaders()
    elif ds == 'cifar10_lt':
        ratio = imbalance_ratio if imbalance_ratio is not None else 0.01
        return _cifar10_lt_loaders(ratio)
    else:
        raise ValueError(f"Unknown dataset: {ds!r}. "
                         f"Choose from 'mnist','cifar10','cifar100','cifar10_lt'.")


def get_hard_val_loaders(dataset: str = None) -> dict:
    """Hard-ID augmented val sets — MNIST only."""
    from config import ID_DATASET
    ds = dataset if dataset is not None else ID_DATASET
    if ds != 'mnist':
        raise ValueError(f"Hard-ID only defined for MNIST, got {ds!r}.")

    def _tf(aug):
        return transforms.Compose([
            transforms.Resize((32, 32)), aug,
            transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
        ])
    return {
        'val_rot45': DataLoader(
            MNIST(DATA_DIR, False, download=True,
                  transform=_tf(transforms.Lambda(lambda img: TF.rotate(img, 45)))),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        'val_rot90': DataLoader(
            MNIST(DATA_DIR, False, download=True,
                  transform=_tf(transforms.Lambda(lambda img: TF.rotate(img, 90)))),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        'val_blur': DataLoader(
            MNIST(DATA_DIR, False, download=True,
                  transform=_tf(transforms.GaussianBlur(kernel_size=5, sigma=2.0))),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }


# ── Private builders ──────────────────────────────────────────────────────────

def _mnist_loaders():
    tf = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tf_sv = transforms.Compose([
        transforms.Resize((32, 32)), transforms.Grayscale(1),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    return {
        'train': DataLoader(MNIST(DATA_DIR, True,  download=True, transform=tf),
                            batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=_PIN),
        'val':   DataLoader(MNIST(DATA_DIR, False, download=True, transform=tf),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=_PIN),
        'ood': {'SVHN': DataLoader(
            SVHN(DATA_DIR, split='test', download=True, transform=tf_sv),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)},
    }, 10, 1


def _cifar10_loaders():
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_sv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
    ])
    return {
        'train': DataLoader(CIFAR10(DATA_DIR, True,  download=True, transform=tf_tr),
                            batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=_PIN),
        'val':   DataLoader(CIFAR10(DATA_DIR, False, download=True, transform=tf_te),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=_PIN),
        'ood': {
            'SVHN':     DataLoader(SVHN(DATA_DIR, 'test', download=True, transform=tf_sv),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
            'CIFAR100': DataLoader(CIFAR100(DATA_DIR, False, download=True, transform=tf_te),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        },
    }, 10, 3


def _cifar10_lt_loaders(imbalance_ratio: float = 0.01):
    """Long-tailed CIFAR-10. Train imbalanced, val balanced, OOD same as classical."""
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    tf_sv = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
    ])
    full_train = CIFAR10(DATA_DIR, True, download=True, transform=tf_tr)
    lt_subset  = _make_lt_subset(full_train, imbalance_ratio)
    return {
        'train': DataLoader(lt_subset,
                            batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=_PIN),
        'val':   DataLoader(CIFAR10(DATA_DIR, False, download=True, transform=tf_te),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=_PIN),
        'ood': {
            'SVHN':     DataLoader(SVHN(DATA_DIR, 'test', download=True, transform=tf_sv),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
            'CIFAR100': DataLoader(CIFAR100(DATA_DIR, False, download=True, transform=tf_te),
                                   batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
        },
    }, 10, 3


def _cifar100_id_loaders():
    """CIFAR-100 as ID + SVHN + TinyImageNet OOD (auto-downloaded)."""
    tf_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    tf_sv = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.198, 0.201, 0.197)),
    ])
    tf_tin = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    ])

    tin_root = _ensure_tinyimagenet()
    tin_val  = os.path.join(tin_root, 'val')

    ood_dict = {
        'SVHN': DataLoader(SVHN(DATA_DIR, 'test', download=True, transform=tf_sv),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    if os.path.exists(tin_val):
        ood_dict['TinyImageNet'] = DataLoader(
            ImageFolder(tin_val, transform=tf_tin),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        print("  TinyImageNet OOD loaded ✓")

    return {
        'train': DataLoader(CIFAR100(DATA_DIR, True,  download=True, transform=tf_tr),
                            batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=_PIN),
        'val':   DataLoader(CIFAR100(DATA_DIR, False, download=True, transform=tf_te),
                            batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=_PIN),
        'ood': ood_dict,
    }, 100, 3


# ── CIFAR-10-C ────────────────────────────────────────────────────────────────

CIFAR10C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    'speckle_noise', 'gaussian_blur', 'spatter', 'saturate',
]   # 19 corruption types as used in F-EDL paper (arXiv:2510.18322)


def _ensure_cifar10c():
    """
    Auto-download CIFAR-10-C from Zenodo (~2.5GB) if not present.
    Saves to DATA_DIR/CIFAR-10-C/.
    """
    cifar10c_dir = os.path.join(DATA_DIR, 'CIFAR-10-C')
    test_file    = os.path.join(cifar10c_dir, 'labels.npy')

    if os.path.exists(test_file):
        return cifar10c_dir

    os.makedirs(cifar10c_dir, exist_ok=True)
    base_url = 'https://zenodo.org/record/2535967/files'

    # Download labels first
    print("  Downloading CIFAR-10-C labels ...")
    urllib.request.urlretrieve(
        f"{base_url}/labels.npy",
        os.path.join(cifar10c_dir, 'labels.npy'))

    # Download each corruption file
    for i, corruption in enumerate(CIFAR10C_CORRUPTIONS):
        dst = os.path.join(cifar10c_dir, f"{corruption}.npy")
        if os.path.exists(dst):
            continue
        print(f"  Downloading {corruption} ({i+1}/{len(CIFAR10C_CORRUPTIONS)}) ...")
        urllib.request.urlretrieve(f"{base_url}/{corruption}.npy", dst)

    print("  CIFAR-10-C downloaded ✓")
    return cifar10c_dir


def get_cifar10c_loaders(severities=None) -> dict:
    """
    Build CIFAR-10-C DataLoaders for distribution shift evaluation.

    Returns
    -------
    dict: {severity: DataLoader} where severity ∈ {1,2,3,4,5}
    Each loader contains all 19 corruption types concatenated at that severity.
    One severity level = 19 × 10,000 = 190,000 samples.

    Usage: AUPR computed per severity, averaged across corruption types
    following F-EDL paper (arXiv:2510.18322) setting.
    """
    import numpy as _np
    from torch.utils.data import TensorDataset

    if severities is None:
        severities = [1, 2, 3, 4, 5]

    cifar10c_dir = _ensure_cifar10c()
    labels_all   = _np.load(os.path.join(cifar10c_dir, 'labels.npy'))

    tf = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    loaders = {}
    for sev in severities:
        # Each .npy file has shape (50000, 32, 32, 3) — 5 severities stacked
        # Severity s occupies indices [(s-1)*10000 : s*10000]
        idx_start = (sev - 1) * 10000
        idx_end   = sev * 10000

        images_sev = []
        labels_sev = []

        for corruption in CIFAR10C_CORRUPTIONS:
            path = os.path.join(cifar10c_dir, f"{corruption}.npy")
            if not os.path.exists(path):
                print(f"  ⚠️  {corruption}.npy not found, skipping")
                continue
            imgs = _np.load(path)[idx_start:idx_end]   # (10000, 32, 32, 3)
            images_sev.append(imgs)
            labels_sev.append(labels_all[idx_start:idx_end])

        # Stack all corruptions: (19*10000, 32, 32, 3)
        images_np = _np.concatenate(images_sev, axis=0)
        labels_np = _np.concatenate(labels_sev, axis=0)

        # Convert to tensor and normalise
        imgs_t = torch.from_numpy(images_np).permute(0, 3, 1, 2).float() / 255.0
        # Apply normalisation per-sample
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
        std  = torch.tensor([0.247,  0.243,  0.261 ]).view(1,3,1,1)
        imgs_t = (imgs_t - mean) / std

        labs_t = torch.from_numpy(labels_np).long()
        ds     = TensorDataset(imgs_t, labs_t)

        loaders[sev] = DataLoader(ds, batch_size=BATCH_SIZE,
                                   shuffle=False, num_workers=2,
                                   pin_memory=_PIN)
        print(f"  CIFAR-10-C severity {sev}: {len(ds)} samples "
              f"({len(CIFAR10C_CORRUPTIONS)} corruptions)")

    return loaders
