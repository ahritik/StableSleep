"""Common utilities: seeding, device selection, dataset & augmentation, samplers,
scheduler (warmup+cosine), checkpoint helpers, and focal loss.
"""
import os
import math
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# --------------------- Repro & device ---------------------
def set_seed(seed: int = 1337):
    """Seed numpy/torch RNGs for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    """Pick CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# --------------------- Checkpoint IO ---------------------
def save_checkpoint(state: Dict, path: str):
    """Save model/epoch/etc. to a file; ensures parent dir exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, map_location=None) -> Dict:
    """Load a torch checkpoint (state dict + extra metadata)."""
    return torch.load(path, map_location=map_location)

# --------------------- Meters ---------------------
class AverageMeter:
    """Track streaming averages of scalar values (e.g., loss)."""
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0
    def update(self, val, n=1): self.sum += float(val) * n; self.count += n
    @property
    def avg(self): return self.sum / max(1, self.count)

# --------------------- Dataset & Augment ---------------------
class SleepDataset(Dataset):
    """Tensor-ready dataset for (epoched) EEG.
    Expects x: (N, C, T) float32 and y: (N,) int64.
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: Dict=None):
        self.x = x
        self.y = y.astype(np.int64)
        self.augment = augment or {}

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        sig = self.x[idx].copy()  # copy so we don't destroy cache
        lab = self.y[idx]
        if self.augment.get("enabled", False):
            sig = self.apply_aug(sig, self.augment)
        return torch.from_numpy(sig), torch.tensor(lab)

    @staticmethod
    def apply_aug(x, cfg):
        """Very light augmentations suitable for EEG (preserve morphology)."""
        C, T = x.shape
        # 1) Additive Gaussian jitter
        std = cfg.get("jitter_std", 0.0)
        if std > 0:
            x = x + np.random.normal(0, std, size=x.shape).astype(np.float32)
        # 2) Global scaling (gain)
        smin, smax = cfg.get("scale_min", 1.0), cfg.get("scale_max", 1.0)
        if smax > 0 and (smin != 1.0 or smax != 1.0):
            scale = np.random.uniform(smin, smax)
            x = (x * scale).astype(np.float32)
        # 3) Short time masking (drop a contiguous window)
        prob = cfg.get("time_mask_prob", 0.0)
        L = int(cfg.get("time_mask_len", 0))
        if prob > 0 and L > 0 and np.random.rand() < prob and L < T:
            start = np.random.randint(0, T - L)
            x[:, start:start+L] = 0.0
        return x

def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Return a sampler that draws classes with equal probability.
    Helpful to counter class imbalance (e.g., rare N1/REM).
    """
    classes, counts = np.unique(labels, return_counts=True)
    freq = {c: counts[i] for i, c in enumerate(classes)}
    weights = np.array([1.0 / freq[int(y)] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(labels), replacement=True)

# --------------------- LR schedule ---------------------
class WarmupCosine(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for a few epochs, then cosine decay to zero."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup factor in [0,1]
            warm = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * warm for base_lr in self.base_lrs]
        # Cosine from 1 → 0 across remaining epochs
        progress = (self.last_epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * coeff for base_lr in self.base_lrs]

# --------------------- Losses ---------------------
class FocalLoss(torch.nn.Module):
    """Focal loss = (1 - p_t)^gamma * CE. Emphasizes hard/rare examples.
    Supports class weights and label smoothing.
    """
    def __init__(self, gamma=2.0, weight=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)              # p_t = exp(-CE)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum": return loss.sum()
        return loss
