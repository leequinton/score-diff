"""Full-Cholesky baseline dataset.

Loads empirical correlation/covariance matrices, takes their Cholesky factor,
normalizes per-entry on the train split, and does a temporal train/val split
with an optional gap so the two never share underlying observations. Optionally
loads and normalizes a per-window conditioning variable (e.g. trailing market
volatility); when `cond_path` is None the setup is fully unconditional.
"""

import torch
from torch.utils.data import Dataset


class CholDataset(Dataset):
    def __init__(self, X, cond=None):
        self.X = X
        self.cond = cond

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.cond is None:
            return self.X[idx]
        return self.X[idx], self.cond[idx]


class CholNormalizer:
    """Per-entry stats for a lower-triangular matrix; upper-tri kept at zero.
    Optionally also tracks per-entry cond stats."""

    def __init__(self, X, cond=None):
        N = X.shape[-1]
        self.mean = X.mean(dim=0)                     # (N, N)
        self.std = X.std(dim=0).clamp_min(1e-6)       # (N, N)
        self.tril = torch.tril(torch.ones(N, N))
        if cond is not None:
            self.cond_mean = cond.mean(dim=0)
            self.cond_std = cond.std(dim=0).clamp_min(1e-6)
        else:
            self.cond_mean = None
            self.cond_std = None

    def to(self, device):
        for n in ("mean", "std", "tril"):
            setattr(self, n, getattr(self, n).to(device))
        if self.cond_mean is not None:
            self.cond_mean = self.cond_mean.to(device)
            self.cond_std = self.cond_std.to(device)
        return self

    def normalize(self, X):
        return ((X - self.mean) / self.std) * self.tril

    def denormalize(self, X):
        return (X * self.std + self.mean) * self.tril

    def normalize_cond(self, cond):
        if self.cond_mean is None:
            raise RuntimeError("CholNormalizer was constructed without cond stats")
        return (cond - self.cond_mean) / self.cond_std

    def denormalize_cond(self, cond):
        if self.cond_mean is None:
            raise RuntimeError("CholNormalizer was constructed without cond stats")
        return cond * self.cond_std + self.cond_mean


def make_full_chol_datasets(C_path, val_frac=0.2, ridge=1e-3, gap=0, cond_path=None,
                            target="correlation"):
    """Load empirical matrices, take Cholesky, normalize, split. Empirical matrices
    with T_obs ~ N are often near-rank-deficient; a small ridge keeps the Cholesky
    stable. `target` controls the shrinkage target:
      "correlation" → ridge toward identity (preserves unit diagonal)
      "covariance"  → ridge toward own diagonal (preserves per-asset scale)
    If `cond_path` is provided, a per-window conditioning vector is loaded and
    normalized on the train split."""
    M = torch.load(C_path, weights_only=True).float()  # (T, N, N)
    N = M.shape[-1]
    if target == "correlation":
        M = (1.0 - ridge) * M + ridge * torch.eye(N)
    elif target == "covariance":
        diag = torch.diag_embed(torch.diagonal(M, dim1=-2, dim2=-1))
        M = (1.0 - ridge) * M + ridge * diag
    else:
        raise ValueError(f"target must be 'correlation' or 'covariance', got {target}")
    L = torch.linalg.cholesky(M)                       # (T, N, N)

    cond = None
    if cond_path is not None:
        cond = torch.load(cond_path, weights_only=True).float()
        assert len(cond) == len(L), f"cond length {len(cond)} != L length {len(L)}"

    T = L.shape[0]
    n_val = int(T * val_frac)
    n_train = T - n_val - gap
    val_start = n_train + gap
    L_tr, L_va = L[:n_train], L[val_start:]
    cond_tr = cond[:n_train] if cond is not None else None
    cond_va = cond[val_start:] if cond is not None else None

    norm = CholNormalizer(L_tr, cond=cond_tr)
    train_ds = CholDataset(
        norm.normalize(L_tr),
        cond=(norm.normalize_cond(cond_tr) if cond_tr is not None else None),
    )
    val_ds = CholDataset(
        norm.normalize(L_va),
        cond=(norm.normalize_cond(cond_va) if cond_va is not None else None),
    )
    return train_ds, val_ds, norm
