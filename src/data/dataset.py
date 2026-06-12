"""Log-covariance baseline dataset.

Loads empirical correlation/covariance matrices, takes their matrix logarithm
S = logm(Sigma) = Q diag(log lambda) Q^T (a single symmetric matrix obtained via
the eigendecomposition), normalizes per-entry on the train split, and does a
temporal train/val split with an optional gap so the two never share underlying
observations. Optionally loads and normalizes a per-window conditioning variable
(e.g. trailing market volatility); when `cond_path` is None the setup is fully
unconditional.

Working in the matrix-log (log-Euclidean) space makes the target unconstrained
(any symmetric matrix is a valid logm), permutation-equivariant, and linear in
the eigenvalue/scale direction. The eigendecomposition is pure preprocessing, so
no gradient ever passes through it.
"""

import torch
from torch.utils.data import Dataset


class MatrixDataset(Dataset):
    def __init__(self, X, cond=None):
        self.X = X
        self.cond = cond

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.cond is None:
            return self.X[idx]
        return self.X[idx], self.cond[idx]


class SymNormalizer:
    """Per-entry stats for a symmetric matrix. Symmetry is preserved by the
    affine transform because the per-entry mean/std are themselves symmetric.
    Optionally also tracks per-entry cond stats."""

    def __init__(self, X, cond=None):
        self.mean = X.mean(dim=0)                     # (N, N)
        self.std = X.std(dim=0).clamp_min(1e-6)       # (N, N)
        if cond is not None:
            self.cond_mean = cond.mean(dim=0)
            self.cond_std = cond.std(dim=0).clamp_min(1e-6)
        else:
            self.cond_mean = None
            self.cond_std = None

    def to(self, device):
        for n in ("mean", "std"):
            setattr(self, n, getattr(self, n).to(device))
        if self.cond_mean is not None:
            self.cond_mean = self.cond_mean.to(device)
            self.cond_std = self.cond_std.to(device)
        return self

    def normalize(self, X):
        return (X - self.mean) / self.std

    def denormalize(self, X):
        return X * self.std + self.mean

    def normalize_cond(self, cond):
        if self.cond_mean is None:
            raise RuntimeError("SymNormalizer was constructed without cond stats")
        return (cond - self.cond_mean) / self.cond_std

    def denormalize_cond(self, cond):
        if self.cond_mean is None:
            raise RuntimeError("SymNormalizer was constructed without cond stats")
        return cond * self.cond_std + self.cond_mean


def _logm(M, eig_floor=1e-8):
    """Symmetric matrix logarithm via eigendecomposition: logm(Q diag(w) Q^T)
    = Q diag(log w) Q^T. The eigenvector ambiguity cancels on reassembly, so the
    result is a smooth single-valued function of M even at repeated eigenvalues."""
    w, Q = torch.linalg.eigh(M)
    log_w = w.clamp_min(eig_floor).log()
    return (Q * log_w.unsqueeze(-2)) @ Q.transpose(-1, -2)


def _split_indices(T, val_frac, gap, split, n_val_blocks):
    """Train/val index arrays for either split type.

    "contiguous": train = first (1-val_frac), val = last val_frac, one `gap` seam.
        On a nonstationary path this puts train and val in different regimes.
    "blocked": scatter `n_val_blocks` equal val blocks evenly across [0, T) and
        embargo `gap` steps on each side of every block (removed from train). Both
        train and val then span the whole timeline, so their marginals match -- the
        correct setup for a distributional-fidelity claim -- while the embargo keeps
        every val sample >= gap from any train sample, so val still detects
        memorisation on the autocorrelated path.

    Returns (train_idx, val_idx) as 1-D long tensors."""
    if split == "contiguous":
        n_val = int(T * val_frac)
        n_train = T - n_val - gap
        train_idx = torch.arange(n_train)
        val_idx = torch.arange(n_train + gap, T)
        return train_idx, val_idx

    if split == "blocked":
        seg = T / n_val_blocks
        vbl = max(1, int(T * val_frac / n_val_blocks))     # length of each val block
        val_mask = torch.zeros(T, dtype=torch.bool)
        embargo = torch.zeros(T, dtype=torch.bool)          # val blocks + their gap halos
        for i in range(n_val_blocks):
            center = int((i + 0.5) * seg)
            start = max(0, center - vbl // 2)
            end = min(T, start + vbl)
            val_mask[start:end] = True
            embargo[max(0, start - gap):min(T, end + gap)] = True
        train_idx = torch.nonzero(~embargo, as_tuple=False).squeeze(-1)
        val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze(-1)
        # feasibility: each block costs ~(vbl + 2*gap); when n_val_blocks*(vbl+2*gap)
        # approaches T the embargo zones merge and starve train. Fail loudly instead
        # of returning a near-empty (or empty) split.
        if len(train_idx) < len(val_idx) or len(val_idx) < n_val_blocks:
            raise ValueError(
                f"blocked split starved: train={len(train_idx)}, val={len(val_idx)} "
                f"(T={T}, n_val_blocks={n_val_blocks}, gap={gap}, val_block_len={vbl}). "
                f"Embargo zones ~n_val_blocks*(vbl+2*gap)={n_val_blocks*(vbl+2*gap)} "
                f"crowd out T={T}; reduce gap or n_val_blocks (need gap << T/(2*n_val_blocks)).")
        return train_idx, val_idx

    raise ValueError(f"split must be 'contiguous' or 'blocked', got {split!r}")


def make_logcov_datasets(C_path, val_frac=0.2, ridge=1e-3, gap=0, cond_path=None,
                         target="correlation", split="contiguous", n_val_blocks=10):
    """Load empirical matrices, take the matrix log, normalize, split. Empirical
    matrices with T_obs ~ N are often near-rank-deficient; a small ridge keeps
    every eigenvalue strictly positive so the matrix log is finite. `target`
    controls the shrinkage target:
      "correlation" -> ridge toward identity (preserves unit diagonal)
      "covariance"  -> ridge toward own diagonal (preserves per-asset scale)
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
    S = _logm(M)                                       # (T, N, N) symmetric

    cond = None
    if cond_path is not None:
        cond = torch.load(cond_path, weights_only=True).float()
        assert len(cond) == len(S), f"cond length {len(cond)} != S length {len(S)}"

    T = S.shape[0]
    train_idx, val_idx = _split_indices(T, val_frac, gap, split, n_val_blocks)
    S_tr, S_va = S[train_idx], S[val_idx]
    cond_tr = cond[train_idx] if cond is not None else None
    cond_va = cond[val_idx] if cond is not None else None

    norm = SymNormalizer(S_tr, cond=cond_tr)
    train_ds = MatrixDataset(
        norm.normalize(S_tr),
        cond=(norm.normalize_cond(cond_tr) if cond_tr is not None else None),
    )
    val_ds = MatrixDataset(
        norm.normalize(S_va),
        cond=(norm.normalize_cond(cond_va) if cond_va is not None else None),
    )
    # expose the source-tensor indices so the eval can recover the matching raw
    # covariances/correlations (a tail-slice no longer works once val is scattered).
    train_ds.idx = train_idx
    val_ds.idx = val_idx
    return train_ds, val_ds, norm
