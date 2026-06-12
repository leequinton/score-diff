"""Convert the simulated DCC covariance path into the .pt tensors that
make_logcov_datasets consumes.

`dcc.py` writes data/sim_cov.npy = the conditional covariance path H_t (T, N, N).
Here we turn it into the same on-disk format as the empirical pipeline:

    data/processed/Cov_sim.pt  - covariances H_t                 (float32, T x N x N)
    data/processed/C_sim.pt    - correlations D^-1/2 H_t D^-1/2  (float32, T x N x N)
    data/processed/cond_sim.pt - causal trailing market vol       (float32, T x 1)

so `train_baseline.py` can point at the simulated data exactly like the empirical
data. The conditioning series is a strictly-lagged trailing realized volatility of
the equal-weight market return (built from sim_returns.npy): cond[t] depends only
on returns up to t-1, so it never sees H_t and is a legitimate regime signal rather
than leakage. Because the DGP is GARCH, trailing vol is predictive of the current
conditional scale -- exactly the regime information an unconditional model lacks.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "sim_cov.npy"
RET = ROOT / "data" / "sim_returns.npy"
OUT = ROOT / "data" / "processed"

# trailing window (steps) for the realized-vol conditioning signal
COND_WINDOW = 21


def to_correlation(Sigma):
    """D^-1/2 Sigma D^-1/2 from a batch of covariance matrices."""
    d = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    return (Sigma / (d.unsqueeze(-1) * d.unsqueeze(-2))).clamp(-1.0, 1.0)


def trailing_market_vol(returns, window):
    """Strictly-causal trailing realized vol of the equal-weight market return.

    returns: (T, N) simulated returns. cond[t] = std of the market return over the
    `window` steps ending at t-1 (the .shift(1) excludes the contemporaneous r_t, so
    cond[t] is a function of information strictly before H_t -- no leakage of the
    target's scale). Leading entries, where the window is not yet full, are
    back-filled with the first computable value (initialises an input feature only).
    Returns a (T, 1) float32 tensor aligned 1:1 with H_t."""
    r_mkt = pd.Series(returns.mean(axis=1))                 # equal-weight market return
    vol = r_mkt.shift(1).rolling(window, min_periods=2).std()
    vol = vol.bfill().to_numpy().copy()                    # .copy() -> writable tensor
    return torch.from_numpy(vol).float().unsqueeze(-1)      # (T, 1)


def main():
    H = torch.from_numpy(np.load(RAW)).float()          # (T, N, N) covariance
    # The simulator builds H = D R D, symmetric up to fp drift; enforce it exactly
    # so logm/eigh see a genuinely symmetric input.
    H = 0.5 * (H + H.transpose(-1, -2))
    C = to_correlation(H)

    returns = np.load(RET)                              # (T, N) aligned with H
    assert len(returns) == len(H), f"returns {len(returns)} != H {len(H)}"
    cond = trailing_market_vol(returns, COND_WINDOW)    # (T, 1) causal regime signal

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(H, OUT / "Cov_sim.pt")
    torch.save(C, OUT / "C_sim.pt")
    torch.save(cond, OUT / "cond_sim.pt")

    eigmin = torch.linalg.eigvalsh(H[:: max(1, len(H) // 256)]).min().item()
    print(f"loaded {tuple(H.shape)} covariances from {RAW.name}")
    print(f"  min eigenvalue (subsampled): {eigmin:.3e}  (must be > 0 for logm)")
    print(f"  Cov diag mean: {H.diagonal(dim1=-2, dim2=-1).mean():.4f}  "
          f"Corr diag mean: {C.diagonal(dim1=-2, dim2=-1).mean():.4f}")
    print(f"  cond (trailing vol, W={COND_WINDOW}): "
          f"mean {cond.mean():.4f}  std {cond.std():.4f}  "
          f"range [{cond.min():.4f}, {cond.max():.4f}]")
    print(f"saved -> {OUT/'Cov_sim.pt'}")
    print(f"saved -> {OUT/'C_sim.pt'}")
    print(f"saved -> {OUT/'cond_sim.pt'}")


if __name__ == "__main__":
    main()
