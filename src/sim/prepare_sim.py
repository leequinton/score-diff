"""Convert the simulated DCC covariance path into the .pt tensors that
make_logcov_datasets consumes.

`dcc.py` writes data/sim_cov.npy = the conditional covariance path H_t (T, N, N).
Here we turn it into the same on-disk format as the empirical pipeline:

    data/processed/Cov_sim.pt  - covariances H_t                 (float32, T x N x N)
    data/processed/C_sim.pt    - correlations D^-1/2 H_t D^-1/2  (float32, T x N x N)

so `train_baseline.py` can point at the simulated data exactly like the empirical
data. Unlike the empirical case there is no exogenous conditioning series, so the
simulated runs are unconditional (no cond_sim.pt is produced).
"""

import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "sim_cov.npy"
OUT = ROOT / "data" / "processed"


def to_correlation(Sigma):
    """D^-1/2 Sigma D^-1/2 from a batch of covariance matrices."""
    d = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    return (Sigma / (d.unsqueeze(-1) * d.unsqueeze(-2))).clamp(-1.0, 1.0)


def main():
    H = torch.from_numpy(np.load(RAW)).float()          # (T, N, N) covariance
    # The simulator builds H = D R D, symmetric up to fp drift; enforce it exactly
    # so logm/eigh see a genuinely symmetric input.
    H = 0.5 * (H + H.transpose(-1, -2))
    C = to_correlation(H)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(H, OUT / "Cov_sim.pt")
    torch.save(C, OUT / "C_sim.pt")

    eigmin = torch.linalg.eigvalsh(H[:: max(1, len(H) // 256)]).min().item()
    print(f"loaded {tuple(H.shape)} covariances from {RAW.name}")
    print(f"  min eigenvalue (subsampled): {eigmin:.3e}  (must be > 0 for logm)")
    print(f"  Cov diag mean: {H.diagonal(dim1=-2, dim2=-1).mean():.4f}  "
          f"Corr diag mean: {C.diagonal(dim1=-2, dim2=-1).mean():.4f}")
    print(f"saved -> {OUT/'Cov_sim.pt'}")
    print(f"saved -> {OUT/'C_sim.pt'}")


if __name__ == "__main__":
    main()
