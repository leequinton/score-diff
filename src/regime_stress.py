"""Regime-conditional portfolio stress test (application, simulated DGP).

For a fixed portfolio w and each trailing-vol regime (low/mid/high tertile), compare
the distribution of annualised portfolio vol sqrt(w'Σw) across estimators: True (DGP
H_t), SGCM (conditional model), SGM (unconditional, regime-blind), Sample and LW
(rolling trailing covariances). Point estimators give one number per date; the
generative models give a whole distribution, which is the point.

Metrics per regime: median, 95th/99th-percentile (stress) vol, and W1 to True. The
high/low ratio is a scale-robust summary. Reuses the part-1 checkpoints, no training."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from src.data.dataset import make_logcov_datasets
from src.diffusion.sde import VPSDE
from src.diffusion.solver import sample_logcov
from src.evaluation.evaluate import (
    logcov_to_covariance, rolling_lw_covariances, rolling_sample_covariances,
)
from src.models.logcov_gnn import LogCovScoreGNN

ROOT = Path(__file__).resolve().parents[1]

# simulated data + split, identical to train_baseline.py sim settings
COV_FILE  = "Cov_sim"
COND_FILE = "cond_sim"
GAP, SPLIT, N_VAL_BLOCKS = 500, "blocked", 10
LW_WINDOW = 252
ANN = math.sqrt(252.0)                       # daily -> annual vol (FF returns in %/day)
BIN_LABELS = ("lo", "mid", "hi")

SEEDS = (0, 1, 2, 3, 4)
N_PER_REGIME = 3000                          # SGCM draws per regime bin
N_POOL = 3000                                # SGM unconditional pool
GEN_CHUNK = 1024                             # sampler batch cap (GPU memory)
GEN_EPS_T, SDE_STEPS = 1e-3, 1000
Q95, Q99 = 0.95, 0.99                        # 5% / 1% tail stress levels (vol units)


def _ckpt_path(seed, cond_dim):
    ctag = "_cond" if cond_dim > 0 else "_uncond"
    return ROOT / "checkpoints" / f"logcov_score_sim_cov{ctag}_seed{seed}.pt"


# --- Portfolio risk + distribution metrics -----------------------------------
def portfolio_vol(Sigma, w):
    """Annualised portfolio volatility sqrt(w'Σw). Sigma: (B, N, N); w: (N,) -> (B,)."""
    var = torch.einsum("i,bij,j->b", w, Sigma, w).clamp_min(0.0)
    return (var.sqrt().numpy()) * ANN


def regime_masks(cond_raw):
    """Boolean masks for the low/mid/high tertiles of the raw regime signal."""
    edges = np.quantile(cond_raw, np.linspace(0, 1, len(BIN_LABELS) + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return [(cond_raw >= edges[k]) & (cond_raw < edges[k + 1])
            for k in range(len(BIN_LABELS))]


def dist_metrics(vols, true_vols):
    """Distribution summary of a portfolio-vol sample vs the True distribution."""
    return {
        "median_vol": float(np.median(vols)),
        "q95_vol":    float(np.quantile(vols, Q95)),
        "q99_vol":    float(np.quantile(vols, Q99)),
        "w1_to_true": float(wasserstein_distance(vols, true_vols)),
    }


# --- Model generation per regime ----------------------------------------------
def load_score_model(ckpt_path, n_assets, device):
    """Build LogCovScoreGNN from a checkpoint's cfg and install its EMA weights."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = LogCovScoreGNN(
        n_assets=n_assets,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg.get("dropout", 0.0),
        cond_dim=cfg.get("cond_dim", 0),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    ema = ckpt.get("ema")
    if ema is not None:                                  # swap in EMA shadow
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    p.copy_(ema[n].to(device))
    model.eval()
    return model, cfg


def _sample_cov(model, sde, cond_norm, n_assets, n_total, norm, device, eps_t):
    """Draw `n_total` covariances at a fixed cond (or None), chunked. CPU tensor."""
    out, drawn = [], 0
    while drawn < n_total:
        b = min(GEN_CHUNK, n_total - drawn)
        cond = None
        if cond_norm is not None:
            cond = cond_norm.to(device).unsqueeze(0).expand(b, -1).contiguous()
        S_norm = sample_logcov(model, sde, b, n_assets, device=device,
                               eps_t=eps_t, cond=cond, guidance_scale=0.0)
        out.append(logcov_to_covariance(norm.denormalize(S_norm)).cpu())
        drawn += b
    return torch.cat(out, dim=0)


def _sample_cov_varcond(model, sde, cond_batch, n_assets, norm, device, eps_t):
    """Draw one covariance per normalised cond row in cond_batch (n, cond_dim), so
    the regime's own conditioning spread enters the risk distribution. CPU tensor."""
    out, drawn, n_total = [], 0, cond_batch.shape[0]
    while drawn < n_total:
        b = min(GEN_CHUNK, n_total - drawn)
        cond = cond_batch[drawn:drawn + b].to(device)
        S_norm = sample_logcov(model, sde, b, n_assets, device=device,
                               eps_t=eps_t, cond=cond, guidance_scale=0.0)
        out.append(logcov_to_covariance(norm.denormalize(S_norm)).cpu())
        drawn += b
    return torch.cat(out, dim=0)


def model_regime_vols(ckpt_path, w, cond_samples, n_assets, norm, device):
    """Per-regime annualised portfolio-vol samples for one trained model.
    cond_samples: per-regime normalised cond batches, or None (unconditional). The
    conditional model draws one Σ per cond row; the unconditional model draws one
    pool reused for every bin. Returns a list of (n,) vol arrays, one per regime."""
    sde = VPSDE(N=SDE_STEPS)
    model, cfg = load_score_model(ckpt_path, n_assets, device)
    use_cond = cfg.get("cond_dim", 0) > 0 and cond_samples is not None

    if not use_cond:
        Sig = _sample_cov(model, sde, None, n_assets, N_POOL, norm, device, GEN_EPS_T)
        pool = portfolio_vol(Sig, w)
        return [pool for _ in BIN_LABELS]                    # same pool every bin

    return [portfolio_vol(_sample_cov_varcond(model, sde, cond_b, n_assets, norm,
                                              device, GEN_EPS_T), w)
            for cond_b in cond_samples]


# --- Runner --------------------------------------------------------------------
def aggregate_runs(runs):
    """list of metric dicts -> {metric: (mean, std)} over shared keys."""
    if not runs:
        return {}
    keys = set(runs[0]).intersection(*[set(r) for r in runs[1:]])
    return {k: (float(np.mean([r[k] for r in runs])),
               float(np.std([r[k] for r in runs]))) for k in sorted(keys)}


def main(seeds=SEEDS):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    cov_path  = ROOT / "data" / "processed" / f"{COV_FILE}.pt"
    cond_path = ROOT / "data" / "processed" / f"{COND_FILE}.pt"

    _, val_ds, norm = make_logcov_datasets(
        cov_path, gap=GAP, cond_path=cond_path, target="covariance",
        split=SPLIT, n_val_blocks=N_VAL_BLOCKS,
    )
    norm = norm.to(device)
    n_assets = val_ds.X.shape[-1]
    w = torch.full((n_assets,), 1.0 / n_assets)              # equal-weight portfolio

    # true H_t and raw regime signal on the validation set
    Cov_val = torch.load(cov_path, weights_only=True).float()[val_ds.idx]
    cond_raw = torch.load(cond_path, weights_only=True).float().reshape(-1)[val_ds.idx].numpy()
    cond_norm = val_ds.cond                                   # normalised, (M, cond_dim)
    masks = regime_masks(cond_raw)
    print(f"val OOS dates: {len(cond_raw)}  N={n_assets}  "
          f"bin sizes: {[int(m.sum()) for m in masks]}")

    # True distribution per regime (ground truth) + rolling sample/LW baselines
    returns = np.load(ROOT / "data" / "sim_returns.npy")
    samp_cov, keep = rolling_sample_covariances(returns, val_ds.idx.numpy(), LW_WINDOW)
    lw_cov, _      = rolling_lw_covariances(returns, val_ds.idx.numpy(), LW_WINDOW)
    cond_roll = cond_raw[np.isin(val_ds.idx.numpy(), keep)]  # regime of each rolling estimate
    edges = np.quantile(cond_raw, np.linspace(0, 1, len(BIN_LABELS) + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    true_vols, samp_vols, lw_vols = [], [], []
    for k in range(len(BIN_LABELS)):
        true_vols.append(portfolio_vol(Cov_val[torch.from_numpy(masks[k])], w))
        rm = (cond_roll >= edges[k]) & (cond_roll < edges[k + 1])
        samp_vols.append(portfolio_vol(samp_cov[torch.from_numpy(rm)], w))
        lw_vols.append(portfolio_vol(lw_cov[torch.from_numpy(rm)], w))

    # sample each bin's empirical normalised cond distribution for conditional generation
    g = torch.Generator().manual_seed(0)
    cond_samples_by_bin = []
    for m in masks:
        pool = cond_norm[torch.from_numpy(m)]                # (n_bin, cond_dim)
        sel = torch.randint(len(pool), (N_PER_REGIME,), generator=g)
        cond_samples_by_bin.append(pool[sel])               # (N_PER_REGIME, cond_dim)

    # assemble per-estimator, per-regime metrics
    agg = {label: {} for label in ("True", "SGM", "SGCM", "Sample", "LW")}
    for k, lbl in enumerate(BIN_LABELS):
        agg["True"][lbl] = {"median_vol": (float(np.median(true_vols[k])), 0.0),
                            "q95_vol": (float(np.quantile(true_vols[k], Q95)), 0.0),
                            "q99_vol": (float(np.quantile(true_vols[k], Q99)), 0.0),
                            "w1_to_true": (0.0, 0.0)}
        agg["Sample"][lbl] = {m: (v, 0.0) for m, v in dist_metrics(samp_vols[k], true_vols[k]).items()}
        agg["LW"][lbl]     = {m: (v, 0.0) for m, v in dist_metrics(lw_vols[k], true_vols[k]).items()}

    for label, cond_dim in (("SGM", 0), ("SGCM", cond_norm.shape[-1])):
        per_seed = {lbl: [] for lbl in BIN_LABELS}
        for seed in seeds:
            ckpt = _ckpt_path(seed, cond_dim)
            if not ckpt.exists():
                print(f"  [{label}] missing {ckpt.name}; skipping seed {seed}")
                continue
            cbb = cond_samples_by_bin if cond_dim > 0 else None
            vols = model_regime_vols(ckpt, w, cbb, n_assets, norm, device)
            for k, lbl in enumerate(BIN_LABELS):
                per_seed[lbl].append(dist_metrics(vols[k], true_vols[k]))
            print(f"  [{label}] seed {seed} done")
        for lbl in BIN_LABELS:
            if per_seed[lbl]:
                agg[label][lbl] = aggregate_runs(per_seed[lbl])

    _write_outputs(agg)
    return agg


# metrics x (estimator) printed/saved per regime
_METRICS = [("q99_vol", "stress vol (q99, ann)"),
            ("q95_vol", "stress vol (q95, ann)"),
            ("median_vol", "median vol (ann)"),
            ("w1_to_true", "W1 to true (vol)")]


def _write_outputs(agg):
    cols = [c for c in ("True", "SGM", "SGCM", "Sample", "LW") if any(agg[c].values())]
    out = ROOT / "results" / "regime_stress_sim.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["regime", "metric"] + [f"{c}_mean" for c in cols] + [f"{c}_std" for c in cols])
        for lbl in BIN_LABELS:
            for key, _ in _METRICS:
                cells = [agg[c].get(lbl, {}).get(key) for c in cols]
                w.writerow([lbl, key]
                           + ["" if x is None else f"{x[0]:.6f}" for x in cells]
                           + ["" if x is None else f"{x[1]:.6f}" for x in cells])
    print(f"\nsaved -> {out}")
    for key, name in _METRICS:
        print(f"\n{name}")
        print(f"{'regime':>8s} | " + " | ".join(f"{c:^16s}" for c in cols))
        for lbl in BIN_LABELS:
            cells = [agg[c].get(lbl, {}).get(key) for c in cols]
            print(f"{lbl:>8s} | " + " | ".join(
                f"{'--':^16s}" if x is None else f"{x[0]:8.4f} ± {x[1]:6.4f}" for x in cells))

    # scale-robust summary: high/low stress-vol ratio per estimator
    for key in ("q95_vol", "q99_vol"):
        print(f"\nhigh/low {key} ratio (scale-robust)")
        for c in cols:
            hi = agg[c].get("hi", {}).get(key)
            lo = agg[c].get("lo", {}).get(key)
            if hi and lo and lo[0] > 0:
                print(f"  {c:>6s}: {hi[0] / lo[0]:.3f}")


if __name__ == "__main__":
    main()
