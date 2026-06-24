"""Regime-conditional portfolio stress test (application, simulated DGP).

The use-case: an investor holds a fixed portfolio w and wants to know how its risk
behaves in a volatility regime that is too rare to estimate from data. A generative
model can answer this: conditioned on the regime, it produces a *distribution* of
plausible covariances, hence a distribution of portfolio risk w'Σw. A point
estimator (Ledoit-Wolf, sample covariance) gives only a single number and cannot.

For a fixed w and each trailing-vol regime (low/mid/high tertile) we compare the
distribution of annualised portfolio volatility sqrt(w'Σw):
  * True  -- the true H_t in that regime (DGP ground truth; only available because
             the data is simulated, which is the whole point).
  * SGCM  -- conditional model, sampled at that regime's conditioning value.
  * SGM   -- unconditional model: one regime-blind distribution, the SAME for every
             regime, so it cannot track the regime.
  * LW    -- rolling Ledoit-Wolf at the regime's validation indices (a point per
             date; implicitly regime-adaptive through its trailing window).

Metrics per regime: median and 95th-percentile (stress) portfolio vol, and the
Wasserstein-1 distance of each estimator's risk distribution to True. The high/low
risk ratio is reported separately as a scale-robust summary (it cancels the model's
overall scale bias, isolating the regime structure).

Reuses the trained simulated checkpoints from the part-1 run -- no new training.
"""

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
from src.evaluation.evaluate import logcov_to_covariance, rolling_lw_covariances
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
N_PER_REGIME = 2000                          # SGCM draws per regime bin
N_POOL = 3000                                # SGM unconditional pool
GEN_CHUNK = 1024                             # sampler batch cap (GPU memory)
GEN_EPS_T, SDE_STEPS = 1e-3, 1000
Q_STRESS = 0.95


def _ckpt_path(seed, cond_dim):
    ctag = "_cond" if cond_dim > 0 else "_uncond"
    return ROOT / "checkpoints" / f"logcov_score_sim_cov{ctag}_seed{seed}.pt"


# =============================================================================
# Portfolio risk + distribution metrics (pure, testable without a model)
# =============================================================================
def portfolio_vol(Sigma, w):
    """Annualised portfolio volatility sqrt(w'Σw) for a covariance batch.
    Sigma: (B, N, N) tensor; w: (N,) tensor -> (B,) numpy array."""
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
        "q95_vol":    float(np.quantile(vols, Q_STRESS)),
        "w1_to_true": float(wasserstein_distance(vols, true_vols)),
    }


# =============================================================================
# Model generation per regime
# =============================================================================
def load_score_model(ckpt_path, n_assets, device):
    """Build LogCovScoreGNN from a checkpoint's saved cfg and install its EMA weights
    (the weights used for sampling in training). Returns (model, cfg)."""
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
    if ema is not None:                                  # swap in the EMA shadow
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in ema:
                    p.copy_(ema[n].to(device))
    model.eval()
    return model, cfg


def _sample_cov(model, sde, cond_norm, n_assets, n_total, norm, device, eps_t):
    """Draw `n_total` covariances at a FIXED cond (or None = unconditional), in
    chunks; returns a CPU tensor (n_total, N, N)."""
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
    """Draw one covariance per (varying) normalised cond row in cond_batch
    (n, cond_dim), so the regime's own conditioning spread enters the risk
    distribution (not just the model's stochasticity at a single cond). CPU tensor."""
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

    cond_samples: list (len = #regimes) of (N_PER_REGIME, cond_dim) NORMALISED cond
      batches sampled from each bin's empirical cond distribution, or None for the
      unconditional model. Conditional: draw one Σ per cond row (the bin's full
      conditioning spread). Unconditional: draw one N_POOL pool, reuse every bin
      (regime-blind). Returns list of (n,) vol arrays, one per regime."""
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


# =============================================================================
# Runner
# =============================================================================
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

    # True distribution per regime (the ground-truth target) + LW per regime
    returns = np.load(ROOT / "data" / "sim_returns.npy")
    lw_cov, keep = rolling_lw_covariances(returns, val_ds.idx.numpy(), LW_WINDOW)
    cond_lw = cond_raw[np.isin(val_ds.idx.numpy(), keep)]    # regime of each LW estimate
    lw_masks = regime_masks(cond_raw)                        # same edges, applied to kept set
    lw_edges = np.quantile(cond_raw, np.linspace(0, 1, len(BIN_LABELS) + 1))
    lw_edges[0], lw_edges[-1] = -np.inf, np.inf

    true_vols, lw_vols = [], []
    for k in range(len(BIN_LABELS)):
        true_vols.append(portfolio_vol(Cov_val[torch.from_numpy(masks[k])], w))
        lwm = (cond_lw >= lw_edges[k]) & (cond_lw < lw_edges[k + 1])
        lw_vols.append(portfolio_vol(lw_cov[torch.from_numpy(lwm)], w))

    # sample each bin's empirical normalised cond distribution for conditional generation
    g = torch.Generator().manual_seed(0)
    cond_samples_by_bin = []
    for m in masks:
        pool = cond_norm[torch.from_numpy(m)]                # (n_bin, cond_dim)
        sel = torch.randint(len(pool), (N_PER_REGIME,), generator=g)
        cond_samples_by_bin.append(pool[sel])               # (N_PER_REGIME, cond_dim)

    # assemble per-estimator, per-regime metrics
    agg = {label: {} for label in ("True", "SGM", "SGCM", "LW")}
    for k, lbl in enumerate(BIN_LABELS):
        agg["True"][lbl] = {"median_vol": (float(np.median(true_vols[k])), 0.0),
                            "q95_vol": (float(np.quantile(true_vols[k], Q_STRESS)), 0.0),
                            "w1_to_true": (0.0, 0.0)}
        agg["LW"][lbl] = {m: (v, 0.0) for m, v in dist_metrics(lw_vols[k], true_vols[k]).items()}

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
_METRICS = [("q95_vol", "stress vol (q95, ann)"),
            ("median_vol", "median vol (ann)"),
            ("w1_to_true", "W1 to true (vol)")]


def _write_outputs(agg):
    cols = [c for c in ("True", "SGM", "SGCM", "LW") if any(agg[c].values())]
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
    print("\nhigh/low stress-vol ratio (scale-robust)")
    for c in cols:
        hi = agg[c].get("hi", {}).get("q95_vol")
        lo = agg[c].get("lo", {}).get("q95_vol")
        if hi and lo and lo[0] > 0:
            print(f"  {c:>6s}: {hi[0] / lo[0]:.3f}")


if __name__ == "__main__":
    main()
