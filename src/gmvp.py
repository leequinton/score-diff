"""Global Minimum Variance Portfolio (GMVP) downstream application.

A self-contained downstream evaluation: feed each covariance estimator into the
*same* portfolio rule and measure realised out-of-sample risk. This tests whether
the score-based generative covariances are *useful*, not merely distributionally
faithful (the stylized-fact tables in src/train_baseline.py).

Estimators compared (one column each):
  * True   -- GMVP from the realised FORWARD covariance (perfect foresight). The
              infeasible lower bound on realised risk, not a usable strategy.
  * SGM    -- posterior-mean covariance of the unconditional score model. One
              static Sigma_hat (the model has no regime input), reused every date.
  * SGCM   -- posterior-mean covariance of the conditional score model, drawn per
              OOS date conditioned on that date's macro-vol regime.
  * Sample -- rolling trailing sample covariance (the classic GMVP baseline).
  * LW     -- rolling trailing Ledoit-Wolf shrinkage covariance.

GMVP rule (unconstrained, shorts allowed): w = Sigma^-1 1 / (1' Sigma^-1 1).

Realised risk is measured on FORWARD actual returns (truly out-of-sample): for an
estimation window ending at raw-return row e, the weights are scored against the
realised sample covariance of the next non-overlapping window returns[e : e+W].
Using same-window data would trivially favour the sample covariance (the in-sample
optimism that LW shrinkage exists to cure), so estimation and realisation windows
never overlap.

Alignment contract (produced by src/data/empirical_prep.py):
  data/processed/idx_daily.pt -- LongTensor (T_mat,), the raw-return row index
  marking the END (exclusive) of each matrix's estimation window, plus the window
  length W is recorded in the same file's metadata (see `load_alignment`).

SGM/SGCM columns are aggregated mean +- std across training seeds; the
deterministic Sample/LW/True columns are single-valued.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.dataset import make_logcov_datasets
from src.diffusion.sde import VPSDE
from src.diffusion.solver import sample_logcov
from src.evaluation.evaluate import (
    logcov_to_covariance, rolling_sample_covariances, rolling_lw_covariances,
)
from src.models.logcov_gnn import LogCovScoreGNN

ROOT = Path(__file__).resolve().parents[1]

# --- empirical data source (this application is empirical-only; see plan) -------
SUFFIX       = "_daily"
COR_FILE     = "C_empirical_daily"
COV_FILE     = "Cov_empirical_daily"
COND_FILE    = "cond_daily"
IDX_FILE     = "idx_daily"                 # raw-return end-row per matrix (+ W metadata)
RETURNS_CSV  = ROOT / "data" / "raw" / "49_Industry_Portfolios_Daily.csv"

# val split must match training (src/train_baseline.py empirical settings).
GAP          = 12
SPLIT        = "blocked"
N_VAL_BLOCKS = 10

ANN = math.sqrt(252.0)                      # daily -> annual vol scaling

# generation budget: matrices drawn per OOS date to form the posterior-mean
# Sigma_hat (SGCM), and total draws for the single static SGM estimate.
K_PER_DATE   = 64
K_STATIC     = 3000
GEN_CHUNK    = 1024                          # sampler batch size cap (GPU memory)

SEEDS = (0, 1, 2, 3, 4)


# =============================================================================
# GMVP core
# =============================================================================
def gmvp_weights(Sigma):
    """Unconstrained global minimum variance weights for a batch of covariances.
    w = Sigma^-1 1 / (1' Sigma^-1 1), solved (not inverted) per matrix. Weights
    sum to 1 and may be negative (shorts). Sigma: (B, N, N) SPD -> (B, N)."""
    N = Sigma.shape[-1]
    ones = torch.ones(N, dtype=Sigma.dtype, device=Sigma.device)
    x = torch.linalg.solve(Sigma, ones.expand(Sigma.shape[0], N).unsqueeze(-1)).squeeze(-1)
    return x / x.sum(dim=-1, keepdim=True)


def _quad(w, Sigma):
    """Per-sample quadratic form w' Sigma w. w: (B,N), Sigma: (B,N,N) -> (B,)."""
    return torch.einsum("bi,bij,bj->b", w, Sigma, w)


def portfolio_diagnostics(w):
    """Concentration / leverage of a weight batch (B, N), averaged over the batch.
      max_weight    -- mean of max_i |w_i|        (single-name concentration)
      gross_leverage-- mean of sum_i |w_i|        (1.0 == fully long, >1 == net shorting)
      eff_N         -- mean of 1 / sum_i w_i^2    (effective number of positions)
      short_frac    -- mean fraction of negative weights
    """
    aw = w.abs()
    return {
        "max_weight":     float(aw.max(dim=-1).values.mean()),
        "gross_leverage": float(aw.sum(dim=-1).mean()),
        "eff_N":          float((1.0 / (w * w).sum(dim=-1)).mean()),
        "short_frac":     float((w < 0).float().mean()),
    }


def gmvp_metrics(Sigma_hat, Sigma_fwd):
    """Score one estimator: GMVP from Sigma_hat, realised against forward cov.
    Sigma_hat: (M,N,N) estimation covariances; Sigma_fwd: (M,N,N) realised
    forward sample covariances (same M, aligned). Returns a metrics dict."""
    w = gmvp_weights(Sigma_hat)
    pred_var = _quad(w, Sigma_hat).clamp_min(0.0)
    real_var = _quad(w, Sigma_fwd).clamp_min(0.0)
    realized_vol = float(real_var.mean().sqrt()) * ANN
    predicted_vol = float(pred_var.mean().sqrt()) * ANN
    out = {
        "realized_vol": realized_vol,
        "predicted_vol": predicted_vol,
        # >1 : realised risk exceeds the ex-ante estimate (optimism); ~1 : calibrated
        "calib_ratio": realized_vol / predicted_vol if predicted_vol > 0 else float("nan"),
    }
    out.update(portfolio_diagnostics(w))
    return out


# =============================================================================
# Generative-model covariances (posterior mean)
# =============================================================================
def load_score_model(ckpt_path, n_assets, device):
    """Build LogCovScoreGNN from a checkpoint's saved cfg and install its EMA
    weights (the weights used for sampling in training). Returns (model, cfg)."""
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
    """Draw `n_total` covariance matrices (cond fixed within a call) in chunks and
    return them as a CPU tensor (n_total, N, N). cond_norm: (cond_dim,) normalized
    regime, or None for the unconditional model."""
    out = []
    drawn = 0
    while drawn < n_total:
        b = min(GEN_CHUNK, n_total - drawn)
        cond = None
        if cond_norm is not None:
            cond = cond_norm.to(device).unsqueeze(0).expand(b, -1).contiguous()
        S_norm = sample_logcov(model, sde, b, n_assets, device=device,
                               eps_t=eps_t, cond=cond, guidance_scale=0.0)
        S = norm.denormalize(S_norm)
        out.append(logcov_to_covariance(S).cpu())
        drawn += b
    return torch.cat(out, dim=0)


def model_covariances(ckpt_path, cond_norm, n_assets, norm, device,
                      k_per_date=K_PER_DATE, k_static=K_STATIC, eps_t=1e-3,
                      sde_steps=1000):
    """Posterior-mean Sigma_hat per OOS date for one trained model.

    cond_norm: (M, cond_dim) normalized regime per OOS date, or None.
      * conditional (cond_dim>0): draw k_per_date samples conditioned on each date's
        regime, average -> Sigma_hat_t  (M, N, N), time-varying.
      * unconditional (cond_dim==0 / cond_norm None): draw k_static samples once,
        average -> a single static Sigma_hat broadcast to all M dates.
    """
    sde = VPSDE(N=sde_steps)
    model, cfg = load_score_model(ckpt_path, n_assets, device)
    use_cond = cfg.get("cond_dim", 0) > 0 and cond_norm is not None

    if not use_cond:
        Sig = _sample_cov(model, sde, None, n_assets, k_static, norm, device, eps_t)
        Sigma_static = Sig.mean(dim=0, keepdim=True)             # (1, N, N)
        M = cond_norm.shape[0] if cond_norm is not None else 1
        return Sigma_static.expand(M, n_assets, n_assets).contiguous()

    mats = []
    for t in range(cond_norm.shape[0]):
        Sig = _sample_cov(model, sde, cond_norm[t], n_assets, k_per_date,
                          norm, device, eps_t)
        mats.append(Sig.mean(dim=0))                             # (N, N)
    return torch.stack(mats, dim=0)                              # (M, N, N)


# =============================================================================
# Forward-return alignment + rolling baselines
# =============================================================================
def load_alignment():
    """Per-matrix raw-return end-row index + window length W from idx_daily.pt
    (written by src/data/empirical_prep.py). Supports either a plain LongTensor
    (T_mat,) of end rows with W inferred, or a dict {'end': (T_mat,), 'window': W}."""
    obj = torch.load(ROOT / "data" / "processed" / f"{IDX_FILE}.pt", weights_only=False)
    if isinstance(obj, dict):
        end = torch.as_tensor(obj["end"]).long()
        W = int(obj["window"])
    else:
        end = torch.as_tensor(obj).long()
        W = int((end[1:] - end[:-1]).median())              # stride as a W fallback
    return end, W


def load_returns():
    """Raw daily FF returns aligned to the matrix construction. Imported lazily so
    the GMVP core stays usable without the data-prep dependencies."""
    from data.loaders import load_daily
    df = load_daily(RETURNS_CSV)
    return df.to_numpy(dtype=np.float64)


# =============================================================================
# Runner
# =============================================================================
def aggregate_runs(runs):
    """list of metric dicts -> {metric: (mean, std)} over the shared keys."""
    if not runs:
        return {}
    keys = set(runs[0]).intersection(*[set(r) for r in runs[1:]])
    return {k: (float(np.mean([r[k] for r in runs])),
               float(np.std([r[k] for r in runs]))) for k in sorted(keys)}


def _ckpt_path(seed, cond_dim):
    ctag = "_cond" if cond_dim > 0 else "_uncond"
    return ROOT / "checkpoints" / f"logcov_score{SUFFIX}_cov{ctag}_seed{seed}.pt"


def _fmt_cell(d):
    return f"{'':^16s}" if d is None else f"{d[0]:8.4f} +- {d[1]:6.4f}"


def _print_table(head_cols, rows):
    print(f"{'metric':>16s} | " + " | ".join(f"{c:^16s}" for c in head_cols))
    for label, cells in rows:
        print(f"{label:>16s} | " + " | ".join(_fmt_cell(c) for c in cells))


def _write_csv(path, head_cols, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [f"{c}_mean" for c in head_cols]
                   + [f"{c}_std" for c in head_cols])
        for label, cells in rows:
            w.writerow([label]
                       + ["" if c is None else f"{c[0]:.6f}" for c in cells]
                       + ["" if c is None else f"{c[1]:.6f}" for c in cells])


METRIC_ROWS = [
    ("realized_vol",   "realised vol (ann)"),
    ("predicted_vol",  "predicted vol (ann)"),
    ("calib_ratio",    "calib (real/pred)"),
    ("max_weight",     "max |weight|"),
    ("gross_leverage", "gross leverage"),
    ("eff_N",          "effective N"),
    ("short_frac",     "short fraction"),
]


def _plot(agg, cols, out_path):
    """Realised-vol bar chart (mean +- std) + ex-ante vs ex-post calibration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rv_m = [agg[c]["realized_vol"][0] for c in cols]
    rv_s = [agg[c]["realized_vol"][1] for c in cols]
    ax1.bar(cols, rv_m, yerr=rv_s, capsize=4, color="steelblue")
    ax1.set_ylabel("annualised realised volatility")
    ax1.set_title("GMVP out-of-sample risk (lower = better)")

    pv = [agg[c]["predicted_vol"][0] for c in cols]
    ax2.scatter(pv, rv_m, color="darkorange")
    for c, x, y in zip(cols, pv, rv_m):
        ax2.annotate(c, (x, y), textcoords="offset points", xytext=(5, 5))
    lim = [0, max(max(pv), max(rv_m)) * 1.1]
    ax2.plot(lim, lim, "k--", alpha=0.5, label="calibrated (y=x)")
    ax2.set_xlim(lim); ax2.set_ylim(lim)
    ax2.set_xlabel("ex-ante predicted vol"); ax2.set_ylabel("ex-post realised vol")
    ax2.set_title("Calibration"); ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main(seeds=SEEDS, k_per_date=K_PER_DATE, k_static=K_STATIC):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cov_path  = ROOT / "data" / "processed" / f"{COV_FILE}.pt"
    cor_path  = ROOT / "data" / "processed" / f"{COR_FILE}.pt"
    cond_path = ROOT / "data" / "processed" / f"{COND_FILE}.pt"

    # val split + normalizer, identical to training (empirical, covariance target).
    _, val_ds, norm = make_logcov_datasets(
        cov_path, gap=GAP, cond_path=cond_path, target="covariance",
        split=SPLIT, n_val_blocks=N_VAL_BLOCKS,
    )
    norm = norm.to(device)
    val_idx = val_ds.idx                                 # indices into the matrix series
    n_assets = val_ds.X.shape[-1]
    cond_norm = val_ds.cond.to(device)                  # (M, cond_dim) normalized
    print(f"val OOS dates: {len(val_idx)}  N={n_assets}  cond_dim={cond_norm.shape[-1]}")

    # --- forward-return alignment: estimation end-row e_t and forward cov S_fwd ---
    end, W = load_alignment()
    returns = load_returns()
    end_val = end[val_idx].numpy()                       # raw-return end rows per OOS date
    # realised FORWARD sample covariance: trailing window of length W ending at e+W,
    # i.e. returns[e : e+W] -- strictly after the estimation window (no overlap).
    fwd_end = end_val + W
    fwd_ok = fwd_end <= len(returns)                     # drop dates without a full forward window
    if not fwd_ok.all():
        print(f"  dropping {int((~fwd_ok).sum())} OOS date(s) without a full forward window")
    keep = np.nonzero(fwd_ok)[0]
    Sigma_fwd, _ = rolling_sample_covariances(returns, fwd_end[keep], W)
    Sigma_fwd = Sigma_fwd.to(device)                    # (M', N, N)

    # estimation-side rolling baselines: trailing window ending at e (causal).
    Sig_samp, _ = rolling_sample_covariances(returns, end_val[keep], W)
    Sig_lw, _   = rolling_lw_covariances(returns, end_val[keep], W)
    cond_keep = cond_norm[torch.from_numpy(keep).to(device)]

    agg = {}
    # --- True foresight oracle: GMVP from the realised forward cov itself --------
    agg["True"] = {k: (v, 0.0) for k, v in gmvp_metrics(Sigma_fwd, Sigma_fwd).items()}
    # --- deterministic rolling baselines ---------------------------------------
    agg["Sample"] = {k: (v, 0.0) for k, v in gmvp_metrics(Sig_samp.to(device), Sigma_fwd).items()}
    agg["LW"]     = {k: (v, 0.0) for k, v in gmvp_metrics(Sig_lw.to(device), Sigma_fwd).items()}

    # --- generative models: posterior-mean Sigma_hat, mean +- std over seeds ----
    for label, cond_dim in (("SGM", 0), ("SGCM", cond_norm.shape[-1])):
        runs = []
        for seed in seeds:
            ckpt = _ckpt_path(seed, cond_dim)
            if not ckpt.exists():
                print(f"  [{label}] missing checkpoint {ckpt.name}; skipping seed {seed}")
                continue
            cn = cond_keep if cond_dim > 0 else None
            Sigma_hat = model_covariances(ckpt, cn, n_assets, norm, device,
                                          k_per_date=k_per_date, k_static=k_static)
            runs.append(gmvp_metrics(Sigma_hat.to(device), Sigma_fwd))
            print(f"  [{label}] seed {seed}: realised vol "
                  f"{runs[-1]['realized_vol']:.4f} (ann)")
        if runs:
            agg[label] = aggregate_runs(runs)

    cols = [c for c in ("True", "SGM", "SGCM", "Sample", "LW") if c in agg]
    rows = [(label, [agg[c].get(key) for c in cols]) for key, label in METRIC_ROWS]
    out_csv = ROOT / "results" / f"gmvp{SUFFIX}.csv"
    out_png = ROOT / "results" / f"gmvp{SUFFIX}.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(out_csv, cols, rows)
    _print_table(cols, rows)
    _plot(agg, cols, out_png)
    print(f"\nsaved table -> {out_csv}\nsaved plot  -> {out_png}")
    return agg


if __name__ == "__main__":
    main()
