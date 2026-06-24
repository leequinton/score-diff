"""Compare real vs generated correlation/covariance matrices.

Correlation stylized facts (CorrGAN-style, Marti et al.): mean off-diagonal
correlation, Gini of the eigenvalue spectrum, cophenetic correlation (single +
ward linkage), Perron–Frobenius violation (negative mass of the leading
eigenvector), and the power-law exponent of the eigenvalue distribution — each
scored by Wasserstein-1 against a fixed real (val) reference. Covariance adds a
variance-scale W1. The irreducible floor is no longer a split-half of the val set:
it is the True (DGP oracle) row of the distributional-fidelity table."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, cophenet


def _offdiag(M):
    N = M.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool, device=M.device)
    return M[:, mask]


def to_correlation(Sigma):
    d = torch.diagonal(Sigma, dim1=-2, dim2=-1)
    valid = (d > 1e-6).all(dim=-1)
    S = Sigma[valid]
    d = d[valid].sqrt()
    C = S / (d.unsqueeze(-1) * d.unsqueeze(-2))
    return C.clamp(-1.0, 1.0), (~valid).sum().item()


def logcov_to_covariance(S):
    """Symmetric matrix-log S (N x N) -> covariance Sigma = expm(S) via the
    eigendecomposition: expm(Q diag(w) Q^T) = Q diag(exp w) Q^T. expm maps any
    symmetric matrix to an SPD one, so the reconstruction is unconstrained."""
    w, Q = torch.linalg.eigh(S)
    return (Q * w.exp().unsqueeze(-2)) @ Q.transpose(-1, -2)


def logcov_to_correlation(S):
    """Symmetric matrix-log S (N x N) -> normalized correlation matrices."""
    return to_correlation(logcov_to_covariance(S))


def w1_offdiag(C_real, C_gen):
    """1-D Wasserstein-1 on the pooled distribution of off-diagonal correlations."""
    a = _offdiag(C_real).flatten().numpy()
    b = _offdiag(C_gen).flatten().numpy()
    return float(wasserstein_distance(a, b))


def eig_gini(eigs_desc):
    asc = np.sort(eigs_desc, axis=-1)             
    N = asc.shape[-1]
    ranks = np.arange(1, N + 1)
    total = asc.sum(-1)
    return 2 * (asc * ranks).sum(-1) / (N * total) - (N + 1) / N


def cophenetic_per_sample(C, method="single"):
    """Cophenetic correlation per matrix using the Mantegna distance
    d_ij = sqrt(2(1 - C_ij)) on the condensed upper triangle, with `method`
    linkage ('single' or 'ward'). Higher = stronger hierarchical structure."""
    C_np = C.numpy() if hasattr(C, "numpy") else np.asarray(C)
    N = C_np.shape[-1]
    iu = np.triu_indices(N, k=1)
    out = np.empty(C_np.shape[0])
    for i in range(C_np.shape[0]):
        d = np.sqrt(np.clip(2.0 * (1.0 - C_np[i]), 0.0, 4.0))[iu]
        Z = linkage(d, method=method)
        out[i] = cophenet(Z, d)[0]
    return out


def _leading_eigvec_neg_sum(C_np):
    """Perron–Frobenius violation: -sum of the negative entries of the leading
    eigenvector (sign-normalised so its entries sum positive). ~0 for a 'clean'
    correlation matrix whose top eigenvector is all-positive. Returns (T,) >= 0."""
    out = np.empty(len(C_np))
    for i in range(len(C_np)):
        _, V = np.linalg.eigh(C_np[i])      # ascending -> last column = leading
        v = V[:, -1]
        if v.sum() < 0:
            v = -v                          # eigh sign is arbitrary
        out[i] = -np.minimum(v, 0.0).sum()
    return out


def _powerlaw_alpha(eig_desc):
    """Power-law exponent alpha of the eigenvalue distribution per matrix
    (powerlaw.Fit). eig_desc: (T, N) descending. Returns (T,); NaN when the
    `powerlaw` package is missing, so the rest of the eval still runs."""
    try:
        import powerlaw
    except ImportError:
        return np.full(eig_desc.shape[0], np.nan)
    out = np.empty(eig_desc.shape[0])
    for i in range(eig_desc.shape[0]):
        ev = eig_desc[i][eig_desc[i] > 1e-12]
        try:
            out[i] = powerlaw.Fit(ev, verbose=False).alpha
        except Exception:
            out[i] = np.nan
    return out


def _safe_w1(a, b):
    """Wasserstein-1 that drops NaNs (e.g. powerlaw missing); NaN if a side empties."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def _subsample(C, k, seed=0):
    """At most k rows of C (torch batch), for bounding per-matrix-fact cost."""
    if len(C) <= k:
        return C
    g = torch.Generator().manual_seed(seed)
    return C[torch.randperm(len(C), generator=g)[:k]]


# CorrGAN-style per-matrix stylized facts, computed on <= FACT_SUBSAMPLE matrices
# (powerlaw.Fit is the bottleneck; 1000 is plenty for stable mean/std/W1).
FACT_NAMES = ["gini", "coph_single", "coph_ward", "perron", "powerlaw"]
FACT_SUBSAMPLE = 1000


def _per_matrix_facts(C):
    """Per-matrix stylized facts -> {name: (T,) array} for FACT_NAMES."""
    C_np = C.numpy() if hasattr(C, "numpy") else np.asarray(C)
    eig_desc = np.ascontiguousarray(np.sort(np.linalg.eigvalsh(C_np), axis=-1)[:, ::-1])
    return {
        "gini":        eig_gini(eig_desc),
        "coph_single": cophenetic_per_sample(C_np, "single"),
        "coph_ward":   cophenetic_per_sample(C_np, "ward"),
        "perron":      _leading_eigvec_neg_sum(C_np),
        "powerlaw":    _powerlaw_alpha(eig_desc),
    }


def variance_diagnostics(Sigma_real, Sigma_gen, Sigma_train=None):
    """Variance-space (scale) fidelity — the dimension correlation metrics discard.
    Wasserstein-1 on the pooled distribution of per-asset variances (diagonals of Σ).
    The irreducible floor lives in the True (DGP oracle) row of the distributional-
    fidelity table, so this no longer computes a split-half floor.

    Sigma_real, Sigma_gen: (T, N, N) covariance batches (not correlation).
    Sigma_train (optional): the train-split covariances. When given, the same
    variance stats are reported for train, plus gen-vs-train W1 — to disentangle
    exp-tail reconstruction bias (gen overshoots BOTH train and val) from
    train/val regime shift (gen tracks train; both differ from val)."""
    var_real = torch.diagonal(Sigma_real, dim1=-2, dim2=-1).flatten().numpy()
    var_gen  = torch.diagonal(Sigma_gen,  dim1=-2, dim2=-1).flatten().numpy()

    out = {
        "w1_variance":     float(wasserstein_distance(var_real, var_gen)),
        "var_mean_real":   float(var_real.mean()),
        "var_mean_gen":    float(var_gen.mean()),
        # median is robust to the exp() tail of log-variance reconstruction
        "var_median_real": float(np.median(var_real)),
        "var_median_gen":  float(np.median(var_gen)),
    }

    if Sigma_train is not None:
        var_train = torch.diagonal(Sigma_train, dim1=-2, dim2=-1).flatten().numpy()
        # gen-vs-train W1: if this is ~floor while w1_variance (val) is >>floor,
        # the scale gap is regime shift, not the model.
        out.update({
            "var_mean_train":    float(var_train.mean()),
            "var_median_train":  float(np.median(var_train)),
            "w1_variance_train": float(wasserstein_distance(var_train, var_gen)),
        })

    return out


def _np1d(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x).reshape(-1)


def _corr_offdiag_pooled(Sigma):
    """Pooled off-diagonal correlations from a covariance batch (scale removed)."""
    d = torch.diagonal(Sigma, dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
    C = (Sigma / (d.unsqueeze(-1) * d.unsqueeze(-2))).clamp(-1.0, 1.0)
    return _offdiag(C).flatten().numpy()


def _var_pooled(Sigma):
    return torch.diagonal(Sigma, dim1=-2, dim2=-1).flatten().numpy()


def regime_binned_eval(cond_real, Sigma_real, Sigma_gen, cond_gen=None,
                       bin_labels=("lo", "mid", "hi")):
    """Conditional-fidelity eval. Bin the trailing-vol regime by quantiles of
    `cond_real`; within each bin compare real vs generated correlation (pooled
    off-diagonal) and scale (pooled variance) via W1. This is the test conditioning
    is *for* -- the marginal eval averages over regimes and hides it.

    cond_real: (T,) regime value aligned with Sigma_real (the val set, raw space).
    cond_gen:  (n,) regime value each generated matrix was drawn at (conditional
               model). If None (unconditional), the whole generated set is compared
               to every bin -- a marginal generator cannot match per-regime, which
               is exactly what this surfaces.

    Notes on the comparison being made (state these in the paper):
      * CDM (cond_gen given): bin-vs-bin -- generated-conditioned-on-bin-k vs
        real-in-bin-k. This is conditional fidelity.
      * DM (cond_gen None): full-marginal-vs-bin -- the *entire* generated set vs
        real-in-bin-k. This is the right baseline (a regime-blind model judged
        against each regime), NOT "DM restricted to bin k". The DM per-bin W1 is
        therefore a marginal-vs-conditional distance, by construction >= a model
        that tracks the regime.
      * regime_pooled_*: the no-binning row (all real vs all gen), computed by the
        SAME machinery as the bins so it sits apples-to-apples atop the lo/mid/hi
        rows -- 'overall' vs per-regime.

    No floor is computed here: the split-half floor was dropped in favour of the
    True (DGP oracle) row of the distributional-fidelity table.
    """
    cond_real = _np1d(cond_real)
    edges = np.quantile(cond_real, np.linspace(0, 1, len(bin_labels) + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    cg = _np1d(cond_gen) if cond_gen is not None else None

    def block(Sr_full, Sg_full, pre):
        """One real-vs-gen W1 comparison, shared by the pooled row and every bin."""
        Sr = _subsample(Sr_full, FACT_SUBSAMPLE)
        # degenerate (non-positive diagonal) gen count -- 0 for expm covariances,
        # but surfaced so a high-vol-bin sample-dropping bias would be visible.
        d = torch.diagonal(Sg_full, dim1=-2, dim2=-1)
        n_invalid = int((~(d > 1e-6).all(dim=-1)).sum().item())
        Sg = _subsample(Sg_full, FACT_SUBSAMPLE)
        res = {pre + "n_real": float(len(Sr)), pre + "n_gen": float(len(Sg)),
               pre + "n_invalid_gen": float(n_invalid)}
        if len(Sr) < 4 or len(Sg) < 4:
            res[pre + "w1_offdiag"] = res[pre + "w1_variance"] = float("nan")
            return res

        res[pre + "w1_offdiag"] = float(wasserstein_distance(
            _corr_offdiag_pooled(Sr), _corr_offdiag_pooled(Sg)))
        res[pre + "w1_variance"] = float(wasserstein_distance(
            _var_pooled(Sr), _var_pooled(Sg)))
        return res

    out = {}
    # pooled row: no binning, all real vs all gen, identical machinery to the bins.
    out.update(block(Sigma_real, Sigma_gen, "regime_pooled_"))
    for k, label in enumerate(bin_labels):
        rmask = torch.from_numpy((cond_real >= edges[k]) & (cond_real < edges[k + 1]))
        if cg is None:
            Sg_full = Sigma_gen
        else:
            gmask = torch.from_numpy((cg >= edges[k]) & (cg < edges[k + 1]))
            Sg_full = Sigma_gen[gmask]
        out.update(block(Sigma_real[rmask], Sg_full, f"regime_{label}_"))

    return out


def fact_means(C):
    """Mean stylized-fact VALUES over a correlation batch (for the values table:
    simulated train/val reference rows and the LW row). Off-diagonal correlation
    is the mean over the full pooled off-diagonal; the per-matrix facts are meaned
    over a FACT_SUBSAMPLE-sized subsample (matching eval_and_plot). Returns
    {offdiag_mean, gini_mean, coph_single_mean, coph_ward_mean, perron_mean, powerlaw_mean}."""
    f = _per_matrix_facts(_subsample(C, FACT_SUBSAMPLE))
    out = {"offdiag_mean": float(_offdiag(C).flatten().mean())}
    for n in FACT_NAMES:
        out[f"{n}_mean"] = float(np.nanmean(f[n]))
    return out


def ledoit_wolf(X):
    """Ledoit-Wolf (2004) scaled-identity shrinkage covariance from observations
    X (n_obs, N). Shrinks the sample covariance S toward mu*I (mu = mean variance)
    by the optimal intensity that minimises expected Frobenius loss. The shrinkage
    constant uses the closed-form b^2 = (sum_t ||x_t||^4 - n||S||^2)/(n^2 N), which
    is the vectorised identity for (1/n^2) sum_t ||x_t x_t^T - S||_F^2 -- so there
    is no per-observation loop."""
    X = np.asarray(X, dtype=np.float64)
    n, N = X.shape
    X = X - X.mean(0, keepdims=True)
    S = (X.T @ X) / n
    mu = np.trace(S) / N
    s2 = float(np.sum(S * S))                       # ||S||_F^2
    d2 = (s2 - np.trace(S) ** 2 / N) / N            # ||S - mu I||_F^2 / N
    b2 = (np.sum((X * X).sum(1) ** 2) - n * s2) / (n * n * N)
    b2 = min(b2, d2)                                # b^2 <= d^2 by construction
    shrink = 0.0 if d2 <= 0 else b2 / d2
    return shrink * mu * np.eye(N) + (1.0 - shrink) * S


def rolling_lw_covariances(returns, idx, window):
    """Strictly-causal trailing Ledoit-Wolf covariance at each index in `idx`.
    For each t in idx with a full trailing window, estimate LW cov from
    returns[t-window:t] (excludes t, so it is causal and never sees H_t). Indices
    with t < window are dropped. returns: (T, N); idx: 1-D int array.
    -> (m, N, N) float32 covariance tensor, plus the kept indices."""
    returns = np.asarray(returns)
    idx = np.asarray(idx)
    keep = idx[idx >= window]
    out = np.empty((len(keep), returns.shape[1], returns.shape[1]))
    for j, t in enumerate(keep):
        out[j] = ledoit_wolf(returns[t - window:t])
    return torch.from_numpy(out).float(), keep


def _normalize_cov_np(M):
    """Batch of covariances (k, N, N) -> correlations, in numpy."""
    d = np.sqrt(np.clip(np.diagonal(M, axis1=-2, axis2=-1), 1e-12, None))
    return np.clip(M / (d[:, :, None] * d[:, None, :]), -1.0, 1.0)


def plot_sample_matrices(real, gen, save_path, n=2, seed=0, show_cov=True):
    """Heatmap grid (viridis) for a visual plausibility check: n real vs n generated
    matrices, showing whether generated matrices reproduce block/sector structure.

    Layout: rows are real (top) / generated (bottom); columns are grouped, with
    `show_cov` (covariance target) giving two side-by-side groups -- covariance and
    the correlation of the SAME sampled matrices -- since the model produces both at
    once, so each matrix is shown on both scales. Otherwise only the correlation
    group is drawn. Each group has its own horizontal colorbar.

    Covariance rows use a *sequential* 0->vmax scale with gamma compression:
    covariance entries are almost all positive, so a symmetric diverging map wastes
    half its range; PowerNorm(gamma<1) expands the low end so off-diagonal structure
    shows while high-variance entries saturate (vmax = robust 99th percentile of the
    displayed real covariances). Correlation rows use a fixed [-1, 1] linear scale."""
    rng = np.random.default_rng(seed)
    n = min(n, len(real), len(gen))
    ridx = rng.choice(len(real), size=n, replace=False)
    gidx = rng.choice(len(gen), size=n, replace=False)
    real_s = (real.numpy() if hasattr(real, "numpy") else np.asarray(real))[ridx]
    gen_s  = (gen.numpy()  if hasattr(gen,  "numpy") else np.asarray(gen))[gidx]

    # column groups (each n wide); rows are real (top) / generated (bottom)
    if show_cov:
        groups = [("covariance",  "cov",  real_s, gen_s),
                  ("correlation", "corr", _normalize_cov_np(real_s), _normalize_cov_np(gen_s))]
        cov_norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(np.percentile(real_s, 99.0)))
    else:
        groups = [("correlation", "corr", real_s, gen_s)]
        cov_norm = None

    ncols = n * len(groups)
    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols + 0.6, 6.2), squeeze=False)
    ims = {}
    for g, (glabel, kind, rdata, gdata) in enumerate(groups):
        for c in range(n):
            col = g * n + c
            for row, data in ((0, rdata), (1, gdata)):
                ax = axes[row, col]
                if kind == "cov":
                    ims[kind] = ax.imshow(data[c], cmap="viridis", norm=cov_norm)
                else:
                    ims[kind] = ax.imshow(data[c], cmap="viridis", vmin=-1.0, vmax=1.0)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(("real", "generated")[row], fontsize=12)
        # one horizontal colorbar under each column group, labelled by scale
        grp_axes = [axes[r, g * n + c] for r in (0, 1) for c in range(n)]
        fig.colorbar(ims[kind], ax=grp_axes, orientation="horizontal",
                     fraction=0.05, pad=0.06, label=glabel)
    fig.suptitle("Sample matrices: real (top) vs generated (bottom)", fontsize=13)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def eval_and_plot(C_real, C_gen, save_path, n_inv_gen=0):
    """Per-run diagnostic plot + marginal W1 stats (gen vs the val reference). The
    irreducible floor is no longer computed here -- it lives in the True (DGP
    oracle) row of the distributional-fidelity table."""
    if n_inv_gen:
        print(f"  [eval] {n_inv_gen} generated samples were dropped (degenerate)")

    off_real = _offdiag(C_real).flatten().numpy()
    off_gen  = _offdiag(C_gen).flatten().numpy()
    w_off    = w1_offdiag(C_real, C_gen)

    facts_real = _per_matrix_facts(_subsample(C_real, FACT_SUBSAMPLE))
    facts_gen  = _per_matrix_facts(_subsample(C_gen,  FACT_SUBSAMPLE))
    w1    = {n: _safe_w1(facts_real[n], facts_gen[n]) for n in FACT_NAMES}

    titles = {"gini": "Eigenvalue Gini", "coph_single": "Cophenetic (single)",
              "coph_ward": "Cophenetic (ward)", "perron": "Perron-Frobenius neg-sum",
              "powerlaw": "Eigenvalue power-law α"}
    panels = [("Off-diagonal correlation", off_real, off_gen, w_off, (-1, 1), 80)]
    for n in FACT_NAMES:
        a, b = facts_real[n], facts_gen[n]
        if np.isnan(a).all() or np.isnan(b).all():        # powerlaw pkg missing
            panels.append((titles[n] + "  (powerlaw missing)", None, None,
                           float("nan"), None, None))
        else:
            lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
            panels.append((titles[n], a, b, w1[n],
                           (lo, hi) if hi > lo else None, 40))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, (title, a, b, wd, rng, bins) in zip(axes.ravel(), panels):
        if a is None:
            ax.set_title(title); ax.set_axis_off(); continue
        ax.hist(a[~np.isnan(a)], bins=bins, alpha=0.5, density=True, label="real",      range=rng)
        ax.hist(b[~np.isnan(b)], bins=bins, alpha=0.5, density=True, label="generated", range=rng)
        ax.set_title(f"{title}  (W₁={wd:.4f})")
        ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

    stats = {
        "offdiag_mean_real": float(off_real.mean()),
        "offdiag_mean_gen":  float(off_gen.mean()),
        "w1_offdiag":        w_off,
        "n_invalid_gen":     n_inv_gen,
    }
    for n in FACT_NAMES:
        stats[f"{n}_mean_real"] = float(np.nanmean(facts_real[n]))
        stats[f"{n}_mean_gen"]  = float(np.nanmean(facts_gen[n]))
        stats[f"w1_{n}"]        = w1[n]
    return stats
