"""Compare real vs generated factor-model covariances.

Diagnostics: off-diagonal correlation distribution, eigenvalue spectrum, ‖C‖_F,
plus Wasserstein-1 metrics on each (pooled off-diagonals, per-rank eigenvalues,
sliced over flattened upper-triangles for joint structure)."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


def _offdiag(M):
    N = M.shape[-1]
    mask = ~torch.eye(N, dtype=torch.bool, device=M.device)
    return M[:, mask]


def _upper_tri(M):
    N = M.shape[-1]
    iu = torch.triu_indices(N, N, offset=1)
    return M[:, iu[0], iu[1]]


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


def gmvp_weights(Sigma, ridge=0.0):
    """Global minimum-variance weights w = Sigma^-1 1 / (1' Sigma^-1 1),
    unconstrained (long-short allowed). Sigma: (..., N, N) -> (..., N).
    In the log representation note Sigma^-1 = expm(-S), so these weights are a
    smooth function of the modeled matrix log."""
    N = Sigma.shape[-1]
    if ridge > 0:
        Sigma = Sigma + ridge * torch.eye(N, device=Sigma.device, dtype=Sigma.dtype)
    ones = torch.ones(*Sigma.shape[:-1], 1, device=Sigma.device, dtype=Sigma.dtype)
    x = torch.linalg.solve(Sigma, ones)              # Sigma^-1 1, (..., N, 1)
    w = x / x.sum(dim=-2, keepdim=True)
    return w.squeeze(-1)                             # (..., N)


def gmvp_diagnostics(Cov_real, Sigma_gen):
    """GMVP distributional-fidelity eval for an unconditional generator.

    Sigma_bar = mean_t Cov_real is the population (realized) covariance over the
    validation period; realized variance of any weight is w' Sigma_bar w. The
    reference points, from best to naive:

      oracle      GMVP(Sigma_bar)                 - lower bound (needs the population cov)
      truesample  mean_t var of GMVP(H_t)         - ceiling a per-sample generator can hit
                                                    (each single draw pays an estimation penalty,
                                                    so this is ~2x oracle, not 1x)
      gen         mean_i var of GMVP(Sigma_gen_i) - the model, used one sample at a time
      genbar      GMVP(mean_i Sigma_gen_i)        - the model, denoised by averaging samples
                                                    (-> oracle if the model's mean is right)
      equalwt     1/N portfolio                   - naive floor

    So `ratio_gen` should be read against `ratio_truesample` (matching it = the
    samples are as useful as real covariances), and `genbar` against `oracle`
    (matching it = the model's mean recovers the population). Gross leverage 1'|w|
    flags ill-conditioned generated covariances (extreme short positions).
    """
    N = Cov_real.shape[-1]
    Sigma_bar = Cov_real.mean(0)                     # (N, N) population covariance

    def realized_var(w):                            # w: (..., N) -> (...)
        return ((w @ Sigma_bar) * w).sum(-1)

    v_oracle = float(realized_var(gmvp_weights(Sigma_bar)))
    v_true = realized_var(gmvp_weights(Cov_real))    # (T,) per-period true-cov GMVP
    w_gen = gmvp_weights(Sigma_gen)                  # (n, N)
    v_gen = realized_var(w_gen)                      # (n,)
    v_genbar = float(realized_var(gmvp_weights(Sigma_gen.mean(0))))
    v_ew = float(realized_var(torch.full((N,), 1.0 / N)))

    return {
        "gmvp_var_oracle":       v_oracle,
        "gmvp_var_truesample":   float(v_true.mean()),
        "gmvp_var_gen_mean":     float(v_gen.mean()),
        "gmvp_var_gen_median":   float(v_gen.median()),
        "gmvp_var_genbar":       v_genbar,
        "gmvp_var_equalwt":      v_ew,
        "gmvp_ratio_gen":        float(v_gen.mean()) / v_oracle,
        "gmvp_ratio_truesample": float(v_true.mean()) / v_oracle,
        "gmvp_gross_lev_gen":    float(w_gen.abs().sum(-1).mean()),
    }


def w1_offdiag(C_real, C_gen):
    """1-D Wasserstein-1 on the pooled distribution of off-diagonal correlations."""
    a = _offdiag(C_real).flatten().numpy()
    b = _offdiag(C_gen).flatten().numpy()
    return float(wasserstein_distance(a, b))


def w1_eigs_per_rank(eig_real_all, eig_gen_all):
    """Per-rank 1-D Wasserstein between distributions of the k-th eigenvalue.
    Inputs are (T, N) arrays sorted descending."""
    return np.array([
        wasserstein_distance(eig_real_all[:, k], eig_gen_all[:, k])
        for k in range(eig_real_all.shape[-1])
    ])


def sliced_w1(C_real, C_gen, n_proj=256, seed=0):
    """Sliced Wasserstein-1 over random projections of flattened upper-triangles.
    Captures joint structure that pooled marginals miss."""
    a = _upper_tri(C_real).numpy()
    b = _upper_tri(C_gen).numpy()
    D = a.shape[-1]
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n_proj, D)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    a_p, b_p = a @ dirs.T, b @ dirs.T
    return float(np.mean([wasserstein_distance(a_p[:, k], b_p[:, k]) for k in range(n_proj)]))


def eig_gini(eigs_desc):
    asc = np.sort(eigs_desc, axis=-1)             
    N = asc.shape[-1]
    ranks = np.arange(1, N + 1)
    total = asc.sum(-1)
    return 2 * (asc * ranks).sum(-1) / (N * total) - (N + 1) / N


def cophenetic_per_sample(C, method="average"):
    C_np = C.numpy() if hasattr(C, "numpy") else np.asarray(C)
    out = np.empty(C_np.shape[0])
    for i in range(C_np.shape[0]):
        dist = squareform(np.clip(1.0 - np.abs(C_np[i]), 0.0, 2.0), checks=False)
        Z = linkage(dist, method=method)
        out[i], _ = cophenet(Z, dist)
    return out


def variance_diagnostics(Sigma_real, Sigma_gen, n_trials=5, seed=0):
    """Variance-space (scale) fidelity — the dimension that correlation-space
    metrics discard. Wasserstein-1 on the pooled distribution of per-asset
    variances (diagonals of Σ) and on the off-diagonal covariance entries,
    each against an empirical floor from random halves of the real data.

    Sigma_real, Sigma_gen: (T, N, N) covariance batches (not correlation)."""
    var_real = torch.diagonal(Sigma_real, dim1=-2, dim2=-1).flatten().numpy()
    var_gen  = torch.diagonal(Sigma_gen,  dim1=-2, dim2=-1).flatten().numpy()
    covoff_real = _offdiag(Sigma_real).flatten().numpy()
    covoff_gen  = _offdiag(Sigma_gen).flatten().numpy()

    w_var = float(wasserstein_distance(var_real, var_gen))
    w_cov = float(wasserstein_distance(covoff_real, covoff_gen))

    # floors from random halves of the real covariances
    fv, fc = [], []
    half = len(Sigma_real) // 2
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + trial)
        perm = torch.randperm(len(Sigma_real), generator=g)
        Sa, Sb = Sigma_real[perm[:half]], Sigma_real[perm[half:half * 2]]
        fv.append(wasserstein_distance(
            torch.diagonal(Sa, dim1=-2, dim2=-1).flatten().numpy(),
            torch.diagonal(Sb, dim1=-2, dim2=-1).flatten().numpy()))
        fc.append(wasserstein_distance(_offdiag(Sa).flatten().numpy(),
                                       _offdiag(Sb).flatten().numpy()))

    return {
        "w1_variance":       w_var,
        "w1_cov_offdiag":    w_cov,
        "floor_w1_variance": float(np.mean(fv)),
        "floor_w1_cov_offdiag": float(np.mean(fc)),
        "var_mean_real":     float(var_real.mean()),
        "var_mean_gen":      float(var_gen.mean()),
        # median is robust to the exp() tail of log-variance reconstruction
        "var_median_real":   float(np.median(var_real)),
        "var_median_gen":    float(np.median(var_gen)),
    }


def empirical_floor(C_real, n_trials=5, seed=0):
    floors = {
        "w1_offdiag": [], "w1_eig_mean": [], "w1_eig_top": [],
        "sliced_w1": [], "w1_gini": [], "w1_coph": [],
    }
    half = len(C_real) // 2
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + trial)
        perm = torch.randperm(len(C_real), generator=g)
        Ca, Cb = C_real[perm[:half]], C_real[perm[half:half * 2]]

        floors["w1_offdiag"].append(w1_offdiag(Ca, Cb))
        floors["sliced_w1"].append(sliced_w1(Ca, Cb, seed=seed + trial))

        eig_a = np.sort(torch.linalg.eigvalsh(Ca).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
        eig_b = np.sort(torch.linalg.eigvalsh(Cb).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
        w_eig = w1_eigs_per_rank(np.ascontiguousarray(eig_a), np.ascontiguousarray(eig_b))
        floors["w1_eig_mean"].append(float(w_eig.mean()))
        floors["w1_eig_top"].append(float(w_eig[0]))

        gini_a, gini_b = eig_gini(eig_a), eig_gini(eig_b)
        floors["w1_gini"].append(float(wasserstein_distance(gini_a, gini_b)))

        coph_a, coph_b = cophenetic_per_sample(Ca), cophenetic_per_sample(Cb)
        floors["w1_coph"].append(float(wasserstein_distance(coph_a, coph_b)))

    return {f"floor_{k}": float(np.mean(v)) for k, v in floors.items()}


def regime_sweep_summary(C_gen_list, cond_values_raw, save_path=None, label=""):
    """For each entry in C_gen_list (one batch of generated correlation matrices
    sampled at a fixed cond value), compute summary statistics and plot how they
    shift with cond. Demonstrates that conditioning actually moves the output.

    C_gen_list:      list of (T_i, N, N) correlation matrix batches.
    cond_values_raw: list of float — the (denormalised) cond value used to draw each batch.
    Returns a list of dicts with cond / mean_offdiag / top_eig / n_valid."""
    rows = []
    for C_gen, c in zip(C_gen_list, cond_values_raw):
        if len(C_gen) == 0:
            rows.append({"cond": float(c), "mean_offdiag": float("nan"),
                         "top_eig": float("nan"), "n_valid": 0})
            continue
        off = _offdiag(C_gen).flatten().numpy()
        eigs = torch.linalg.eigvalsh(C_gen).clamp_min(1e-12).numpy()
        top = np.sort(eigs, axis=-1)[:, -1].mean()
        rows.append({
            "cond":         float(c),
            "mean_offdiag": float(off.mean()),
            "top_eig":      float(top),
            "n_valid":      int(len(C_gen)),
        })

    if save_path is not None:
        conds = [r["cond"] for r in rows]
        offs  = [r["mean_offdiag"] for r in rows]
        eigs  = [r["top_eig"]      for r in rows]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(conds, offs, marker="o")
        axes[0].set_xlabel("conditioning value (raw)")
        axes[0].set_ylabel("mean off-diagonal correlation")
        axes[0].set_title(f"Regime → mean off-diag  {label}".strip())
        axes[1].plot(conds, eigs, marker="o")
        axes[1].set_xlabel("conditioning value (raw)")
        axes[1].set_ylabel("top eigenvalue (mean)")
        axes[1].set_title(f"Regime → top eigenvalue  {label}".strip())
        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close(fig)

    return rows


def plot_sample_matrices(C_real, C_gen, save_path, n=5, seed=0):
    """Heatmap grid for a quick visual plausibility check: n real (top row) vs n
    generated (bottom row) correlation matrices on a shared diverging scale [-1, 1].
    Shows whether generated matrices reproduce block/sector structure, not just
    aggregate statistics."""
    rng = np.random.default_rng(seed)
    n = min(n, len(C_real), len(C_gen))
    ridx = rng.choice(len(C_real), size=n, replace=False)
    gidx = rng.choice(len(C_gen), size=n, replace=False)

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    axes = np.atleast_2d(axes)
    for col in range(n):
        for row, (C, idx, lbl) in enumerate(
            [(C_real, ridx[col], "real"), (C_gen, gidx[col], "generated")]
        ):
            ax = axes[row, col]
            im = ax.imshow(C[idx].numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(lbl, fontsize=13)
    fig.suptitle("Sample correlation matrices: real (top) vs generated (bottom)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="correlation")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def eval_and_plot(C_real, C_gen, save_path, n_inv_gen=0):
    if n_inv_gen:
        print(f"  [eval] {n_inv_gen} generated samples were dropped (degenerate)")

    off_real = _offdiag(C_real).flatten().numpy()
    off_gen  = _offdiag(C_gen).flatten().numpy()

    eig_real_all = np.sort(torch.linalg.eigvalsh(C_real).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
    eig_gen_all  = np.sort(torch.linalg.eigvalsh(C_gen).clamp_min(1e-12).numpy(),  axis=-1)[:, ::-1]
    eig_real, eig_gen = eig_real_all.mean(0), eig_gen_all.mean(0)

    w_off    = w1_offdiag(C_real, C_gen)
    w_eig    = w1_eigs_per_rank(np.ascontiguousarray(eig_real_all),
                                np.ascontiguousarray(eig_gen_all))
    w_sliced = sliced_w1(C_real, C_gen)

    gini_real = eig_gini(np.ascontiguousarray(eig_real_all))
    gini_gen  = eig_gini(np.ascontiguousarray(eig_gen_all))
    w_gini    = float(wasserstein_distance(gini_real, gini_gen))

    coph_real = cophenetic_per_sample(C_real)
    coph_gen  = cophenetic_per_sample(C_gen)
    w_coph    = float(wasserstein_distance(coph_real, coph_gen))

    floor = empirical_floor(C_real)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    N = C_real.shape[-1]

    ax = axes[0, 0]
    ax.hist(off_real, bins=80, alpha=0.5, density=True, label="real",      range=(-1, 1))
    ax.hist(off_gen,  bins=80, alpha=0.5, density=True, label="generated", range=(-1, 1))
    ax.set_title(f"Off-diagonal correlation entries  (W₁={w_off:.4f}, floor={floor['floor_w1_offdiag']:.4f})")
    ax.set_xlabel("correlation")
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(eig_real, marker="o", label="real")
    ax.semilogy(eig_gen,  marker="x", label="generated")
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("Eigenvalue spectrum of C (mean, sorted desc)")
    ax.set_xlabel("rank")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(w_eig, marker=".", label="model vs real")
    ax.axhline(floor["floor_w1_eig_mean"], color="grey", linestyle="--", linewidth=0.8,
               label=f"floor mean={floor['floor_w1_eig_mean']:.4f}")
    ax.set_yscale("log")
    ax.set_title(f"Per-rank W₁ between eigenvalue distributions  (mean={w_eig.mean():.4f})")
    ax.set_xlabel("rank")
    ax.set_ylabel("W₁")
    ax.legend()

    ax = axes[1, 1]
    g_lo = min(gini_real.min(), gini_gen.min())
    g_hi = max(gini_real.max(), gini_gen.max())
    ax.hist(gini_real, bins=40, alpha=0.5, density=True, label="real",      range=(g_lo, g_hi))
    ax.hist(gini_gen,  bins=40, alpha=0.5, density=True, label="generated", range=(g_lo, g_hi))
    ax.set_title(f"Eigenvalue Gini coefficient  (W₁={w_gini:.4f}, floor={floor['floor_w1_gini']:.4f})")
    ax.set_xlabel("Gini")
    ax.legend()

    ax = axes[1, 2]
    c_lo = min(coph_real.min(), coph_gen.min())
    c_hi = max(coph_real.max(), coph_gen.max())
    ax.hist(coph_real, bins=40, alpha=0.5, density=True, label="real",      range=(c_lo, c_hi))
    ax.hist(coph_gen,  bins=40, alpha=0.5, density=True, label="generated", range=(c_lo, c_hi))
    ax.set_title(f"Cophenetic correlation  (W₁={w_coph:.4f}, floor={floor['floor_w1_coph']:.4f})")
    ax.set_xlabel("cophenetic corr")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)

    return {
        "offdiag_mean_real":   float(off_real.mean()),
        "offdiag_mean_gen":    float(off_gen.mean()),
        "offdiag_std_real":    float(off_real.std()),
        "offdiag_std_gen":     float(off_gen.std()),
        "top_eig_real":        float(eig_real[0]),
        "top_eig_gen":         float(eig_gen[0]),
        "gini_mean_real":      float(gini_real.mean()),
        "gini_mean_gen":       float(gini_gen.mean()),
        "coph_mean_real":      float(coph_real.mean()),
        "coph_mean_gen":       float(coph_gen.mean()),
        "w1_offdiag":          w_off,
        "w1_eig_mean":         float(w_eig.mean()),
        "w1_eig_top":          float(w_eig[0]),
        "sliced_w1":           w_sliced,
        "w1_gini":             w_gini,
        "w1_coph":             w_coph,
        **floor,
        "n_invalid_gen":       n_inv_gen,
    }
