"""Compare real vs generated correlation/covariance matrices.

Correlation stylized facts (CorrGAN-style): mean off-diagonal correlation, Gini of
the eigenvalue spectrum, and cophenetic correlation — each scored by Wasserstein-1
against an empirical split-half floor. Covariance adds a variance-scale W1."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform


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


def cophenetic_per_sample(C, method="average"):
    C_np = C.numpy() if hasattr(C, "numpy") else np.asarray(C)
    out = np.empty(C_np.shape[0])
    for i in range(C_np.shape[0]):
        dist = squareform(np.clip(1.0 - np.abs(C_np[i]), 0.0, 2.0), checks=False)
        Z = linkage(dist, method=method)
        out[i], _ = cophenet(Z, dist)
    return out


def variance_diagnostics(Sigma_real, Sigma_gen, Sigma_train=None, n_trials=5, seed=0):
    """Variance-space (scale) fidelity — the dimension correlation metrics discard.
    Wasserstein-1 on the pooled distribution of per-asset variances (diagonals of Σ)
    against an empirical floor from random halves of the real (val) data.

    Sigma_real, Sigma_gen: (T, N, N) covariance batches (not correlation).
    Sigma_train (optional): the train-split covariances. When given, the same
    variance stats are reported for train, plus gen-vs-train W1 — to disentangle
    exp-tail reconstruction bias (gen overshoots BOTH train and val) from
    train/val regime shift (gen tracks train; both differ from val)."""
    var_real = torch.diagonal(Sigma_real, dim1=-2, dim2=-1).flatten().numpy()
    var_gen  = torch.diagonal(Sigma_gen,  dim1=-2, dim2=-1).flatten().numpy()

    w_var = float(wasserstein_distance(var_real, var_gen))

    # floor from random halves of the real covariances
    fv = []
    half = len(Sigma_real) // 2
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + trial)
        perm = torch.randperm(len(Sigma_real), generator=g)
        Sa, Sb = Sigma_real[perm[:half]], Sigma_real[perm[half:half * 2]]
        fv.append(wasserstein_distance(
            torch.diagonal(Sa, dim1=-2, dim2=-1).flatten().numpy(),
            torch.diagonal(Sb, dim1=-2, dim2=-1).flatten().numpy()))

    out = {
        "w1_variance":       w_var,
        "floor_w1_variance": float(np.mean(fv)),
        "var_mean_real":     float(var_real.mean()),
        "var_mean_gen":      float(var_gen.mean()),
        # median is robust to the exp() tail of log-variance reconstruction
        "var_median_real":   float(np.median(var_real)),
        "var_median_gen":    float(np.median(var_gen)),
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


def empirical_floor(C_real, n_trials=5, seed=0):
    floors = {"w1_offdiag": [], "w1_gini": [], "w1_coph": []}
    half = len(C_real) // 2
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + trial)
        perm = torch.randperm(len(C_real), generator=g)
        Ca, Cb = C_real[perm[:half]], C_real[perm[half:half * 2]]

        floors["w1_offdiag"].append(w1_offdiag(Ca, Cb))

        eig_a = np.sort(torch.linalg.eigvalsh(Ca).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
        eig_b = np.sort(torch.linalg.eigvalsh(Cb).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
        gini_a, gini_b = eig_gini(eig_a), eig_gini(eig_b)
        floors["w1_gini"].append(float(wasserstein_distance(gini_a, gini_b)))

        coph_a, coph_b = cophenetic_per_sample(Ca), cophenetic_per_sample(Cb)
        floors["w1_coph"].append(float(wasserstein_distance(coph_a, coph_b)))

    return {f"floor_{k}": float(np.mean(v)) for k, v in floors.items()}


def plot_sample_matrices(M_real, M_gen, save_path, n=5, seed=0, kind="covariance"):
    """Heatmap grid for a quick visual plausibility check: n real (top row) vs n
    generated (bottom row) matrices. Shows whether generated matrices reproduce
    block/sector structure, not just aggregate statistics.

    kind="correlation" uses the fixed diverging scale [-1, 1] (correlations are
    genuinely signed). kind="covariance" uses a *sequential* 0->vmax scale with
    gamma compression: covariance entries are almost all positive, so a symmetric
    diverging map wastes half its range and renders the bulk as washed-out pale
    pink. PowerNorm(gamma<1) expands the low end so the off-diagonal structure
    shows, while the high-variance entries saturate (vmax = robust 99th percentile
    of the displayed real matrices, shared across both rows for comparability)."""
    rng = np.random.default_rng(seed)
    n = min(n, len(M_real), len(M_gen))
    ridx = rng.choice(len(M_real), size=n, replace=False)
    gidx = rng.choice(len(M_gen), size=n, replace=False)

    if kind == "correlation":
        cmap, norm, vmin, vmax = "RdBu_r", None, -1.0, 1.0
        cbar_label = title_word = "correlation"
    else:
        hi = float(np.percentile(M_real[ridx].numpy(), 99.0))
        cmap, norm, vmin, vmax = "Reds", PowerNorm(gamma=0.5, vmin=0.0, vmax=hi), None, None
        cbar_label = title_word = "covariance"

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.4))
    axes = np.atleast_2d(axes)
    for col in range(n):
        for row, (M, idx, lbl) in enumerate(
            [(M_real, ridx[col], "real"), (M_gen, gidx[col], "generated")]
        ):
            ax = axes[row, col]
            im = ax.imshow(M[idx].numpy(), cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(lbl, fontsize=13)
    fig.suptitle(f"Sample {title_word} matrices: real (top) vs generated (bottom)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=cbar_label)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def eval_and_plot(C_real, C_gen, save_path, n_inv_gen=0):
    if n_inv_gen:
        print(f"  [eval] {n_inv_gen} generated samples were dropped (degenerate)")

    off_real = _offdiag(C_real).flatten().numpy()
    off_gen  = _offdiag(C_gen).flatten().numpy()

    eig_real_all = np.sort(torch.linalg.eigvalsh(C_real).clamp_min(1e-12).numpy(), axis=-1)[:, ::-1]
    eig_gen_all  = np.sort(torch.linalg.eigvalsh(C_gen).clamp_min(1e-12).numpy(),  axis=-1)[:, ::-1]

    w_off     = w1_offdiag(C_real, C_gen)

    gini_real = eig_gini(np.ascontiguousarray(eig_real_all))
    gini_gen  = eig_gini(np.ascontiguousarray(eig_gen_all))
    w_gini    = float(wasserstein_distance(gini_real, gini_gen))

    coph_real = cophenetic_per_sample(C_real)
    coph_gen  = cophenetic_per_sample(C_gen)
    w_coph    = float(wasserstein_distance(coph_real, coph_gen))

    floor = empirical_floor(C_real)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    ax = axes[0]
    ax.hist(off_real, bins=80, alpha=0.5, density=True, label="real",      range=(-1, 1))
    ax.hist(off_gen,  bins=80, alpha=0.5, density=True, label="generated", range=(-1, 1))
    ax.set_title(f"Off-diagonal correlation entries  (W₁={w_off:.4f}, floor={floor['floor_w1_offdiag']:.4f})")
    ax.set_xlabel("correlation")
    ax.legend()

    ax = axes[1]
    g_lo = min(gini_real.min(), gini_gen.min())
    g_hi = max(gini_real.max(), gini_gen.max())
    ax.hist(gini_real, bins=40, alpha=0.5, density=True, label="real",      range=(g_lo, g_hi))
    ax.hist(gini_gen,  bins=40, alpha=0.5, density=True, label="generated", range=(g_lo, g_hi))
    ax.set_title(f"Eigenvalue Gini coefficient  (W₁={w_gini:.4f}, floor={floor['floor_w1_gini']:.4f})")
    ax.set_xlabel("Gini")
    ax.legend()

    ax = axes[2]
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
        "gini_mean_real":      float(gini_real.mean()),
        "gini_mean_gen":       float(gini_gen.mean()),
        "coph_mean_real":      float(coph_real.mean()),
        "coph_mean_gen":       float(coph_gen.mean()),
        "w1_offdiag":          w_off,
        "w1_gini":             w_gini,
        "w1_coph":             w_coph,
        **floor,
        "n_invalid_gen":       n_inv_gen,
    }
