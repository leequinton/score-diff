"""Denoising score matching loss (epsilon-prediction) for the log-covariance
VP-SDE baseline. With `cond`, applies classifier-free-guidance dropout (Ho and
Salimans, 2022) so one network does both conditional and unconditional sampling."""

import torch


def _cfg_mask(bs, cond, cond_dropout, device):
    """(bs,) mask: True = use null embedding. None when nothing to drop."""
    if cond is None or cond_dropout <= 0.0:
        return None
    return torch.rand(bs, device=device) < cond_dropout


def sym_randn_like(X):
    """Symmetric Gaussian noise: the free upper-triangle entries are iid and
    mirrored into the lower triangle, keeping X_t symmetric under diffusion."""
    z = torch.randn_like(X)
    return torch.triu(z) + torch.triu(z, 1).transpose(-1, -2)


def vpsde_dsm_loss_logcov(model, sde, X0, eps_t=1e-3, cond=None, cond_dropout=0.1):
    """DSM loss for a single symmetric matrix (log-covariance baseline)."""
    bs = X0.shape[0]
    N = X0.shape[-1]
    device = X0.device

    t = torch.rand(bs, device=device) * (sde.T - eps_t) + eps_t
    mean, std = sde.marginal_prob(X0, t)

    z = sym_randn_like(X0)
    X_t = mean + std[:, None, None] * z

    cond_mask = _cfg_mask(bs, cond, cond_dropout, device)
    eps_pred = model(X_t, t, cond=cond, cond_mask=cond_mask)
    # score over the free (upper-triangle) entries so each off-diagonal pair counts once
    triu = torch.triu(torch.ones(N, N, device=device))
    return ((eps_pred - z) * triu).pow(2).sum() / (bs * triu.sum())
