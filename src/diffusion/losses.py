"""Denoising score matching loss for the full-Cholesky VP-SDE baseline.

The model is trained to predict the Gaussian noise added by the forward kernel
(epsilon-prediction). At inference we recover the score via score = -eps / std.

When `cond` is supplied, classifier-free-guidance dropout (Ho and Salimans, 2022)
is applied: with probability `cond_dropout`, individual samples have their
conditioning replaced by the model's learned null embedding. The same trained
network thus implements both conditional and unconditional sampling.
"""

import torch


def _cfg_mask(bs, cond, cond_dropout, device):
    """Boolean mask of shape (bs,): True = use null embedding for this sample.
    Returns None when there is nothing to drop (either no cond or zero dropout)."""
    if cond is None or cond_dropout <= 0.0:
        return None
    return torch.rand(bs, device=device) < cond_dropout


def vpsde_dsm_loss_chol(model, sde, X0, eps_t=1e-3, cond=None, cond_dropout=0.1):
    """DSM loss for a single lower-triangular matrix (full-Cholesky baseline)."""
    bs = X0.shape[0]
    N = X0.shape[-1]
    device = X0.device

    t = torch.rand(bs, device=device) * (sde.T - eps_t) + eps_t
    mean, std = sde.marginal_prob(X0, t)

    tril = torch.tril(torch.ones(N, N, device=device))
    z = torch.randn_like(X0) * tril
    X_t = mean + std[:, None, None] * z

    cond_mask = _cfg_mask(bs, cond, cond_dropout, device)
    eps_pred = model(X_t, t, cond=cond, cond_mask=cond_mask)
    return ((eps_pred - z) * tril).pow(2).sum() / (bs * tril.sum())
