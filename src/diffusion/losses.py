import torch

# for cfg dropout, mask the cond input with a learned null embedding (for uncond model)
def _cfg_mask(bs, cond, cond_dropout, device):
    if cond is None or cond_dropout <= 0.0:
        return None
    return torch.rand(bs, device=device) < cond_dropout


def sym_randn_like(X):
    """Symmetric Gaussian noise: the free upper-triangle entries are iid and
    mirrored into the lower triangle, keeping X_t symmetric under diffusion."""
    z = torch.randn_like(X)
    return torch.triu(z) + torch.triu(z, 1).transpose(-1, -2)

# desnoising score matching loss for a single symmetric matrix
def vpsde_dsm_loss_logcov(model, sde, X0, eps_t=1e-3, cond=None, cond_dropout=0.1):
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
