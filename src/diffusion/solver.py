"""Euler-Maruyama reverse-SDE sampler for the full-Cholesky baseline.

Supports optional classifier-free guidance (Ho and Salimans, 2022):

- cond=None, guidance_scale=0   → unconditional (model uses its null embedding)
- cond=tensor, guidance_scale=0 → pure conditional
- cond=tensor, guidance_scale=w → CFG: ε = (1+w)·ε_cond − w·ε_uncond
                                   w > 0 amplifies the regime beyond training data;
                                   w = -1 is exactly unconditional.

When CFG is active the model is called twice per timestep (cond + uncond), so
sampling cost roughly doubles.
"""

import torch


def _cfg_combine(eps_cond, eps_uncond, w):
    """Classifier-free-guidance combination: (1+w)·cond − w·uncond."""
    return (1 + w) * eps_cond - w * eps_uncond


@torch.no_grad()
def sample_full_chol(model, sde, batch_size, n_assets, device, eps_t=1e-3,
                     cond=None, guidance_scale=0.0):
    """Reverse-SDE Euler-Maruyama for the full Cholesky baseline."""
    N = n_assets
    tril = torch.tril(torch.ones(N, N, device=device))
    X = torch.randn(batch_size, N, N, device=device) * tril

    timesteps = torch.linspace(sde.T, eps_t, sde.N, device=device)
    use_cfg = (cond is not None) and (guidance_scale != 0.0)

    for i in range(sde.N):
        t = torch.full((batch_size,), timesteps[i].item(), device=device)
        f, G = sde.discretize(X, t)

        if use_cfg:
            eps_c = model(X, t, cond=cond)
            eps_u = model(X, t, cond=None)
            eps_pred = _cfg_combine(eps_c, eps_u, guidance_scale)
        else:
            eps_pred = model(X, t, cond=cond)

        _, std = sde.marginal_prob(X, t)
        score = -eps_pred / std[:, None, None]
        rev_f = f - G[:, None, None] ** 2 * score

        X = X - rev_f
        if i < sde.N - 1:
            z = torch.randn_like(X) * tril
            X = X + G[:, None, None] * z
        X = (X * tril).clamp(-8, 8)

    return X
