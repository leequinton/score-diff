"""Euler-Maruyama reverse-SDE sampler for the log-covariance baseline.

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

from src.diffusion.losses import sym_randn_like


def _cfg_combine(eps_cond, eps_uncond, w):
    """Classifier-free-guidance combination: (1+w)·cond − w·uncond."""
    return (1 + w) * eps_cond - w * eps_uncond


@torch.no_grad()
def sample_logcov(model, sde, batch_size, n_assets, device, eps_t=1e-3,
                  cond=None, guidance_scale=0.0):
    """Reverse-SDE Euler-Maruyama for the log-covariance baseline. The iterate
    stays symmetric: the model output is symmetric and the injected noise is
    symmetric, and the per-entry SDE coefficients preserve that symmetry."""
    N = n_assets
    X = sym_randn_like(torch.empty(batch_size, N, N, device=device))

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
            z = sym_randn_like(X)
            X = X + G[:, None, None] * z
        X = X.clamp(-8, 8)

    return X
