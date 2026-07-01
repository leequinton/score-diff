"""Variance-preserving SDE (Song et al., 2021) for score-based diffusion:
    dx = -0.5 β(t) x dt + sqrt(β(t)) dW,   β(t) = β_min + t (β_max - β_min),
discretised DDPM-style (Ho et al., 2020). Shape-agnostic: `_bcast` broadcasts the
per-batch scalar coefficients across any trailing dims."""

import torch


class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        self.N = N
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas

    @property
    def T(self):
        return 1.0

    def _bcast(self, c, x):
        """Reshape (batch,) scalar c to broadcast across trailing dims of x."""
        return c.reshape((-1,) + (1,) * (x.ndim - 1))

    def marginal_prob(self, x, t):
        """Mean and std of the perturbation kernel p(x_t | x_0)."""
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(self._bcast(log_mean_coeff, x)) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def discretize(self, x, t):
        """DDPM-style discretisation. Returns (f, G) such that one reverse step is
        x_{t-1} = x_t - rev_f + G · z   with rev_f = f - G^2 · score, z ~ N(0, I)."""
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        f = self._bcast(torch.sqrt(alpha), x) * x - x
        G = torch.sqrt(beta)
        return f, G
