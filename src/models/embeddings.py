import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class CondEmbedding(nn.Module):
    """Maps a continuous conditioning vector to a hidden-dim bias, with a learned
    null token enabling classifier-free guidance (Ho and Salimans, 2022).

    Modes:
      cond=None                          → broadcast null embedding (unconditional)
      cond=(B, cond_dim), cond_mask=None → fully conditional
      cond=(B, cond_dim), cond_mask=(B,) → masked: True entries use null (CFG dropout)

    Returns a (B, hidden_dim) bias to be folded into the time bias.
    """

    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, cond, cond_mask=None, batch_size=None):
        if cond is None:
            assert batch_size is not None, "batch_size required when cond is None"
            return self.null_embedding.unsqueeze(0).expand(batch_size, -1)
        out = self.mlp(cond)
        if cond_mask is not None:
            out = torch.where(cond_mask.unsqueeze(-1), self.null_embedding.unsqueeze(0), out)
        return out
