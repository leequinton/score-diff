import torch
import torch.nn as nn

from src.models.embeddings import SinusoidalTimeEmbedding, CondEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, h):
        q = self.ln1(h)
        h = h + self.attn(q, q, q, need_weights=False)[0]
        h = h + self.ffn(self.ln2(h))
        return h


class LogCovScoreGNN(nn.Module):
    """Score network for the symmetric matrix-log S = logm(Sigma).

    x : (batch, N, N) symmetric
    t : (batch,) diffusion time in [0, 1]
    returns (batch, N, N), symmetric noise prediction
    """

    def __init__(self, n_assets, hidden_dim=128, n_layers=4, n_heads=4,
                 time_dim=128, dropout=0.0, cond_dim=0):
        super().__init__()
        self.n_assets = n_assets
        self.cond_dim = cond_dim
        self.embed = nn.Linear(n_assets, hidden_dim)
        self.pos = nn.Parameter(torch.randn(n_assets, hidden_dim) * 0.02)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embedding = CondEmbedding(cond_dim, hidden_dim) if cond_dim > 0 else None
        self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, t, cond=None, cond_mask=None):
        h = self.embed(x) + self.pos
        t_bias = self.time_mlp(t)[:, None]
        if self.cond_embedding is not None:
            cond_bias = self.cond_embedding(cond, cond_mask=cond_mask, batch_size=x.shape[0])
            t_bias = t_bias + cond_bias[:, None]
        h = h + t_bias
        for block in self.blocks:
            h = block(h)
            h = h + t_bias
        h = self.ln_out(h)
        out = h @ self.W(h).transpose(-1, -2)
        return 0.5 * (out + out.transpose(-1, -2))
