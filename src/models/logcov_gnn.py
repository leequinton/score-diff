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

# score network for predicting the injected noise in the matrix-log space of a symmetric matrix
class LogCovScoreGNN(nn.Module):

    def __init__(self, n_assets, hidden_dim=128, n_layers=4, n_heads=4,
                 time_dim=128, dropout=0.0, cond_dim=0):
        super().__init__()
        self.n_assets = n_assets
        self.cond_dim = cond_dim
        self.embed = nn.Linear(n_assets, hidden_dim)   # tokenize matrix row-wise
        self.pos = nn.Parameter(torch.randn(n_assets, hidden_dim) * 0.02)  # fixed per-asset pos -> no equivariance
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_embedding = CondEmbedding(cond_dim, hidden_dim) if cond_dim > 0 else None
        if cond_dim > 0:
            # zero-init per-injection-point gate: starts unconditional, learns the dose
            self.cond_gate = nn.Parameter(torch.zeros(n_layers + 1))
            self.blocks = nn.ModuleList(
            TransformerBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)  # output projection: out = h W h^T

    def forward(self, x, t, cond=None, cond_mask=None):
        h = self.embed(x) + self.pos
        t_bias = self.time_mlp(t)[:, None]                  # (B, 1, d), full strength every layer
        cond_bias = None
        if self.cond_embedding is not None:
            cond_bias = self.cond_embedding(cond, cond_mask=cond_mask,
                                            batch_size=x.shape[0])[:, None]   # (B, 1, d)

        h = h + t_bias
        if cond_bias is not None:
            h = h + self.cond_gate[0] * cond_bias
        for i, block in enumerate(self.blocks, start=1):
            h = block(h)
            h = h + t_bias
            if cond_bias is not None:
                h = h + self.cond_gate[i] * cond_bias

        h = self.ln_out(h)
        out = h @ self.W(h).transpose(-1, -2)
        return 0.5 * (out + out.transpose(-1, -2))
