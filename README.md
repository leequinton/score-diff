# graph_diffusion

Score-based diffusion for **factor-structured covariance matrices** of asset returns.

A bipartite graph neural network predicts the noise in a variance-preserving SDE
over the factor decomposition `Σ = B F Bᵀ + D`, where `B` are factor loadings,
`F` the factor covariance, and `D` the idiosyncratic variances. A full-Cholesky
model on the unstructured `N×N` correlation matrix serves as the baseline, so the
only thing that varies between the two is the structural prior.

Regime conditioning (trailing equity- and interest-rate volatility) is supported
via classifier-free guidance.

## Setup

```bash
pip install -r requirements.txt
```

Data is **not** versioned; it is regenerated from public sources. Download from
the Kenneth French data library and FRED:

- `49_Industry_Portfolios_Daily.CSV` (value-weighted returns)
- `F-F_Research_Data_5_Factors_2x3_daily.CSV`
- `DGS10.csv` (FRED 10-year Treasury yield)

Then build the processed tensors:

```bash
python data/prep_factors_daily.py \
    --industries 49_Industry_Portfolios_Daily.CSV \
    --factors    F-F_Research_Data_5_Factors_2x3_daily.CSV \
    --dgs10      DGS10.csv
```

This writes `B/F/C/D` tensors and the `cond` tensor to `data/processed/`.

## Running

```bash
python -m src.train               # bipartite, single seed
python -m src.train_baseline      # full-Cholesky baseline, single seed
python -m src.train_multiseed --seeds 0 1 2     # both models, aggregated mean ± std
```

Outputs (loss curves, real-vs-generated diagnostics, regime sweeps, multiseed JSON)
are written to `results/`; checkpoints to `checkpoints/`. Both directories are
created on demand and are git-ignored.

## Layout

| Path | Contents |
|------|----------|
| `data/factor_model.py` | rolling-window factor regressions (B, F, C, D) |
| `data/prep_factors_daily.py` | daily data prep + conditioning variables |
| `src/data/dataset.py` | datasets, normalisers, temporal train/val split |
| `src/diffusion/sde.py` | variance-preserving SDE |
| `src/diffusion/losses.py` | denoising score-matching loss (+ CFG dropout) |
| `src/diffusion/solver.py` | reverse-time sampler (+ classifier-free guidance) |
| `src/models/gnn.py` | bipartite factor-structured score network |
| `src/models/full_chol_gnn.py` | unstructured full-Cholesky baseline |
| `src/evaluation/evaluate.py` | Wasserstein metrics, stylized-fact diagnostics, regime sweep |
| `src/train.py`, `src/train_baseline.py` | training drivers |
| `src/train_multiseed.py` | multi-seed wrapper with aggregated reporting |

## Configuration

Each training script has a top-of-file `CFG` dict and `FREQ`/`SUFFIX`/`GAP`
constants. Set `cond_dim=0` to disable conditioning, `1` for equity-vol only,
`2` for `[equity vol, rate vol]`.
