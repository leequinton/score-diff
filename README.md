# graph_diffusion

Score-based diffusion for covariance matrices of asset returns, trained and
evaluated on a calibrated DCC-GARCH simulation.

The model diffuses the **matrix logarithm** `S = logm(Σ)` of a covariance (or
correlation) matrix. Working in log-Euclidean space makes the target unconstrained
— any symmetric matrix is a valid `logm`, and `expm` maps back to an SPD matrix —
so the network never has to enforce positive-definiteness. A transformer score
network (`LogCovScoreGNN`) predicts the noise in a variance-preserving SDE
(epsilon-prediction / denoising score matching).

Regime conditioning (trailing market volatility) is supported via classifier-free
guidance, giving two variants that are compared throughout:

- **SGM** — unconditional score-based generative model.
- **SGCM** — conditional model, conditioned on the trailing-vol regime.

Because the data-generating process is a known DCC-GARCH, the true conditional
covariance `H_t` is available as an oracle, so distributional fidelity can be
measured directly rather than against a proxy.

## Setup

```bash
pip install -r requirements.txt
```

Raw data is regenerated from public sources. Place these in `data/raw/`:

- `49_Industry_Portfolios_Daily.csv` (Kenneth French data library, value-weighted)
- `F-F_Research_Data_5_Factors_2x3_daily.csv` (Kenneth French data library)
- `DGS10.csv` (FRED 10-year Treasury yield)

## Data pipeline

The training data is a simulated DCC-GARCH covariance path calibrated to the
industry returns:

```bash
python src/sim/diagnostics.py     # (optional) stationarity / GARCH-suitability checks
python src/sim/dcc.py             # fit + simulate -> data/sim_cov.npy, data/sim_returns.npy
python src/sim/prepare_sim.py     # -> data/processed/{Cov_sim,C_sim,cond_sim}.pt
```

`prepare_sim.py` writes the covariance/correlation tensors plus a strictly-causal
trailing-market-vol conditioning series that never sees `H_t`.

## Running

```bash
python -m src.train_baseline      # multi-seed sweep: SGM + SGCM, mean ± std tables
python -m src.regime_stress       # portfolio stress-test application (reuses checkpoints)
```

`train_baseline.py` runs every `(variant, seed)` combination, then writes the
part-1 result tables (stylized-fact values, regime-specific W1, variance/scale)
alongside Ledoit-Wolf and simulated-training/validation reference rows. Loss curves,
real-vs-generated diagnostics and sample-matrix heatmaps go to `results/`;
checkpoints (with EMA weights) to `checkpoints/`. Both directories are created on
demand and git-ignored.

`regime_stress.py` reuses the trained checkpoints, with no new training, to compare
the distribution of portfolio volatility `sqrt(wᵀΣw)` per volatility regime against
the DGP oracle and the Ledoit-Wolf / sample-covariance baselines.

## Layout

| Path | Contents |
|------|----------|
| `data/loaders.py` | Kenneth French daily CSV loader |
| `src/sim/diagnostics.py` | stationarity / GARCH-suitability diagnostics + return panel |
| `src/sim/dcc.py` | DCC-GARCH calibration and simulation |
| `src/sim/prepare_sim.py` | simulated path → processed `.pt` tensors + conditioning |
| `src/data/dataset.py` | matrix-log dataset, normaliser, blocked/contiguous split |
| `src/diffusion/sde.py` | variance-preserving SDE |
| `src/diffusion/losses.py` | denoising score-matching loss (+ CFG dropout) |
| `src/diffusion/solver.py` | reverse-time sampler (+ classifier-free guidance) |
| `src/models/logcov_gnn.py` | transformer score network over the matrix log |
| `src/models/embeddings.py` | sinusoidal time + conditioning embeddings |
| `src/evaluation/evaluate.py` | Wasserstein metrics, stylized-fact + regime diagnostics, LW baseline |
| `src/train_baseline.py` | training driver + multi-seed result tables |
| `src/regime_stress.py` | regime-conditional portfolio stress-test application |
| `src/train_utils.py` | EMA, data cycling, loss plotting |

## Configuration

`train_baseline.py` has a top-of-file `CFG` dict plus `SOURCE` (`"sim"` /
`"empirical"`), `TARGET` (`"covariance"` / `"correlation"`), and `SPLIT`
(`"blocked"` / `"contiguous"`) constants. Set `cond_dim=0` for the unconditional
SGM and `1` for the trailing-vol-conditioned SGCM; the `VARIANTS` dict runs both.
