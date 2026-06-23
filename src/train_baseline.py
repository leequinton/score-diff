from __future__ import annotations

import contextlib
import csv
import math
import time
import gc
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import make_logcov_datasets
from src.diffusion.losses import vpsde_dsm_loss_logcov
from src.diffusion.sde import VPSDE
from src.diffusion.solver import sample_logcov
from src.evaluation.evaluate import (
    logcov_to_correlation, logcov_to_covariance, eval_and_plot,
    plot_sample_matrices, variance_diagnostics, regime_binned_eval,
    w1_facts, to_correlation, rolling_lw_covariances, _subsample,
)
from src.models.logcov_gnn import LogCovScoreGNN
from src.train_utils import EMA, cycle, plot_losses

ROOT = Path(__file__).resolve().parents[1]

# --- data source -------------------------------------------------------------
# "empirical" : rolling-window matrices from the FF industry data
# "sim"       : the calibrated DCC-GARCH covariance path (src/sim/dcc.py -> prepare_sim)
SOURCE = "sim"

# "correlation" diffuses the matrix log of correlation matrices (the standard baseline)
# "covariance" diffuses the matrix log of covariance matrices, capturing per-asset scale.
# GMVP needs the scales + Sigma^-1, so the portfolio application uses "covariance".
TARGET = "covariance"

if SOURCE == "empirical":
    SUFFIX = "_daily"
    GAP    = 12
    COR_FILE, COV_FILE, COND_FILE = "C_empirical_daily", "Cov_empirical_daily", "cond_daily"
elif SOURCE == "sim":
    # one long, highly autocorrelated path -> bigger train/val gap. cond_sim is the
    # causal trailing market vol from prepare_sim (set cond_dim=0 below to ignore it).
    # GAP=500 exceeds the path's empirical decorrelation length: the ACF of realized
    # variance / top eigenvalue falls below 0.05 by ~460 steps, so a 500-step embargo
    # makes train and val effectively independent at each block seam (300 left ~0.2).
    SUFFIX = "_sim"
    GAP    = 500
    COR_FILE, COV_FILE, COND_FILE = "C_sim", "Cov_sim", "cond_sim"
else:
    raise ValueError(f"unknown SOURCE {SOURCE!r}")

# train/val split. "blocked" scatters N_VAL_BLOCKS val blocks across the path with a
# `gap` embargo each side, so train and val share the same regime mixture -- the right
# eval for a distributional-fidelity claim (vs "contiguous", which tests regime
# extrapolation). NOTE the embargo cost: each block discards ~2*GAP steps from train,
# so with many blocks reduce GAP (ACF-justified) or N_VAL_BLOCKS to avoid starving train.
SPLIT        = "blocked"
N_VAL_BLOCKS = 10

TAG       = "_cov" if TARGET == "covariance" else ""
DATA_FILE = COV_FILE if TARGET == "covariance" else COR_FILE
DATA_PATH = ROOT / "data" / "processed" / f"{DATA_FILE}.pt"
C_PATH    = ROOT / "data" / "processed" / f"{COR_FILE}.pt"   # correlation eval target
COV_PATH  = ROOT / "data" / "processed" / f"{COV_FILE}.pt"   # covariance target
COND_PATH = (ROOT / "data" / "processed" / f"{COND_FILE}.pt") if COND_FILE else None


def output_paths(seed, cond_dim=0):
    ctag = "_cond" if cond_dim > 0 else "_uncond"      # keep DM/CDM run outputs separate
    stem = f"{SUFFIX}{TAG}{ctag}_seed{seed}"
    return dict(
        ckpt      = ROOT / "checkpoints" / f"logcov_score{stem}.pt",
        plot      = ROOT / "results"     / f"real_vs_generated_baseline{stem}.png",
        samples   = ROOT / "results"     / f"sample_matrices{stem}.png",
        log       = ROOT / "results"     / f"losses_baseline{stem}.csv",
        loss_plot = ROOT / "results"     / f"losses_baseline{stem}.png",
    )

CFG = dict(
    hidden_dim=128,
    n_layers=4,
    n_heads=4,
    dropout=0.0,
    batch_size=64,
    lr=2e-4,
    weight_decay=1e-5,
    lr_schedule="cosine",  # "cosine" (linear warmup -> cosine decay) or "constant"
    warmup_frac=0.05,      # fraction of total steps spent linearly warming up
    min_lr_ratio=0.05,     # cosine floor, as a fraction of peak lr
    n_steps=20_000,        # total optimizer steps (dataset-size-independent; anchors the LR cosine)
    val_every=1_000,       # validate every k steps
    log_every=100,         # in steps (logging granularity, not the train schedule)
    ema_decay=0.999,
    sde_steps=1000,
    n_samples=3000,        # ~1000 per regime bin (3 bins)
    eps_t=1e-3,
    seed=42,
    cond_dim=1,  # sim: trailing market vol (1-D). empirical: [equity vol, rate vol]. 
    cond_dropout=0.1,
    guidance_scale=0.0,
)

# multi-seed sweep: run every (variant, seed) and report mean +- std across seeds.
# SGM  = unconditional score-based generative model (no regime conditioning).
# SGCM = conditional score-based generative model (trailing-vol conditioning).
SEEDS = (0, 1, 2, 3, 4)
VARIANTS = {
    "SGM":  {"cond_dim": 0},
    "SGCM": {"cond_dim": 1},
}

# Ledoit-Wolf baseline: trailing window (steps) for the rolling OOS covariance
# estimate, and the number of bootstrap-resample repeats used to give the LW row a
# W1 band comparable to the other columns. 252 ~ one trading year; window/N ~ 5 at
# N=49, a realistic estimation horizon rather than a rigged one.
LW_WINDOW = 252
LW_TRIALS = 10


def main(cfg=CFG):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cond_dim = cfg.get("cond_dim", 0)
    use_cond = cond_dim > 0
    paths = output_paths(seed, cond_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  seed: {seed}  cond_dim: {cond_dim}")

    train_ds, val_ds, norm = make_logcov_datasets(
        DATA_PATH, gap=GAP, cond_path=COND_PATH if use_cond else None, target=TARGET,
        split=SPLIT, n_val_blocks=N_VAL_BLOCKS,
    )
    n_assets = train_ds.X.shape[-1]
    print(f"train: {len(train_ds)}  val: {len(val_ds)}  N={n_assets}  target={TARGET}")

    norm = norm.to(device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"])

    # Training schedule is specified directly in optimizer steps so the compute
    # budget (and the LR cosine horizon) is fixed independent of dataset size T.
    # That keeps the overfitting study clean: vary T at fixed steps to isolate the
    # effect of data quantity (per-sample revisits = steps*batch/T) rather than
    # confounding it with more gradient updates. steps_per_epoch is kept only for
    # the epoch-denominated log line.
    steps_per_epoch = len(train_loader)
    n_steps = cfg["n_steps"]
    val_every = cfg["val_every"]
    log_every = cfg["log_every"]
    print(f"total steps: {n_steps}  steps/epoch: {steps_per_epoch}  "
          f"(~{n_steps / steps_per_epoch:.1f} epochs)  val every: {val_every} steps")

    model = LogCovScoreGNN(
        n_assets=n_assets,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        cond_dim=cond_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    sde = VPSDE(N=cfg["sde_steps"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Linear warmup -> cosine decay over the full run (n_steps is known up front).
    # "constant" leaves lr flat, so the schedule is a clean on/off ablation.
    warmup_steps = max(1, int(cfg.get("warmup_frac", 0.0) * n_steps))
    min_ratio = cfg.get("min_lr_ratio", 0.0)

    def lr_lambda(step):
        if cfg.get("lr_schedule", "constant") != "cosine":
            return 1.0
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, cfg["ema_decay"])

    best_val_loss = float("inf")
    best_step = 0
    best_ema_shadow = None      # set to a clone of ema.shadow each time val improves

    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    with paths["log"].open("w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["step", "train_loss", "val_loss"])

        train_iter = cycle(train_loader)
        t0 = time.time()
        cond_dropout_train = cfg.get("cond_dropout", 0.1) if use_cond else 0.0
        for step in range(1, n_steps + 1):
            batch = next(train_iter)
            if use_cond:
                X0, cond0 = batch[0].to(device), batch[1].to(device)
            else:
                X0, cond0 = batch.to(device), None
            loss = vpsde_dsm_loss_logcov(model, sde, X0, eps_t=cfg["eps_t"],
                                         cond=cond0, cond_dropout=cond_dropout_train)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ema.update(model)

            val_loss_avg = None
            if step % val_every == 0:
                with ema.swap_in(model), torch.no_grad():
                    vlosses = []
                    for vb in val_loader:
                        if use_cond:
                            Xv, condv = vb[0].to(device), vb[1].to(device)
                        else:
                            Xv, condv = vb.to(device), None
                        vlosses.append(vpsde_dsm_loss_logcov(model, sde, Xv,
                                                             eps_t=cfg["eps_t"],
                                                             cond=condv, cond_dropout=0.0).item())
                val_loss_avg = sum(vlosses) / len(vlosses)

                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_step = step
                    best_ema_shadow = {n: v.detach().clone() for n, v in ema.shadow.items()}

            if step % log_every == 0:
                print(f"epoch {step/steps_per_epoch:5.1f}  step {step:>6d}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"train_loss {loss.item():.4f}  ({(time.time()-t0)/step:.3f}s/step)")
                if val_loss_avg is not None:
                    marker = "  **new best**" if step == best_step else ""
                    print(f"           val_loss   {val_loss_avg:.4f}{marker}")
                log_writer.writerow([step, f"{loss.item():.6f}",
                                     f"{val_loss_avg:.6f}" if val_loss_avg is not None else ""])
                log_file.flush()

    plot_losses(paths["log"], paths["loss_plot"])
    print(f"saved loss plot -> {paths['loss_plot']}")

    if best_ema_shadow is not None:
        ema.shadow = best_ema_shadow
        print(f"using best EMA from step {best_step}  (val_loss {best_val_loss:.4f})")
    else:
        print("No best val recorded, using final step weights")

    if use_cond and model.cond_embedding is not None:
        gate = ema.shadow["cond_gate"].detach().cpu()
        print(f"cond_gate (ema) per injection point: {gate.numpy().round(4)}")
        print(f"cond_gate (ema) abs-mean: {gate.abs().mean().item():.4f}")  # to see whether conditioning gate is precedent

    paths["ckpt"].parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "ema": ema.shadow,
        "best_step": best_step,
        "best_val_loss": best_val_loss,
        "cfg": cfg,
    }, paths["ckpt"])
    print(f"saved checkpoint -> {paths['ckpt']}")

    sample_cond = None
    if use_cond:
        # Draw a REPRESENTATIVE sample of the val regime distribution. (The old
        # val_cond[:n_samples] took the first matrices in time order -> one early
        # block, skewing the conditioning toward whatever regime that slice is in.)
        # A fixed generator keeps the conditioning identical across seeds, so the
        # per-regime numbers reflect model variance, not cond-draw variance.
        val_cond = val_ds.cond
        n = cfg["n_samples"]
        if n <= len(val_cond):
            g = torch.Generator().manual_seed(0)
            sel = torch.randperm(len(val_cond), generator=g)[:n]
            sample_cond = val_cond[sel].to(device)
        else:
            n_rep = (n + len(val_cond) - 1) // len(val_cond)
            sample_cond = val_cond.repeat(n_rep, 1)[:n].to(device)

    print(f"sampling {cfg['n_samples']} matrices "
          f"(cond={'matched-val' if use_cond else 'unconditional'}, "
          f"guidance_scale={cfg.get('guidance_scale', 0.0)})...")
    with ema.swap_in(model):
        model.eval()
        S_gen_n = sample_logcov(
            model, sde, cfg["n_samples"], n_assets,
            device=device, eps_t=cfg["eps_t"],
            cond=sample_cond, guidance_scale=cfg.get("guidance_scale", 0.0),
        )

    S_gen = norm.denormalize(S_gen_n).cpu()
    C_gen, n_inv = logcov_to_correlation(S_gen)

    C_real_all = torch.load(C_PATH, weights_only=True).float()
    C_real = C_real_all[val_ds.idx]                    # gather val matrices (split-agnostic)

    paths["plot"].parent.mkdir(parents=True, exist_ok=True)
    stats = eval_and_plot(C_real, C_gen, paths["plot"], n_inv_gen=n_inv)

    # same as train.py only if target set to covariance
    if TARGET == "covariance":
        Sigma_gen = logcov_to_covariance(S_gen)
        Cov_all = torch.load(COV_PATH, weights_only=True).float()
        Cov_real = Cov_all[val_ds.idx]                    # val split (split-agnostic gather)
        Cov_train = Cov_all[train_ds.idx]                 # train split
        # covariance + correlation of the same sampled matrices (model emits both)
        plot_sample_matrices(Cov_real, Sigma_gen, paths["samples"], show_cov=True)
        stats.update(variance_diagnostics(Cov_real, Sigma_gen, Sigma_train=Cov_train))

        # conditional-fidelity eval: bin by trailing-vol regime (raw cond, available
        # for DM and CDM alike) and compare real vs gen within each bin. CDM feeds the
        # per-sample cond it was drawn at; DM has none -> its marginal is judged per bin.
        if COND_PATH is not None:
            cond_val = torch.load(COND_PATH, weights_only=True).float().reshape(-1)[val_ds.idx]
            cond_gen = norm.denormalize_cond(sample_cond).reshape(-1).cpu() if use_cond else None
            stats.update(regime_binned_eval(cond_val, Cov_real, Sigma_gen, cond_gen=cond_gen))
    else:
        plot_sample_matrices(C_real, C_gen, paths["samples"], kind="correlation")
    print(f"saved samples -> {paths['samples']}")

    print(f"saved plot -> {paths['plot']}")
    for k, v in stats.items():
        print(f"  {k:>22s}: {v:.4f}")

    del model, sde, opt, ema, train_loader, val_loader, train_ds, val_ds  # free GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    return {**stats, "best_val_loss": best_val_loss, "best_step": best_step}


def aggregate_runs(runs):
    """runs: list of per-seed stats dicts. -> {metric: (mean, std)} over the seeds,
    for the metrics present in every run."""
    if not runs:
        return {}
    keys = set(runs[0]).intersection(*[set(r) for r in runs[1:]])
    return {k: (float(np.mean([r[k] for r in runs])),
               float(np.std([r[k] for r in runs]))) for k in sorted(keys)}


def _bootstrap(C, seed):
    """Resample a matrix batch with replacement to its own size (for the LW band)."""
    g = torch.Generator().manual_seed(seed)
    return C[torch.randint(len(C), (len(C),), generator=g)]


def distributional_fidelity_baselines(cfg=CFG):
    """Model-independent rows of the distributional-fidelity table, scored against
    the SAME fixed val reference as SGM/SGCM:
      * True -- independent DGP draws (data/sim_cov_seed*.npy from dcc.py). Each
        path is an i.i.d. realization of the calibrated law, so W1(draw, val) is the
        irreducible floor; reported mean +- std over the K paths (between-draw var).
      * LW   -- rolling Ledoit-Wolf covariance on the OOS returns at the val
        indices; mean +- std over LW_TRIALS bootstrap resamples (W1 sampling noise).
    Returns {label: {metric: (mean, std)}} for the labels that are computable."""
    if TARGET != "covariance":
        print("[dist-fidelity] baselines need TARGET='covariance'; skipping")
        return {}
    _, val_ds, _ = make_logcov_datasets(
        DATA_PATH, gap=GAP, cond_path=None, target=TARGET,
        split=SPLIT, n_val_blocks=N_VAL_BLOCKS,
    )
    Cov_real = torch.load(COV_PATH, weights_only=True).float()[val_ds.idx]
    C_real = torch.load(C_PATH, weights_only=True).float()[val_ds.idx]
    n = cfg["n_samples"]
    out = {}

    # --- True (DGP oracle): independent re-simulated H_t paths -----------------
    oracle_paths = sorted((ROOT / "data").glob("sim_cov_seed*.npy"))
    runs = []
    for p in oracle_paths:
        H = torch.from_numpy(np.load(p)).float()
        H = 0.5 * (H + H.transpose(-1, -2))            # enforce symmetry (fp drift)
        Sig = _subsample(H, n)                          # match the generated sample size
        C, _ = to_correlation(Sig)
        runs.append(w1_facts(C_real, C, Sigma_real=Cov_real, Sigma_gen=Sig))
    if runs:
        out["True"] = aggregate_runs(runs)
        print(f"[dist-fidelity] True row from {len(runs)} oracle path(s)")
    else:
        print("[dist-fidelity] no oracle paths (data/sim_cov_seed*.npy) -- run dcc.py "
              "with ORACLE_SEEDS to populate the True row; skipping it for now")

    # --- LW baseline: rolling Ledoit-Wolf at the val indices -------------------
    returns = np.load(ROOT / "data" / "sim_returns.npy")
    lw_cov, keep = rolling_lw_covariances(returns, val_ds.idx.numpy(), LW_WINDOW)
    var_real = torch.diagonal(Cov_real, dim1=-2, dim2=-1).flatten()  # val ref (constant)
    runs = []
    for t in range(LW_TRIALS):
        Sig = _bootstrap(lw_cov, t)
        C, _ = to_correlation(Sig)
        r = w1_facts(C_real, C, Sigma_real=Cov_real, Sigma_gen=Sig)
        v = torch.diagonal(Sig, dim1=-2, dim2=-1).flatten()
        # variance levels so the LW column of the scale table is self-contained
        r.update(var_mean_gen=float(v.mean()), var_median_gen=float(v.median()),
                 var_mean_real=float(var_real.mean()), var_median_real=float(var_real.median()))
        runs.append(r)
    out["LW"] = aggregate_runs(runs)
    print(f"[dist-fidelity] LW row from {len(lw_cov)} trailing-window estimates "
          f"(window={LW_WINDOW}, {len(val_ds.idx) - len(keep)} early val idx dropped)")
    return out


def check_true_variance(cfg=CFG):
    """TEMP diagnostic (no training): does the DGP oracle ALSO overshoot val in
    scale? If so, val is a low-variance finite draw rather than the model being
    biased. Prints the True (oracle) variance W1 to val and the val/train/oracle
    variance levels. Reads only the oracle .npy paths and the val split, so it runs
    independently of any model. On Colab:

        python -c "import src.train_baseline as t; t.check_true_variance()"
    """
    from scipy.stats import wasserstein_distance
    from src.data.dataset import _split_indices

    Cov_all = torch.load(COV_PATH, weights_only=True).float()
    train_idx, val_idx = _split_indices(len(Cov_all), 0.2, GAP, SPLIT, N_VAL_BLOCKS)
    diag = lambda S: torch.diagonal(S, dim1=-2, dim2=-1).flatten().numpy()
    var_val, var_train = diag(Cov_all[val_idx]), diag(Cov_all[train_idx])
    n = cfg["n_samples"]

    oracle_paths = sorted((ROOT / "data").glob("sim_cov_seed*.npy"))
    if not oracle_paths:
        print("[true-var] no oracle paths (data/sim_cov_seed*.npy); run dcc.py first")
        return
    w1s, means, medians = [], [], []
    for p in oracle_paths:
        H = torch.from_numpy(np.load(p)).float()
        H = 0.5 * (H + H.transpose(-1, -2))
        v = diag(_subsample(H, n))                      # match the generated sample size
        w1s.append(float(wasserstein_distance(var_val, v)))
        means.append(float(v.mean())); medians.append(float(np.median(v)))

    print(f"[true-var] oracle paths: {len(oracle_paths)}   n_subsample={n}")
    print(f"  val     mean={var_val.mean():.3f}  median={np.median(var_val):.3f}")
    print(f"  train   mean={var_train.mean():.3f}  median={np.median(var_train):.3f}")
    print(f"  oracle  mean={np.mean(means):.3f}  median={np.mean(medians):.3f}")
    print(f"  TRUE w1_variance (oracle vs val) = {np.mean(w1s):.4f} +/- {np.std(w1s):.4f}")
    print(f"  reference (your run): SGM 0.389  SGCM 0.440  LW 0.177")


def _fmt_cell(d):
    """(mean, std) -> 'mean ± std', or blank for a missing/None cell."""
    if d is None:
        return f"{'':^16s}"
    return f"{d[0]:8.4f} ± {d[1]:6.4f}"


def _write_csv(path, head_label, head_cols, rows):
    """rows: list of (label, [cell_or_None per head_col]); each cell (mean,std)."""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([head_label] + [f"{c}_mean" for c in head_cols] + [f"{c}_std" for c in head_cols])
        for label, cells in rows:
            w.writerow([label]
                       + ["" if c is None else f"{c[0]:.6f}" for c in cells]
                       + ["" if c is None else f"{c[1]:.6f}" for c in cells])


def _print_table(head_label, head_cols, rows):
    print(f"{head_label:>24s} | " + " | ".join(f"{c:^16s}" for c in head_cols))
    for label, cells in rows:
        print(f"{label:>24s} | " + " | ".join(_fmt_cell(c) for c in cells))


# --- Table 1: stylized facts (W1 to val), True / SGM / SGCM / LW ---------------
FACTS_ROWS = [
    ("w1_offdiag",     "off-diagonal correlation"),
    ("w1_gini",        "eigenvalue Gini"),
    ("w1_coph_single", "cophenetic (single)"),
    ("w1_coph_ward",   "cophenetic (ward)"),
    ("w1_perron",      "Perron-Frobenius"),
    ("w1_powerlaw",    "eigenvalue power-law"),
]


def write_facts_table(agg):
    """Table 1 -- stylized-fact W1 to the val reference, one column per model.
    Read SGM/SGCM against True (the floor they should approach) and LW (baseline)."""
    cols = [c for c in ("True", "SGM", "SGCM", "LW") if c in agg]
    rows = [(label, [agg[c].get(key) for c in cols]) for key, label in FACTS_ROWS]
    out = ROOT / "results" / f"facts{SUFFIX}{TAG}.csv"
    _write_csv(out, "stylized_fact", cols, rows)
    print(f"\n[Table 1] stylized facts (W1 to val) -> {out}")
    _print_table("stylized fact", cols, rows)


# --- Table 2: regime-specific W1, SGM vs SGCM ----------------------------------
REGIME_BINS = [("regime_pooled_", "pooled"), ("regime_lo_", "lo"),
               ("regime_mid_", "mid"), ("regime_hi_", "hi")]


def write_regime_table(agg):
    """Table 2 -- per-regime W1 (off-diag + variance) for SGM vs SGCM. SGM is the
    regime-blind marginal-vs-bin baseline; SGCM the conditional bin-vs-bin."""
    models = [c for c in ("SGM", "SGCM") if c in agg]
    if not models:
        return
    cols = [(f"{mdl} {mlbl}", mdl, suf)
            for suf, mlbl in (("w1_offdiag", "offdiag"), ("w1_variance", "var"))
            for mdl in models]
    headers = [h for h, _, _ in cols]
    rows = [(blabel, [agg[m].get(pre + suf) for _, m, suf in cols])
            for pre, blabel in REGIME_BINS]
    out = ROOT / "results" / f"regime{SUFFIX}{TAG}.csv"
    _write_csv(out, "regime", headers, rows)
    print(f"\n[Table 2] regime-specific W1 -> {out}")
    _print_table("regime", headers, rows)


# --- Table 3: variance / scale (real vs generated) -----------------------------
VAR_ROWS = [("mean variance", "mean"), ("median variance", "median"),
            ("W1 variance (to val)", "w1")]


def write_variance_table(agg):
    """Table 3 -- pooled per-asset variance: real-vs-generated levels (mean,
    median) and W1-to-val. No True column: the DGP oracle's scale is the real law
    by construction, so the meaningful comparison is real vs each generator."""
    model_cols = [c for c in ("SGM", "SGCM", "LW") if c in agg]
    ref = next((agg[c] for c in model_cols if "var_mean_real" in agg[c]), None)
    if ref is None:
        return
    cols = ["real"] + model_cols

    def cell(col, kind):
        if col == "real":
            return None if kind == "w1" else ref.get(f"var_{kind}_real")
        if kind == "w1":
            return agg[col].get("w1_variance")
        return agg[col].get(f"var_{kind}_gen")

    rows = [(label, [cell(c, kind) for c in cols]) for label, kind in VAR_ROWS]
    out = ROOT / "results" / f"variance{SUFFIX}{TAG}.csv"
    _write_csv(out, "metric", cols, rows)
    print(f"\n[Table 3] variance / scale (real vs generated) -> {out}")
    _print_table("metric", cols, rows)


def run_experiments(seeds=SEEDS, variants=VARIANTS):
    """Run every (variant, seed) combination, then assemble the part-1 result
    tables. Model columns (SGM unconditional vs SGCM conditional) are aggregated
    mean +- std across seeds; the model-independent True (DGP oracle) and LW
    baseline columns come from distributional_fidelity_baselines(). Writes the full
    per-metric model summary plus three focused tables: (1) stylized facts, (2)
    regime-specific W1, (3) variance/scale."""
    agg = {}
    for label, override in variants.items():
        runs = []
        for seed in seeds:
            print(f"\n===== variant={label}  seed={seed} =====")
            runs.append(main({**CFG, "seed": seed, **override}))
        agg[label] = aggregate_runs(runs)

    # full per-metric model summary (SGM/SGCM only -- the baselines carry W1 facts
    # only, so folding them into this dump would be mostly NaN).
    metrics = sorted(set().union(*[set(a) for a in agg.values()]))
    out = ROOT / "results" / f"summary{SUFFIX}{TAG}.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + [f"{lbl}_mean" for lbl in agg] + [f"{lbl}_std" for lbl in agg])
        for m in metrics:
            row = [m]
            row += [f"{agg[lbl].get(m, (float('nan'),))[0]:.6f}" for lbl in agg]
            row += [f"{agg[lbl].get(m, (float('nan'), float('nan')))[1]:.6f}" for lbl in agg]
            w.writerow(row)
    print(f"\nsaved summary table -> {out}")
    print(f"{'metric':>22s} | " + " | ".join(f"{lbl:^22s}" for lbl in agg))
    for m in metrics:
        cells = []
        for lbl in agg:
            mean, std = agg[lbl].get(m, (float("nan"), float("nan")))
            cells.append(f"{mean:9.4f} ± {std:7.4f}")
        print(f"{m:>22s} | " + " | ".join(cells))

    # part-1 result tables: add True + LW columns, then write the three focused tables.
    agg.update(distributional_fidelity_baselines())
    write_facts_table(agg)       # Table 1: stylized facts (W1 to val)
    write_regime_table(agg)      # Table 2: regime-specific W1 (SGM vs SGCM)
    write_variance_table(agg)    # Table 3: variance / scale (real vs generated)
    return agg


if __name__ == "__main__":
    run_experiments()
