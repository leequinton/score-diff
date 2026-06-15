from __future__ import annotations

import contextlib
import csv
import math
import time
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
    SUFFIX = "_sim"
    GAP    = 300
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
    n_layers=5,
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
    n_samples=3000,        # ~1000 per regime bin (3 bins) so the per-regime W1 isn't
                           #   sample-starved; with the FACT_SUBSAMPLE=1000 cap this also
                           #   matches DM/CDM per-bin counts (both cap to 1000)
    eps_t=1e-3,
    seed=42,
    cond_dim=1,  # sim: trailing market vol (1-D). empirical: [equity vol, rate vol]. 0 disables
    cond_dropout=0.1,
    guidance_scale=0.0,
)

# multi-seed sweep: run every (variant, seed) and report mean +- std across seeds.
# DM = unconditional, CDM = conditional (the reference paper's two columns).
SEEDS = (0, 1, 2)
VARIANTS = {
    "DM":  {"cond_dim": 0},
    "CDM": {"cond_dim": 1},
}


def main(cfg=CFG):
    seed = cfg["seed"]
    cond_dim = cfg.get("cond_dim", 0)
    use_cond = cond_dim > 0
    paths = output_paths(seed, cond_dim)
    torch.manual_seed(seed)
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
    best_ema_shadow = None

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
        plot_sample_matrices(Cov_real, Sigma_gen, paths["samples"], kind="covariance")
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

    return {**stats, "best_val_loss": best_val_loss, "best_step": best_step}


def aggregate_runs(runs):
    """runs: list of per-seed stats dicts. -> {metric: (mean, std)} over the seeds,
    for the metrics present in every run."""
    if not runs:
        return {}
    keys = set(runs[0]).intersection(*[set(r) for r in runs[1:]])
    return {k: (float(np.mean([r[k] for r in runs])),
               float(np.std([r[k] for r in runs]))) for k in sorted(keys)}


def run_experiments(seeds=SEEDS, variants=VARIANTS):
    """Run every (variant, seed) combination and write a mean +- std table across
    seeds, one column per variant -- the DM (unconditional) vs CDM (conditional)
    comparison. `variants` maps a column label to a cfg override (e.g. cond_dim)."""
    agg = {}
    for label, override in variants.items():
        runs = []
        for seed in seeds:
            print(f"\n===== variant={label}  seed={seed} =====")
            runs.append(main({**CFG, "seed": seed, **override}))
        agg[label] = aggregate_runs(runs)

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
    return agg


if __name__ == "__main__":
    run_experiments()
