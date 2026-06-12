from __future__ import annotations

import contextlib
import csv
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import make_logcov_datasets
from src.diffusion.losses import vpsde_dsm_loss_logcov
from src.diffusion.sde import VPSDE
from src.diffusion.solver import sample_logcov
from src.evaluation.evaluate import (
    logcov_to_correlation, logcov_to_covariance, eval_and_plot,
    plot_sample_matrices, gmvp_diagnostics, regime_sweep_summary,
    variance_diagnostics,
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
    # one long, highly autocorrelated path -> bigger train/val gap, no conditioning series
    SUFFIX = "_sim"
    GAP    = 1000
    COR_FILE, COV_FILE, COND_FILE = "C_sim", "Cov_sim", None
else:
    raise ValueError(f"unknown SOURCE {SOURCE!r}")

TAG       = "_cov" if TARGET == "covariance" else ""
DATA_FILE = COV_FILE if TARGET == "covariance" else COR_FILE
DATA_PATH = ROOT / "data" / "processed" / f"{DATA_FILE}.pt"
C_PATH    = ROOT / "data" / "processed" / f"{COR_FILE}.pt"   # correlation eval target
COV_PATH  = ROOT / "data" / "processed" / f"{COV_FILE}.pt"   # covariance target
COND_PATH = (ROOT / "data" / "processed" / f"{COND_FILE}.pt") if COND_FILE else None


def output_paths(seed):
    return dict(
        ckpt        = ROOT / "checkpoints" / f"logcov_score{SUFFIX}{TAG}_seed{seed}.pt",
        plot        = ROOT / "results"     / f"real_vs_generated_baseline{SUFFIX}{TAG}_seed{seed}.png",
        samples     = ROOT / "results"     / f"sample_matrices{SUFFIX}{TAG}_seed{seed}.png",
        log         = ROOT / "results"     / f"losses_baseline{SUFFIX}{TAG}_seed{seed}.csv",
        loss_plot   = ROOT / "results"     / f"losses_baseline{SUFFIX}{TAG}_seed{seed}.png",
        regime_plot = ROOT / "results"     / f"regime_sweep_baseline{SUFFIX}{TAG}_seed{seed}.png",
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
    n_steps=30_000,        # total optimizer steps (dataset-size-independent; anchors the LR cosine)
    val_every=1_000,       # validate every k steps
    log_every=100,         # in steps (logging granularity, not the train schedule)
    ema_decay=0.999,
    sde_steps=1000,
    n_samples=500,
    eps_t=1e-3,
    seed=42,
    cond_dim=0,  # [equity vol, rate vol]; set 0 to disable, 1 for equity-only ablation, 2 both
    cond_dropout=0.1,
    guidance_scale=0.0,
)


def main(cfg=CFG):
    seed = cfg["seed"]
    paths = output_paths(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond_dim = cfg.get("cond_dim", 0)
    use_cond = cond_dim > 0
    print(f"device: {device}  seed: {seed}  cond_dim: {cond_dim}")

    train_ds, val_ds, norm = make_logcov_datasets(
        DATA_PATH, gap=GAP, cond_path=COND_PATH if use_cond else None, target=TARGET,
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
        val_cond = val_ds.cond.to(device)
        n_rep = (cfg["n_samples"] + len(val_cond) - 1) // len(val_cond)
        sample_cond = val_cond.repeat(n_rep, 1)[:cfg["n_samples"]]

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
    C_real = C_real_all[-len(val_ds):]

    paths["plot"].parent.mkdir(parents=True, exist_ok=True)
    stats = eval_and_plot(C_real, C_gen, paths["plot"], n_inv_gen=n_inv)

    # same as train.py only if target set to covariance
    if TARGET == "covariance":
        Sigma_gen = logcov_to_covariance(S_gen)
        Cov_real = torch.load(COV_PATH, weights_only=True).float()[-len(val_ds):]
        plot_sample_matrices(Cov_real, Sigma_gen, paths["samples"], kind="covariance")
        stats.update(variance_diagnostics(Cov_real, Sigma_gen))
        stats.update(gmvp_diagnostics(Cov_real, Sigma_gen))
    else:
        plot_sample_matrices(C_real, C_gen, paths["samples"], kind="correlation")
    print(f"saved samples -> {paths['samples']}")

    print(f"saved plot -> {paths['plot']}")
    for k, v in stats.items():
        print(f"  {k:>22s}: {v:.4f}")

    if use_cond:
        print()
        sweep_levels_norm = [-1.28, 0.0, 1.28]
        sweep_n = max(64, cfg["n_samples"] // 4)
        C_gen_list, cond_raw_list = [], []
        with ema.swap_in(model):
            model.eval()
            for c_norm in sweep_levels_norm:
                c_tensor = torch.full((sweep_n, cond_dim), c_norm, device=device)
                S_g = sample_logcov(model, sde, sweep_n, n_assets,
                                    device=device, eps_t=cfg["eps_t"],
                                    cond=c_tensor, guidance_scale=0.0)
                C_, _ = logcov_to_correlation(norm.denormalize(S_g).cpu())
                c_raw = norm.denormalize_cond(c_tensor[:1])[0, 0].item()
                C_gen_list.append(C_)
                cond_raw_list.append(c_raw)
        sweep_rows = regime_sweep_summary(C_gen_list, cond_raw_list,
                                          save_path=paths["regime_plot"],
                                          label="(baseline, w=0)")
        print(f"saved regime sweep -> {paths['regime_plot']}")
        print("regime sweep (matched cond → output stats):")
        for r in sweep_rows:
            print(f"  cond={r['cond']:6.2f}  mean_offdiag={r['mean_offdiag']:6.4f}  "
                  f"top_eig={r['top_eig']:6.4f}  n_valid={r['n_valid']}")

    return {**stats, "best_val_loss": best_val_loss, "best_step": best_step}


if __name__ == "__main__":
    main()
