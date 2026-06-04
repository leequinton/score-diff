from __future__ import annotations

import contextlib

import matplotlib.pyplot as plt
import pandas as pd


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    @contextlib.contextmanager
    def swap_in(self, model):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[n])
        try:
            yield
        finally:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.data.copy_(backup[n])


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def plot_losses(csv_path, out_path):
    df = pd.read_csv(csv_path)
    val = df.dropna(subset=["val_loss"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["step"], df["train_loss"], alpha=0.7, label="train")
    if len(val):
        ax.plot(val["step"], val["val_loss"], marker="o", label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
