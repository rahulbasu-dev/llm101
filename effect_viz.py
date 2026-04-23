"""Effect visualizations — how training hyperparameters shape the loss.

Renders schematic charts showing the *direction* of each hyperparameter's
influence, based on the standard ML literature and observed behavior on
small transformer models. Data is synthetic but follows realistic dynamics:

  max_epochs     — train drops monotonically, val forms a U (overfitting)
  batch_size     — larger batches = smoother gradients, fewer updates
  learning_rate  — too small = undertrain, too big = diverge, sweet spot
  dropout        — closes the train/val gap up to a point, then underfits
  warmup_steps   — shape of the linear-warmup + cosine-decay LR schedule

Each function writes a PNG to the given path and returns the path.
"""

from __future__ import annotations
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Consistent palette across all effect plots
_BLUE = "#08519c"
_RED = "#cb181d"
_ORANGE = "#e6550d"
_GREEN = "#238b45"
_GREY = "#999999"


# ───────────────────────────────────────────────────────────────
# Effect 1 — max_epochs
# ───────────────────────────────────────────────────────────────

def plot_epochs_effect(path: str) -> str:
    """Train loss drops monotonically; val loss forms a U → overfitting."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    epochs = np.arange(1, 31)
    # Train: smooth exponential decay (gets ever closer to minimum)
    train = 7.5 * np.exp(-epochs / 8) + 2.5 + np.random.default_rng(1).normal(0, 0.03, len(epochs))
    # Val: similar early, but flattens at epoch ~10 then rises (memorization)
    val_base = 7.5 * np.exp(-epochs / 9) + 3.2
    # Use maximum(..., 0) so we never raise a negative to a fractional power
    # (which would produce NaN and trigger a numpy RuntimeWarning).
    val_overfit = 0.015 * np.maximum(epochs - 10, 0) ** 1.4
    val = val_base + val_overfit + np.random.default_rng(2).normal(0, 0.03, len(epochs))

    ax.plot(epochs, train, "o-", color=_BLUE, linewidth=2.2, markersize=5,
            label="Train loss")
    ax.plot(epochs, val, "s-", color=_RED, linewidth=2.2, markersize=5,
            label="Val loss")

    # Best val
    best_idx = int(val.argmin())
    ax.scatter([epochs[best_idx]], [val[best_idx]], s=220, color=_RED,
               zorder=10, edgecolor="black", linewidth=1.6, marker="*",
               label=f"Best val @ epoch {epochs[best_idx]}")

    # Overfitting region shading
    ax.axvspan(12, 30, color="#fcbba1", alpha=0.25, label="Overfitting zone")
    ax.axvspan(1, 5, color="#c7e9c0", alpha=0.25)

    ax.text(3, 7.3, "under-\nfitting", ha="center", fontsize=9, color="#238b45",
            fontweight="bold")
    ax.text(21, 7.3, "over-\nfitting", ha="center", fontsize=9, color="#cb181d",
            fontweight="bold")
    ax.text(9, 7.3, "sweet spot", ha="center", fontsize=9, color="#08519c",
            fontweight="bold")

    ax.set_xlabel("max_epochs", fontsize=11)
    ax.set_ylabel("Cross-entropy loss", fontsize=11)
    ax.set_title("Effect of max_epochs on training\n"
                 "Train loss keeps dropping; val loss hits a minimum then rises",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 30)

    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


# ───────────────────────────────────────────────────────────────
# Effect 2 — batch_size
# ───────────────────────────────────────────────────────────────

def plot_batch_size_effect(path: str) -> str:
    """Larger batch = smoother gradient, fewer steps per epoch."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Left: loss curves for different batch sizes
    ax = axes[0]
    epochs_scale = 15
    sizes = [("bs=8",   0.28, _ORANGE, 0.25),
             ("bs=32",  0.22, _GREEN,  0.12),
             ("bs=64",  0.20, _BLUE,   0.06),
             ("bs=128", 0.19, "#6a51a3", 0.03)]
    x = np.linspace(0, epochs_scale, 240)
    for label, asymptote, color, noise in sizes:
        base = 7.5 * np.exp(-x / 5) + 2.5 + asymptote
        rng = np.random.default_rng(abs(hash(label)) % 2**31)
        y = base + rng.normal(0, noise, len(x))
        ax.plot(x, y, linewidth=1.6, color=color, alpha=0.85, label=label)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train loss", fontsize=11)
    ax.set_title("Gradient noise vs batch size", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(2.5, 8)

    # Right: memory cost
    ax2 = axes[1]
    bs_list = [8, 16, 32, 64, 128]
    mem_gb = [0.5 + 0.055 * b for b in bs_list]  # ~linear, rough approximation
    bars = ax2.bar([str(b) for b in bs_list], mem_gb, color=_BLUE, alpha=0.85)
    for b, m in zip(bars, mem_gb):
        ax2.text(b.get_x() + b.get_width() / 2, m + 0.2,
                 f"{m:.1f}GB", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("batch_size", fontsize=11)
    ax2.set_ylabel("VRAM (approx)", fontsize=11)
    ax2.set_title("Memory scales ~linearly", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Effect of batch_size on training",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


# ───────────────────────────────────────────────────────────────
# Effect 3 — learning_rate
# ───────────────────────────────────────────────────────────────

def plot_learning_rate_effect(path: str) -> str:
    """Classic LR sweep: too small = undertrain, too big = diverge."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    steps = np.linspace(0, 500, 300)

    configs = [
        ("lr=1e-5  (too small)",  _GREY,    lambda s: 8.0 - 0.8 * (1 - np.exp(-s / 2000))),
        ("lr=1e-4  (slow but converges)", _GREEN,
         lambda s: 8.0 - 4.5 * (1 - np.exp(-s / 300))),
        ("lr=3e-4  (optimal)",    _BLUE,
         lambda s: 8.0 - 5.5 * (1 - np.exp(-s / 100)) + 0.05 * np.sin(s / 20)),
        ("lr=3e-3  (diverges!)",  _RED,
         lambda s: np.where(s < 40, 8.0 - 4 * (s / 40),
                            4.0 + 6.0 * (1 - np.exp(-(s - 40) / 40)))),
    ]

    for label, color, fn in configs:
        y = fn(steps)
        y = np.clip(y, 1.5, 12)  # visual clip
        ax.plot(steps, y, linewidth=2.2, color=color, label=label, alpha=0.9)

    ax.axhline(y=3.0, color="black", linestyle="--", alpha=0.35, linewidth=1)
    ax.text(500, 3.0, "  typical\n  best val", fontsize=8, va="center", color="black")

    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Effect of learning_rate\n"
                 "Too small → undertrained.  Too big → training diverges.  "
                 "Sweet spot around 1e-4 to 3e-4 for AdamW.",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.5, 12)
    ax.set_xlim(0, 500)

    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


# ───────────────────────────────────────────────────────────────
# Effect 4 — dropout
# ───────────────────────────────────────────────────────────────

def plot_dropout_effect(path: str) -> str:
    """Dropout closes the train/val gap up to a point, then underfits."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    rates = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])

    # Final train loss: rises with dropout (regularization degrades fit)
    train_final = 2.8 + 1.5 * rates + 2.0 * rates ** 1.5
    # Final val loss: U-shape with minimum around dropout=0.25
    val_final = 4.5 - 3.0 * rates + 8.0 * rates ** 2

    ax.plot(rates, train_final, "o-", color=_BLUE, linewidth=2.2,
            markersize=7, label="Train loss (final)")
    ax.plot(rates, val_final, "s-", color=_RED, linewidth=2.2,
            markersize=7, label="Val loss (final)")
    ax.fill_between(rates, train_final, val_final, color="#fee5d9", alpha=0.5,
                    label="Generalization gap")

    # Best dropout
    best_i = int(val_final.argmin())
    ax.scatter([rates[best_i]], [val_final[best_i]], s=220, marker="*",
               color=_RED, edgecolor="black", linewidth=1.5, zorder=10,
               label=f"Best val @ dropout={rates[best_i]:.2f}")

    # Annotate regions
    ax.axvspan(0.0, 0.1, color="#fcbba1", alpha=0.25)
    ax.axvspan(0.4, 0.6, color="#fcbba1", alpha=0.25)
    ax.text(0.05, 5.8, "overfits", ha="center", fontsize=9,
            color=_RED, fontweight="bold")
    ax.text(0.5, 5.8, "underfits", ha="center", fontsize=9,
            color=_RED, fontweight="bold")
    ax.text(0.25, 5.8, "well-regularized", ha="center", fontsize=9,
            color=_BLUE, fontweight="bold")

    ax.set_xlabel("dropout rate", fontsize=11)
    ax.set_ylabel("Cross-entropy loss (final)", fontsize=11)
    ax.set_title("Effect of dropout on bias/variance\n"
                 "Low dropout overfits; high dropout underfits; the valley between is the sweet spot",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 0.62)

    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


# ───────────────────────────────────────────────────────────────
# Effect 5 — warmup_steps (the LR schedule shape)
# ───────────────────────────────────────────────────────────────

def plot_warmup_effect(path: str) -> str:
    """Schedule shape for different warmup_steps values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: schedule shapes
    ax = axes[0]
    total_steps = 480
    peak_lr = 3e-4
    min_lr = 0.1 * peak_lr

    def schedule(step, warmup, total, peak=peak_lr, floor_frac=0.1):
        if step < warmup:
            return peak * (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return peak * floor_frac + 0.5 * peak * (1 - floor_frac) * (1 + math.cos(math.pi * progress))

    steps_axis = np.arange(total_steps)

    configs = [
        ("warmup=200  (42% — the bad case)", 200, _RED, "-"),
        ("warmup=50  (10% — recommended)",   50,  _BLUE, "-"),
        ("warmup=32  (auto-scaled)",         32,  _GREEN, "-"),
    ]
    for label, warmup, color, style in configs:
        lrs = [schedule(s, warmup, total_steps) for s in steps_axis]
        ax.plot(steps_axis, lrs, linestyle=style, linewidth=2.2,
                color=color, label=label, alpha=0.9)
        # Mark where peak is reached
        ax.axvline(warmup, color=color, linestyle=":", alpha=0.4, linewidth=1)

    ax.axhline(peak_lr, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(total_steps * 0.99, peak_lr, "  peak_lr=3e-4",
            fontsize=9, va="center", ha="right", color="black")
    ax.axhline(min_lr, color="black", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(total_steps * 0.99, min_lr, "  min_lr=3e-5",
            fontsize=9, va="center", ha="right", color="black")

    ax.set_xlabel("Training step (total = 480)", fontsize=11)
    ax.set_ylabel("Learning rate", fontsize=11)
    ax.set_title("LR schedule: warmup → cosine decay",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_steps)

    # Right: resulting loss curves
    ax2 = axes[1]
    rng = np.random.default_rng(3)

    def synthetic_loss(warmup, total, peak):
        """Synthesize a loss curve assuming long peak-LR time = faster convergence."""
        effective_training = max(total - warmup, 1)
        rate = 5.5 / effective_training  # faster if more time at peak
        losses = []
        for s in range(total):
            lr_here = schedule(s, warmup, total, peak)
            # Loss drops when we're at meaningful LR
            if s == 0:
                losses.append(8.0)
            else:
                progress_factor = (lr_here / peak) ** 0.6
                delta = -rate * progress_factor * 20 * (losses[-1] - 2.5)
                losses.append(max(2.5, losses[-1] + delta))
        return np.array(losses) + rng.normal(0, 0.04, total)

    for label, warmup, color, style in configs:
        y = synthetic_loss(warmup, total_steps, peak_lr)
        ax2.plot(steps_axis, y, color=color, linewidth=1.8,
                 alpha=0.85, label=label)

    ax2.set_xlabel("Training step", fontsize=11)
    ax2.set_ylabel("Loss", fontsize=11)
    ax2.set_title("Resulting loss: shorter warmup = more peak-LR time = lower final loss",
                  fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Effect of warmup_steps on the LR schedule and final loss",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()
    return path


# ───────────────────────────────────────────────────────────────
# Orchestrator + captions
# ───────────────────────────────────────────────────────────────

_PLOTS = {
    "max_epochs":    (plot_epochs_effect,
                      "**Train loss drops monotonically. Val loss hits a minimum "
                      "then rises.** The gap between them is your overfitting "
                      "signal — the wider the gap, the more your model is "
                      "memorizing rather than generalizing. \"Best val\" gives you "
                      "the ideal number of epochs for this data.\n\n"
                      "**Rules of thumb:** on TinyShakespeare, expect sweet-spot "
                      "around epoch 8-12; by epoch 20+ overfitting dominates."),
    "batch_size":    (plot_batch_size_effect,
                      "**Larger batch = smoother gradient + fewer updates per "
                      "epoch + linear memory cost.** Small batches (bs=8) are "
                      "noisy but train in less wall-clock time; large batches "
                      "(bs=128) are stable but may need more epochs to match. "
                      "Final loss is roughly equal across reasonable batch sizes.\n\n"
                      "**Rules of thumb:** pick the largest batch that fits in "
                      "VRAM, then tune learning rate to match "
                      "(bigger batch → slightly higher LR)."),
    "learning_rate": (plot_learning_rate_effect,
                      "**Too small** → loss barely drops in your step budget. "
                      "**Too big** → loss diverges, possibly NaNs. "
                      "There's a sweet spot where convergence is fast and stable.\n\n"
                      "**Rules of thumb:** for AdamW with warmup + cosine "
                      "schedule, 1e-4 to 3e-4 works for most small transformers. "
                      "If loss explodes, halve the LR. If loss drops slowly, "
                      "double it."),
    "dropout":       (plot_dropout_effect,
                      "**Dropout is a bias-variance knob.** With dropout=0, "
                      "the model fits training data tightly but generalizes "
                      "poorly — train and val diverge. With high dropout "
                      "(>0.4), the model can't fit even the training data. "
                      "The valley between is where val loss is lowest.\n\n"
                      "**Rules of thumb:** start at 0.1 for small datasets, "
                      "0.2-0.3 if you see a growing train/val gap."),
    "warmup_steps":  (plot_warmup_effect,
                      "**The LR schedule is linear warmup → cosine decay.** "
                      "If `warmup_steps` is too large a fraction of total_steps "
                      "(the `200/480 = 42%` case you just hit), the model barely "
                      "gets any time at peak LR and learns slowly. Short warmup "
                      "gives the model more peak-LR training time.\n\n"
                      "**Rules of thumb:** warmup ≈ 5-10% of total steps. "
                      "The auto-scale in `train_iter` now enforces this "
                      "automatically when you leave the slider at 0."),
}


def render(param: str, outdir: str) -> tuple[str | None, str]:
    """Render one effect plot. Returns (image_path, caption_markdown)."""
    if param not in _PLOTS:
        return None, f"Unknown parameter: {param}"
    fn, caption = _PLOTS[param]
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"effect_{param}.png")
    if not os.path.exists(path):
        fn(path)
    return path, f"### {param}\n\n{caption}"


def render_all(outdir: str) -> dict[str, str]:
    """Pre-render all five effect plots. Returns dict of param → path."""
    paths = {}
    for p in _PLOTS:
        path, _ = render(p, outdir)
        paths[p] = path
    return paths


PARAMS = list(_PLOTS.keys())
