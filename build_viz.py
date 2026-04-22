"""Diagram generators for the 'Build Steps' tab of the Gradio UI.

These are pure-matplotlib drawings that don't need a trained model. They
complement the teach.py slides (which DO need a model) by rendering the
structural diagrams — sliding windows, transformer block, KV cache flow,
test matrix, etc.

Each function takes an output path and writes a PNG. Idempotent: if the
file exists, we skip re-rendering (callers can delete the dir to force).
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ───────────────────────────────────────────────────────────────
# Common plotting helpers
# ───────────────────────────────────────────────────────────────

_BLUE = "#08519c"
_BLUE_LT = "#c6dbef"
_ORANGE = "#e6550d"
_ORANGE_LT = "#fdd0a2"
_GREEN = "#238b45"
_GREEN_LT = "#c7e9c0"
_GREY = "#bbbbbb"


def _header(ax, title, subtitle=""):
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    if subtitle:
        ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=10,
                color="#555", style="italic")


def _box(ax, xy, w, h, text, face=_BLUE_LT, edge=_BLUE, fontsize=10,
         fontweight="bold", textcolor="black"):
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02",
                         facecolor=face, edgecolor=edge, linewidth=1.8)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=textcolor)


def _arrow(ax, xy0, xy1, color="black", lw=1.5, label=None):
    arr = FancyArrowPatch(xy0, xy1, arrowstyle="->", mutation_scale=14,
                          color=color, linewidth=lw)
    ax.add_patch(arr)
    if label:
        mx, my = (xy0[0] + xy1[0]) / 2, (xy0[1] + xy1[1]) / 2
        ax.text(mx, my + 0.02, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


# ───────────────────────────────────────────────────────────────
# Step 3 — Sliding-window dataset
# ───────────────────────────────────────────────────────────────

def draw_sliding_window(path):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _header(ax, "Sliding-window dataset",
            "Stride = seq_len / 2 by default (50% overlap). "
            "targets[t] = inputs[t+1] within each window.")
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 6)
    ax.axis("off")

    # Token stream
    for i in range(20):
        ax.add_patch(patches.Rectangle((i, 4.5), 1, 0.8,
                                       facecolor="#e6f2ff", edgecolor="#4a90d9"))
        ax.text(i + 0.5, 4.9, f"t{i}", ha="center", va="center", fontsize=9)
    ax.text(-0.3, 4.9, "tokens:", ha="right", va="center", fontsize=10, fontweight="bold")

    # Windows
    windows = [(0, 8, 3.2, _BLUE), (4, 12, 2.0, _ORANGE), (8, 16, 0.8, _GREEN)]
    for (start, end, ypos, color) in windows:
        ax.add_patch(patches.Rectangle((start, ypos), end - start, 0.8,
                                       facecolor="none", edgecolor=color,
                                       linewidth=2.5, linestyle="--"))
        ax.text(start - 0.2, ypos + 0.4, f"window", ha="right", va="center",
                fontsize=9, color=color, fontweight="bold")
        # Input range
        ax.text(start + (end - start) / 2, ypos + 0.4,
                f"input: t{start}..t{end-1}",
                ha="center", va="center", fontsize=9)
        # Target is shifted +1
        ax.text(start + (end - start) / 2, ypos - 0.35,
                f"target: t{start+1}..t{end}",
                ha="center", va="center", fontsize=8.5, color="#555")

    ax.text(10, -0.8,
            "Each (input, target) pair trains the model to predict the NEXT token at every position.",
            ha="center", fontsize=10, style="italic", color="#333")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 4 — Transformer block (shapes annotated)
# ───────────────────────────────────────────────────────────────

def draw_transformer_block(path):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(12, 6.5))
    _header(ax, "TransformerBlock — one of N=6 layers",
            "Pre-Norm: normalize BEFORE sublayer, add residual AFTER. "
            "All tensors are (B, T, d_model=384) unless shown.")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Input
    _box(ax, (1, 8.3), 2.2, 0.9, "x : (B, T, d_model)", face=_GREY, edge="black")

    # attn_norm
    _box(ax, (1, 7.0), 2.2, 0.8, "RMSNorm\n(attn_norm)", face=_BLUE_LT, edge=_BLUE)
    _arrow(ax, (2.1, 8.3), (2.1, 7.8))

    # Self-attention
    _box(ax, (1, 5.2), 2.2, 1.5,
         "CausalSelfAttention\n  qkv_proj → Q,K,V\n  RoPE(Q), RoPE(K)\n  softmax(Q·Kᵀ/√d)·V",
         face=_ORANGE_LT, edge=_ORANGE, fontsize=8.5)
    _arrow(ax, (2.1, 7.0), (2.1, 6.7))

    # Residual +
    _box(ax, (4, 6.5), 0.8, 0.8, "+", face="white", edge="black", fontsize=16)
    _arrow(ax, (3.2, 5.9), (4.0, 6.9), color=_ORANGE)   # from attn out
    _arrow(ax, (2.1, 8.8), (4.0, 7.1), color="#555", label="residual")  # skip connection

    # Same for FFN
    _box(ax, (6, 7.0), 2.4, 0.8, "RMSNorm\n(ffn_norm)", face=_BLUE_LT, edge=_BLUE)
    _arrow(ax, (4.8, 6.9), (6, 7.4))

    _box(ax, (6, 5.2), 2.4, 1.5,
         "FeedForward (SwiGLU)\n  silu(gate(x)) ⊙ up(x)\n  down_proj(...)",
         face=_GREEN_LT, edge=_GREEN, fontsize=8.5)
    _arrow(ax, (7.2, 7.0), (7.2, 6.7))

    _box(ax, (9.2, 6.5), 0.8, 0.8, "+", face="white", edge="black", fontsize=16)
    _arrow(ax, (8.4, 5.9), (9.2, 6.9), color=_GREEN)
    _arrow(ax, (4.8, 6.9), (9.2, 7.1), color="#555", label="residual")

    _box(ax, (10.8, 6.5), 2.4, 0.8, "output\n(B, T, d_model)", face=_GREY, edge="black", fontsize=9)
    _arrow(ax, (10.0, 6.9), (10.8, 6.9))

    # Footer facts
    ax.text(7, 3.2, "Tensor shapes inside attention:", fontweight="bold",
            ha="center", fontsize=10)
    ax.text(7, 2.6, "Q, K, V : (B, n_heads=6, T, d_head=64)", ha="center", fontsize=9,
            family="monospace")
    ax.text(7, 2.2, "attn scores : (B, n_heads, T, T)", ha="center", fontsize=9,
            family="monospace")
    ax.text(7, 1.8, "output   : (B, T, d_model) ← concat(heads) then out_proj", ha="center",
            fontsize=9, family="monospace")

    ax.text(7, 0.9,
            "Inside FFN: hidden_dim ≈ 2·d_ff/3 rounded to ×8 "
            "(SwiGLU uses 3 matmuls vs 2, so shrink hidden to keep param count).",
            ha="center", fontsize=9, style="italic", color="#444")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 7 — KV cache prefill vs decode
# ───────────────────────────────────────────────────────────────

def draw_kv_cache_flow(path):
    if os.path.exists(path):
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Panel 1: Prefill ──
    ax = axes[0]
    ax.set_title("PREFILL  (T = prompt_len, past_kv = None)",
                 fontsize=11, fontweight="bold", color=_BLUE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, (0.5, 8), 9, 0.9, "prompt : (B, prompt_len)", face=_GREY, edge="black")
    _arrow(ax, (5, 8), (5, 7.4))
    _box(ax, (0.5, 6), 9, 1.3,
         "model.forward(idx=prompt, past_kv=None)\n"
         "• RoPE with start_pos = 0\n"
         "• Causal mask APPLIED (upper triangle → -∞)",
         face=_BLUE_LT, edge=_BLUE, fontsize=9)
    _arrow(ax, (5, 6), (5, 5.4))
    _box(ax, (0.5, 4.3), 9, 0.9, "(logits, cache₀)", face=_GREEN_LT, edge=_GREEN, fontsize=10)

    ax.text(5, 3.2, "cache₀ = [(K_layer0, V_layer0), …, (K_layer5, V_layer5)]",
            ha="center", fontsize=9, family="monospace")
    ax.text(5, 2.6, "each (K, V) shape: (B, n_heads, prompt_len, d_head)",
            ha="center", fontsize=9, family="monospace", color="#444")
    ax.text(5, 1.6, "Heavy: one forward pass over the entire prompt.",
            ha="center", fontsize=9, style="italic", color="#444")

    # ── Panel 2: Decode step ──
    ax = axes[1]
    ax.set_title("DECODE STEP k  (T = 1, past_kv = cacheₖ)",
                 fontsize=11, fontweight="bold", color=_ORANGE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    _box(ax, (0.5, 8), 9, 0.9, "next_token : (B, 1)", face=_GREY, edge="black")
    _arrow(ax, (5, 8), (5, 7.4))
    _box(ax, (0.5, 5.2), 9, 2.2,
         "model.forward(idx=next_token, past_kv=cacheₖ)\n"
         "• Q, K, V for the new token : (B, nh, 1, d_head)\n"
         "• RoPE with start_pos = past_len   ← KEY INVARIANT\n"
         "• Append new K,V to cacheₖ → cacheₖ₊₁\n"
         "• NO causal mask (T=1, single query is always newest)",
         face=_ORANGE_LT, edge=_ORANGE, fontsize=9)
    _arrow(ax, (5, 5.2), (5, 4.6))
    _box(ax, (0.5, 3.5), 9, 0.9, "(logits, cacheₖ₊₁)", face=_GREEN_LT, edge=_GREEN, fontsize=10)

    ax.text(5, 2.5, "Light: one small forward per new token.",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.text(5, 1.8,
            "Step cost: O(past_len + 1) not O((past_len + 1)²)",
            ha="center", fontsize=10, fontweight="bold", color=_ORANGE)
    ax.text(5, 1.0,
            "Loop: sample_next(logits) → next_token → repeat",
            ha="center", fontsize=9, family="monospace", color="#333")

    fig.suptitle("Step 7 — KV cache: prefill then decode",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 10 — UI layout
# ───────────────────────────────────────────────────────────────

def draw_ui_layout(path):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    _header(ax, "Gradio UI — four tabs",
            "Each tab reuses existing teach.py / visualise.py plumbing — no logic is duplicated.")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    tabs = [
        ("Generate",   "live streaming\nKV cache toggle",      _BLUE,   _BLUE_LT),
        ("Teach",      "16 slides\non-the-fly",                _ORANGE, _ORANGE_LT),
        ("Attention",  "per-head heatmap\n+ rollout",          _GREEN,  _GREEN_LT),
        ("Benchmark",  "generate vs\ngenerate_fast",           "#6a51a3", "#dadaeb"),
        ("Build Steps", "this tab\n— 12-step tour",            "#99000d", "#fcbba1"),
    ]
    w = 3.0
    gap = 0.1
    for i, (name, sub, edge, face) in enumerate(tabs):
        x = 0.2 + i * (w + gap)
        _box(ax, (x, 6), w, 2.8, f"{name}\n\n{sub}",
             face=face, edge=edge, fontsize=10)

    # Shared model arrow
    ax.text(8, 4.4, "All tabs share one loaded model (module-level singleton)",
            ha="center", fontsize=10, color="#333")
    _arrow(ax, (8, 5.3), (8, 3.8), color="#555")
    _box(ax, (5.5, 2.5), 5, 1.0,
         "_MODEL, _TOKENIZER, _CONFIG   ←  app.py _load_model()",
         face="#fff2cc", edge="#bf9000", fontsize=10)

    ax.text(8, 1.3,
            "Tests monkeypatch these three globals with the tiny_model fixture — "
            "zero I/O, sub-second UI smoke tests.",
            ha="center", fontsize=9, style="italic", color="#444")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 11 — Test pyramid
# ───────────────────────────────────────────────────────────────

def draw_test_matrix(path):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _header(ax, "Test suite — 48 tests, ~18s on CPU",
            "Unit tests per component + integration + UI smoke. Run: bash run.sh test")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    buckets = [
        ("Config",      5,  "d_head validation, defaults",           _BLUE),
        ("Tokenizer",  12,  "roundtrip, compression, save/load",     _ORANGE),
        ("Dataset",     5,  "shapes, target-shift invariant",        _GREEN),
        ("Model",      15,  "KV-cache equivalence (multi-step),\n"
                            "causal-mask leakage, weight tying,\n"
                            "RoPE offset, greedy determinism",       "#6a51a3"),
        ("Integration", 4,  "overfit loss-drop, full pipeline,\n"
                            "checkpoint roundtrip",                  "#99000d"),
        ("UI smoke",    6,  "build Blocks tree, streaming yields,\n"
                            "benchmark handler, error messaging",    "#8c6d31"),
    ]

    y = 6.8
    for name, n, desc, color in buckets:
        ax.barh(y, n, height=0.7, color=color, alpha=0.85)
        ax.text(-0.1, y, name, ha="right", va="center", fontsize=10,
                fontweight="bold")
        ax.text(n + 0.15, y, f"{n}", ha="left", va="center",
                fontsize=10, fontweight="bold", color=color)
        ax.text(n + 0.8, y, desc, ha="left", va="center", fontsize=9,
                color="#333")
        y -= 1.0

    # Total
    ax.text(5, -0.5, "Total: 48 tests · 42 unit + 6 UI smoke · pytest fixtures in conftest.py",
            ha="center", fontsize=10, fontweight="bold", color="#333")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 1 — Config param summary (as a simple bar chart)
# ───────────────────────────────────────────────────────────────

def draw_config_summary(path):
    if os.path.exists(path):
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _header(ax, "LLM101 — the hyperparameters (NanoLLMConfig dataclass)",
            "One dataclass, every tunable. Derived values as @property.")

    names = ["d_model", "n_layers", "n_heads", "d_ff",
             "max_seq_len", "target_vocab_size", "batch_size"]
    vals  = [384, 6, 6, 1536, 256, 4096, 64]
    colors = [_BLUE] * len(names)
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v * 1.01, i, f"{v}", va="center", fontsize=10, fontweight="bold",
                color="#08519c")
    ax.set_xscale("log")
    ax.set_xlabel("value (log scale)")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Step 5 — Training curve schematic (if no real loss_curve.png exists)
# ───────────────────────────────────────────────────────────────

def draw_training_schematic(path):
    if os.path.exists(path):
        return
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5))
    _header(ax, "Training curve — what to expect",
            "Train loss drops quickly; val loss eventually plateaus. "
            "Best checkpoint = lowest val loss, not train loss.")

    x = np.linspace(0, 100, 200)
    train = 8.3 * np.exp(-x / 30) + 3.0 + np.random.default_rng(0).normal(0, 0.05, size=len(x))
    val = 8.3 * np.exp(-x / 40) + 3.4 + np.random.default_rng(1).normal(0, 0.03, size=len(x))
    val[120:] += np.linspace(0, 0.4, len(val) - 120)  # overfitting drift

    ax.plot(x, train, color="#08519c", linewidth=2, label="Train")
    ax.plot(x, val, color="#cb181d", linewidth=2, label="Val")

    best = int(val.argmin())
    ax.scatter([x[best]], [val[best]], s=120, color="#cb181d",
               zorder=5, edgecolor="black", linewidth=1.5)
    ax.annotate("best val loss\n(checkpoint saved)", xy=(x[best], val[best]),
                xytext=(x[best] + 15, val[best] - 0.6),
                fontsize=10, color="#cb181d",
                arrowprops=dict(arrowstyle="->", color="#cb181d"))

    ax.set_xlabel("global step", fontsize=11)
    ax.set_ylabel("cross-entropy loss (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()


# ───────────────────────────────────────────────────────────────
# Orchestration — render all diagrams to a directory
# ───────────────────────────────────────────────────────────────

def render_all(outdir: str) -> dict[str, str]:
    """Render every diagram to outdir. Returns dict of step_key → png path.
    Idempotent: skips files that already exist."""
    os.makedirs(outdir, exist_ok=True)
    paths = {
        "config":      os.path.join(outdir, "step_01_config.png"),
        "sliding":     os.path.join(outdir, "step_03_sliding_window.png"),
        "block":       os.path.join(outdir, "step_04_block.png"),
        "training":    os.path.join(outdir, "step_05_training.png"),
        "kv_cache":    os.path.join(outdir, "step_07_kv_cache.png"),
        "ui_layout":   os.path.join(outdir, "step_10_ui_layout.png"),
        "test_matrix": os.path.join(outdir, "step_11_tests.png"),
    }
    draw_config_summary(paths["config"])
    draw_sliding_window(paths["sliding"])
    draw_transformer_block(paths["block"])
    draw_training_schematic(paths["training"])
    draw_kv_cache_flow(paths["kv_cache"])
    draw_ui_layout(paths["ui_layout"])
    draw_test_matrix(paths["test_matrix"])
    return paths
