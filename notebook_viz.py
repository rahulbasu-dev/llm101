"""Notebook-derived visualizations for the Gradio app.

Mirrors the interactive demos in `LLM101_From_Scratch.ipynb` sections 2, 3,
4, and 9 — the pieces that the existing app didn't expose. Complements
`build_viz.py` (structural diagrams, no model needed) and `teach.py` (full
forward-pass slides on a real prompt) by focusing on atomic, single-concept
demos cell-by-cell.

Functions return matplotlib Figures, summary strings, or plain dicts — never
write to disk. The Gradio handlers in `app.py` save figures to a tempdir.
This separation keeps the helpers unit-testable without launching Gradio.

Layout:
  Tokenizer  —  encode_breakdown / format_breakdown / draw_tokenizer_overview
  Dataset    —  draw_window_view
  Components —  rmsnorm_summary / draw_rmsnorm_dist
                rope_summary / draw_rope_demo
                attention_summary / draw_causal_mask
                swiglu_summary / draw_swiglu_breakdown
  KV Cache   —  kv_cache_single_step / kv_cache_multi_step
                format_kv_single_step / format_kv_multi_step
                draw_length_sweep
"""

from __future__ import annotations
from typing import List, Tuple

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from config import NanoLLMConfig
from tokenizer import BPETokenizer, NUM_BASE, NUM_SPECIAL, BYTE_OFFSET
from model import (
    NanoLLM, RMSNorm, RotaryPositionEmbedding,
    CausalSelfAttention, FeedForward,
)


# ═══════════════════════════════════════════════════════════════
# Section 2: Tokenizer demos
# ═══════════════════════════════════════════════════════════════

def encode_breakdown(tokenizer: BPETokenizer, text: str) -> dict:
    """Encode `text` and return a per-token breakdown.

    Mirrors notebook cell 9 but as a pure data structure so the UI can format
    it however it likes.

    Returns:
        {
          "input": str,
          "encoded": list[int],
          "decoded": str,
          "round_trip_match": bool,
          "tokens": [{"id": int, "kind": "SPECIAL"|"BYTE"|"MERGE", "label": str}, ...],
        }
    """
    encoded = tokenizer.encode(text, add_special=False)
    decoded = tokenizer.decode(encoded)
    tokens = []
    for tid in encoded:
        if tid >= NUM_BASE:
            kind = "MERGE"
        elif tid >= BYTE_OFFSET:
            kind = "BYTE"
        else:
            kind = "SPECIAL"
        tokens.append({
            "id": int(tid),
            "kind": kind,
            "label": tokenizer.decode_token(tid),
        })
    return {
        "input": text,
        "encoded": [int(x) for x in encoded],
        "decoded": decoded,
        "round_trip_match": text == decoded,
        "tokens": tokens,
    }


def format_breakdown(breakdown: dict) -> str:
    """Render an encode_breakdown result as plain text. Format mirrors notebook cell 9."""
    lines = [
        f"Original:  {breakdown['input']!r}",
        f"Token IDs: {breakdown['encoded']}",
        f"Decoded:   {breakdown['decoded']!r}",
        f"Round-trip match: {breakdown['round_trip_match']}",
        f"Tokens: {len(breakdown['tokens'])}",
        "",
        "Token breakdown (id  kind     label):",
    ]
    for t in breakdown["tokens"]:
        lines.append(f"  {t['id']:>5}  {t['kind']:<7}  {t['label']!r}")
    return "\n".join(lines)


def draw_tokenizer_overview(tokenizer: BPETokenizer,
                            sample_texts: List[str] | None = None) -> plt.Figure:
    """Vocabulary composition + compression ratios. Mirrors notebook cell 10."""
    if sample_texts is None:
        sample_texts = [
            "Hello world!",
            "The king said to his servant",
            "Once upon a time there was a little cat.",
            "abcdefghijklmnop",
        ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    categories = ["Special\n(PAD,BOS,EOS,UNK)", "Byte tokens\n(0x00–0xFF)", "BPE merges\n(learned)"]
    counts = [NUM_SPECIAL, 256, max(0, tokenizer.vocab_size - NUM_BASE)]
    colors = ["#dc3545", "#fd8d3c", "#28a745"]
    axes[0].bar(categories, counts, color=colors, edgecolor="white", linewidth=2)
    pad = max(counts) * 0.02 if max(counts) else 1
    for i, v in enumerate(counts):
        axes[0].text(i, v + pad, str(v), ha="center", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Number of tokens")
    axes[0].set_title("Vocabulary composition", fontsize=12, fontweight="bold")
    axes[0].grid(True, axis="y", alpha=0.3)

    ratios = []
    for s in sample_texts:
        toks = tokenizer.encode(s, add_special=False)
        ratios.append(len(s.encode("utf-8")) / max(1, len(toks)))
    y_pos = list(range(len(sample_texts)))
    axes[1].barh(y_pos, ratios, color="#08519c")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([s[:35] + ("…" if len(s) > 35 else "") for s in sample_texts],
                            fontsize=9)
    axes[1].set_xlabel("Compression ratio (bytes / tokens)")
    axes[1].set_title("BPE compression ratio (higher = denser)",
                      fontsize=12, fontweight="bold")
    for i, v in enumerate(ratios):
        axes[1].text(v + 0.04, i, f"{v:.2f}×", va="center", fontsize=10)
    axes[1].axvline(1.0, color="#888", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 3: Dataset / sliding windows
# ═══════════════════════════════════════════════════════════════

def draw_window_view(tokenizer: BPETokenizer, tokens: List[int],
                     seq_len: int, stride: int, sample_idx: int = 0,
                     n_show: int = 12, n_windows: int = 4) -> plt.Figure:
    """Two-panel viz: input→target shift (top) + overlapping windows (bottom).

    Mirrors notebook cell 13.
    """
    if not tokens:
        raise ValueError("No tokens to visualize")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    start = sample_idx * stride
    if start + seq_len + 1 > len(tokens):
        start = max(0, len(tokens) - seq_len - 1)
    inp = tokens[start : start + seq_len]
    tgt = tokens[start + 1 : start + seq_len + 1]

    n_show = min(n_show, len(inp))
    inp_labels = [tokenizer.decode_token(int(t)).replace("\n", "\\n")[:6]
                  for t in inp[:n_show]]
    tgt_labels = [tokenizer.decode_token(int(t)).replace("\n", "\\n")[:6]
                  for t in tgt[:n_show]]

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.5))

    axes[0].set_xlim(-0.5, n_show - 0.5)
    axes[0].set_ylim(-0.5, 1.5)
    for i in range(n_show):
        axes[0].text(i, 1, inp_labels[i], ha="center", va="center", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#e6f2ff", edgecolor="#4a90d9"))
        axes[0].text(i, 0, tgt_labels[i], ha="center", va="center", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#ffe6e6", edgecolor="#dc3545"))
        axes[0].annotate("", xy=(i, 0.35), xytext=(i, 0.65),
                         arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Target\n(shifted +1)", "Input"], fontsize=10)
    axes[0].set_xticks(range(n_show))
    axes[0].set_xticklabels([f"pos {i}" for i in range(n_show)], fontsize=8)
    axes[0].set_title("Input vs target — target = input shifted right by 1",
                      fontsize=12, fontweight="bold")

    colors_w = ["#08519c", "#e6550d", "#2ca02c", "#9467bd", "#8c564b"]
    drawn = 0
    for w in range(n_windows):
        s = w * stride
        if s + seq_len > len(tokens):
            break
        axes[1].barh(w, seq_len, left=s, height=0.6,
                     color=colors_w[w % len(colors_w)], alpha=0.7,
                     edgecolor="white")
        axes[1].text(s + seq_len / 2, w,
                     f"Window {w} (tokens {s}…{s + seq_len - 1})",
                     ha="center", va="center", fontsize=9, color="white",
                     fontweight="bold")
        drawn = w + 1
    axes[1].set_xlabel("Token position in corpus")
    axes[1].set_yticks(range(max(1, drawn)))
    axes[1].set_yticklabels([f"Sample {i}" for i in range(max(1, drawn))])
    axes[1].set_title(
        f"Overlapping sliding windows — stride={stride}, seq_len={seq_len}",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 4a: RMSNorm
# ═══════════════════════════════════════════════════════════════

def rmsnorm_summary(config: NanoLLMConfig) -> str:
    """Multi-line text summary mirroring notebook cell 15 prints."""
    norm = RMSNorm(config.d_model)
    dummy = torch.randn(2, 10, config.d_model)
    normed = norm(dummy)
    ln = nn.LayerNorm(config.d_model)
    rms_params = sum(p.numel() for p in norm.parameters())
    ln_params = sum(p.numel() for p in ln.parameters())
    saving_pct = (ln_params - rms_params) / ln_params * 100 if ln_params else 0
    return (
        "RMSNorm\n"
        "  Formula: output = x · rsqrt(mean(x²) + eps) · weight\n"
        f"  Input shape:  {tuple(dummy.shape)}\n"
        f"  Output shape: {tuple(normed.shape)}\n"
        f"  Learnable params (gamma, one per dim): {norm.weight.shape[0]}\n"
        "\n"
        f"  Input  — mean: {dummy.mean():.4f}, std: {dummy.std():.4f}\n"
        f"  Output — mean: {normed.mean():.4f}, std: {normed.std():.4f}\n"
        "\n"
        f"  RMSNorm has   {rms_params:,} params (weight only)\n"
        f"  LayerNorm has {ln_params:,} params (weight + bias)\n"
        f"  Saving: {ln_params - rms_params:,} params per norm layer "
        f"({saving_pct:.0f}% reduction)"
    )


def draw_rmsnorm_dist(config: NanoLLMConfig) -> plt.Figure:
    """Histograms of activations before / after RMSNorm.

    Uses input with std≈3, mean≈0.5 so the rescaling effect is visible.
    """
    torch.manual_seed(0)
    norm = RMSNorm(config.d_model)
    dummy = torch.randn(2, 10, config.d_model) * 3.0 + 0.5
    normed = norm(dummy)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(dummy.flatten().numpy(), bins=60, color="#9ecae1",
                 edgecolor="#08519c", linewidth=0.5)
    axes[0].axvline(0, color="#666", linestyle="--", linewidth=1)
    axes[0].set_title(f"Before — std={dummy.std():.2f}, mean={dummy.mean():.2f}",
                      fontsize=11, fontweight="bold")
    axes[0].set_xlabel("activation value")
    axes[0].set_ylabel("count")

    normed_np = normed.detach().flatten().numpy()
    axes[1].hist(normed_np, bins=60, color="#a1d99b",
                 edgecolor="#238b45", linewidth=0.5)
    axes[1].axvline(0, color="#666", linestyle="--", linewidth=1)
    axes[1].set_title(f"After — std={normed.std():.2f}, mean={normed.mean():.2f}",
                      fontsize=11, fontweight="bold")
    axes[1].set_xlabel("activation value")

    fig.suptitle("RMSNorm rescales without recentering (no bias subtraction)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 4b: RoPE
# ═══════════════════════════════════════════════════════════════

def rope_summary(config: NanoLLMConfig) -> str:
    rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len)
    return (
        "RotaryPositionEmbedding (RoPE)\n"
        f"  d_head: {config.d_head}  →  {config.d_head // 2} frequency bands\n"
        f"  max_seq_len: {config.max_seq_len}\n"
        f"  cos_cached shape: {tuple(rope.cos_cached.shape)}\n"
        f"  sin_cached shape: {tuple(rope.sin_cached.shape)}\n"
        "\n"
        "  Each pair of dims (i, i + d_head/2) is treated as a 2D vector and\n"
        "  rotated by an angle pos · θᵢ where θᵢ = 10000^(-2i/d_head).\n"
        "  Lower frequencies (small i) rotate slowly → encode coarse position.\n"
        "  Higher frequencies (large i) rotate fast → encode fine position.\n"
        "\n"
        "  Why RoPE: angles depend only on the *relative* offset between\n"
        "  positions, which generalises better to longer sequences than\n"
        "  absolute positional embeddings."
    )


def draw_rope_demo(config: NanoLLMConfig, show_pos: int = 64) -> plt.Figure:
    rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len)
    cos_table = rope.cos_cached.numpy()
    sin_table = rope.sin_cached.numpy()
    show_pos = min(show_pos, config.max_seq_len)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im1 = axes[0].imshow(cos_table[:show_pos], aspect="auto",
                         cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_title("cos(pos · θᵢ)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Frequency index i")
    axes[0].set_ylabel("Position")
    fig.colorbar(im1, ax=axes[0], shrink=0.8)

    im2 = axes[1].imshow(sin_table[:show_pos], aspect="auto",
                         cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1].set_title("sin(pos · θᵢ)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency index i")
    fig.colorbar(im2, ax=axes[1], shrink=0.8)

    positions = [p for p in [0, 2, 4, 8, 16, 32] if p < show_pos]
    colors_pos = plt.cm.viridis(np.linspace(0.15, 0.85, len(positions)))
    for p, color in zip(positions, colors_pos):
        c, s = float(cos_table[p, 0]), float(sin_table[p, 0])
        axes[2].arrow(0, 0, c, s, head_width=0.04, length_includes_head=True,
                      color=color, linewidth=2)
        axes[2].text(c * 1.15, s * 1.15, f"pos {p}", color=color,
                     fontsize=9, fontweight="bold")
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_ylim(-1.3, 1.3)
    axes[2].set_aspect("equal")
    axes[2].axhline(0, color="#bbb", linewidth=0.5)
    axes[2].axvline(0, color="#bbb", linewidth=0.5)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title("Unit vector rotated by position\n(lowest-freq dim pair)",
                      fontsize=12, fontweight="bold")

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 4c: Causal self-attention
# ═══════════════════════════════════════════════════════════════

def attention_summary(config: NanoLLMConfig) -> str:
    attn = CausalSelfAttention(config)
    x = torch.randn(2, 10, config.d_model)
    out, kv = attn(x)
    k, v = kv
    return (
        "CausalSelfAttention\n"
        f"  Input:    {tuple(x.shape)}  (batch, seq_len, d_model)\n"
        f"  Output:   {tuple(out.shape)}  (residual-friendly: same shape)\n"
        f"  K cache:  {tuple(k.shape)}  (batch, n_heads, seq_len, d_head)\n"
        f"  V cache:  {tuple(v.shape)}\n"
        "\n"
        f"  Combined QKV projection: {config.d_model} → {3 * config.d_model} (single matmul)\n"
        f"  n_heads: {config.n_heads}, d_head: {config.d_head}\n"
        f"  Causal mask buffer: {tuple(attn.causal_mask.shape)}\n"
        "\n"
        "  Why combined QKV: one matmul + one slice is faster than three matmuls.\n"
        "  Why causal mask: a position must not see future positions during\n"
        "  training — without the mask, the model would 'cheat' by reading the\n"
        "  answer it's supposed to predict."
    )


def draw_causal_mask(config: NanoLLMConfig, n_show: int = 12) -> plt.Figure:
    attn = CausalSelfAttention(config)
    n_show = min(n_show, attn.causal_mask.shape[-1])
    mask = attn.causal_mask[0, 0, :n_show, :n_show].numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mask, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Causal mask (first {n_show} positions)\n"
                 "1 = can attend, 0 = masked",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Key position (j)")
    ax.set_ylabel("Query position (i)")
    for i in range(n_show):
        for j in range(n_show):
            ax.text(j, i, int(mask[i, j]), ha="center", va="center",
                    fontsize=9, color="white" if mask[i, j] > 0.5 else "black")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 4d: SwiGLU
# ═══════════════════════════════════════════════════════════════

def _swiglu_hidden_dim(config: NanoLLMConfig) -> int:
    """SwiGLU's effective hidden dim — 2/3 · d_ff, rounded up to multiple of 8."""
    h = int(2 * config.d_ff / 3)
    return ((h + 7) // 8) * 8


def swiglu_summary(config: NanoLLMConfig) -> str:
    ffn = FeedForward(config)
    x = torch.randn(2, 10, config.d_model)
    out = ffn(x)
    hidden_dim = _swiglu_hidden_dim(config)
    total = sum(p.numel() for p in ffn.parameters())
    return (
        "FeedForward (SwiGLU)\n"
        f"  Input:      {tuple(x.shape)}\n"
        f"  Output:     {tuple(out.shape)}\n"
        f"  d_model:    {config.d_model}\n"
        f"  d_ff:       {config.d_ff}\n"
        f"  hidden_dim: {hidden_dim}  (2/3 · d_ff, rounded to multiple of 8)\n"
        "\n"
        f"  gate_proj: ({config.d_model}, {hidden_dim}) — learns WHAT to let through\n"
        f"  up_proj:   ({config.d_model}, {hidden_dim}) — expands the representation\n"
        f"  down_proj: ({hidden_dim}, {config.d_model}) — projects back down\n"
        f"  Total FFN params: {total:,}\n"
        "\n"
        "  Forward: down_proj(silu(gate_proj(x)) ⊙ up_proj(x))\n"
        "  silu(x) = x · sigmoid(x)  — smooth approximation of ReLU.\n"
        "\n"
        "  Why three projections: the gate selects which hidden units fire,\n"
        "  similar to GLU. Same parameter budget as a 2-projection FFN with\n"
        "  d_ff hidden, but consistently lower loss empirically."
    )


def draw_swiglu_breakdown(config: NanoLLMConfig) -> plt.Figure:
    """Bar chart of parameter counts in each of SwiGLU's 3 projections."""
    hidden = _swiglu_hidden_dim(config)
    gate = config.d_model * hidden
    up = config.d_model * hidden
    down = hidden * config.d_model
    names = ["gate_proj\n(d_model→hidden)", "up_proj\n(d_model→hidden)",
             "down_proj\n(hidden→d_model)"]
    counts = [gate, up, down]
    colors = ["#fdae6b", "#9ecae1", "#a1d99b"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=2)
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.01,
                f"{v:,}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylabel("parameters")
    ax.set_title(f"SwiGLU parameter breakdown — total {sum(counts):,} params",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════
# Section 9: KV Cache deep dive
# ═══════════════════════════════════════════════════════════════

def kv_cache_single_step(model: NanoLLM, config: NanoLLMConfig,
                         T: int = 16) -> dict:
    """Compare full forward vs (prefill on T-1) + (1-step decode with cache).

    Mirrors notebook cell 46.
    """
    model.eval()
    device = next(model.parameters()).device
    T = max(2, min(T, config.max_seq_len))
    test = torch.randint(0, config.vocab_size, (1, T), device=device)
    with torch.no_grad():
        full_logits, _ = model(test)
        prefill_logits, cache = model(test[:, :-1])
        step_logits, _ = model(test[:, -1:], past_kv=cache)
    max_diff = (full_logits - step_logits).abs().max().item()
    mean_diff = (full_logits - step_logits).abs().mean().item()
    k0, v0 = cache[0]
    return {
        "T": T,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "passed": max_diff < 1e-3,
        "n_layers": len(cache),
        "k_shape": tuple(k0.shape),
        "v_shape": tuple(v0.shape),
    }


def format_kv_single_step(r: dict) -> str:
    pass_str = "YES ✓" if r["passed"] else "NO ✗"
    return (
        "KV Cache equivalence — single decode step\n"
        + "─" * 50 + "\n"
        f"  Sequence length:   {r['T']} tokens\n"
        f"  Max  |Δ logit|:    {r['max_diff']:.2e}\n"
        f"  Mean |Δ logit|:    {r['mean_diff']:.2e}\n"
        f"  PASS (<1e-3):      {pass_str}\n"
        "\n"
        "Cache structure (per layer):\n"
        f"  Layers:  {r['n_layers']}\n"
        f"  K shape: {r['k_shape']}  (B, n_heads, cached_T, d_head)\n"
        f"  V shape: {r['v_shape']}"
    )


def kv_cache_multi_step(model: NanoLLM, config: NanoLLMConfig,
                        T: int = 20) -> dict:
    """Feed tokens one-by-one through the cache; compare last-position logits
    against a single full forward pass. Mirrors notebook cell 47.
    """
    model.eval()
    device = next(model.parameters()).device
    T = max(2, min(T, config.max_seq_len))
    test = torch.randint(0, config.vocab_size, (1, T), device=device)
    with torch.no_grad():
        ref_logits, _ = model(test)
        cache = None
        last = None
        for t in range(T):
            tok = test[:, t : t + 1]
            last, cache = model(tok, past_kv=cache)
    max_diff = (ref_logits - last).abs().max().item()
    return {
        "T": T,
        "max_diff": max_diff,
        "passed": max_diff < 1e-3,
    }


def format_kv_multi_step(r: dict) -> str:
    pass_str = "YES ✓" if r["passed"] else "NO ✗"
    return (
        f"Multi-step equivalence — {r['T']} tokens fed one at a time\n"
        + "─" * 50 + "\n"
        f"  Max |Δ logit| on final position: {r['max_diff']:.2e}\n"
        f"  PASS (<1e-3):                    {pass_str}"
    )


def draw_length_sweep(model: NanoLLM, config: NanoLLMConfig,
                      prompt_lens: List[int],
                      gen_len: int = 50) -> Tuple[plt.Figure, str]:
    """Bar chart of generate() vs generate_fast() time at varying prompt lengths.

    Mirrors notebook cell 48. Returns (Figure, summary_text).
    """
    model.eval()
    device = next(model.parameters()).device
    times_std: list[float] = []
    times_fast: list[float] = []

    max_safe = max(1, config.max_seq_len - gen_len - 1)
    prompt_lens = [max(1, min(p, max_safe)) for p in prompt_lens]

    warm = torch.randint(0, config.vocab_size, (1, max(prompt_lens)), device=device)
    with torch.no_grad():
        model.generate(warm, max_new_tokens=4)
        model.generate_fast(warm, max_new_tokens=4)
    if device.type == "cuda":
        torch.cuda.synchronize()

    for plen in prompt_lens:
        prompt = torch.randint(0, config.vocab_size, (1, plen), device=device)

        t0 = time.time()
        with torch.no_grad():
            model.generate(prompt.clone(), max_new_tokens=gen_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_std.append(time.time() - t0)

        t0 = time.time()
        with torch.no_grad():
            model.generate_fast(prompt.clone(), max_new_tokens=gen_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_fast.append(time.time() - t0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(prompt_lens))
    width = 0.35
    ax.bar(x - width / 2, times_std, width, label="generate() — no cache",
           color="#dc3545")
    ax.bar(x + width / 2, times_fast, width, label="generate_fast() — KV cache",
           color="#28a745")
    for i, (ts, tf) in enumerate(zip(times_std, times_fast)):
        speedup = ts / max(tf, 1e-6)
        ax.text(i, max(ts, tf) * 1.02, f"{speedup:.1f}×",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"KV cache speedup — generating {gen_len} new tokens",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prompt_lens])
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    summary_lines = [
        f"{'prompt':>8} | {'no-cache':>10} | {'cached':>10} | {'speedup':>8}",
        "-" * 46,
    ]
    for plen, ts, tf in zip(prompt_lens, times_std, times_fast):
        speedup = ts / max(tf, 1e-6)
        summary_lines.append(
            f"{plen:>8} | {ts*1000:>8.1f}ms | {tf*1000:>8.1f}ms | {speedup:>6.2f}×"
        )
    return fig, "\n".join(summary_lines)
