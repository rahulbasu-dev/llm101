"""NanoLLM — Step-by-step forward-pass walkthrough for students.

Produces a numbered sequence of annotated PNG "slides" in teaching_plots/
that walk through what happens when a single prompt passes through the model.
Designed for a webinar or classroom.

Usage:
    python teach.py                                 # default prompt
    python teach.py --text "The cat sat on the"
    python teach.py --checkpoint checkpoints/best.pt --layer 0 --head 0

Slides produced:
    01_tokenization.png      — text -> bytes -> BPE -> token IDs             [Raschka Ch 2]
    02_embeddings.png        — token IDs -> embedding vectors                [Raschka Ch 2]
    03_qkv.png               — Q, K, V projections (side-by-side heatmaps)   [Raschka Ch 3]
    04_scores_raw.png        — Q.K^T / sqrt(d)  (before mask)                [Raschka Ch 3]
    05_scores_masked.png     — same, with causal mask applied                [Raschka Ch 3]
    06_attn_weights.png      — after softmax (rows sum to 1)                 [Raschka Ch 3]
    07_value_sum.png         — weighted sum of V for one query position      [Raschka Ch 3]
    08_all_heads.png         — every head of the chosen layer side-by-side  [Raschka Ch 3]
    09_ffn_delta.png         — hidden state before and after the FFN         [Raschka Ch 4]
    10_logits_topk.png       — top-20 next-token distribution                [Raschka Ch 5]
    11_sampling_rollout.png  — 10 decode steps, top-5 candidates each        [Raschka Ch 5]
    12_positional_rope.png   — RoPE frequency table + position rotation      [Raschka Ch 2]
    13_scaling_rationale.png — softmax saturates without /sqrt(d)            [Raschka Ch 3]
    14_temperature_effect.png — same logits at T=0.3 / 1.0 / 2.0             [Raschka Ch 5]
    15_greedy_vs_sample.png  — greedy vs temperature sampling text           [Raschka Ch 5]
    16_param_breakdown.png   — pie chart: where the 15M params live          [Raschka Ch 4]

The model forward is NOT modified. Intermediate tensors are captured via
forward hooks (same pattern as visualise.py). This keeps the production code
clean and lets students compare the "real" code against teaching instrumentation.
"""

import argparse
import math
import os
import sys
import torch
import torch.nn.functional as F

from config import NanoLLMConfig
from tokenizer import BPETokenizer, NUM_BASE, BYTE_OFFSET
from model import NanoLLM, _sample_from_logits


# ═══════════════════════════════════════════════════════════════
# Matplotlib setup (lazy, with clear error if missing)
# ═══════════════════════════════════════════════════════════════

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib is required for teach.py. Install with:")
        print("  pip install matplotlib")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# Hook-based tensor capture
# ═══════════════════════════════════════════════════════════════

class ForwardCapture:
    """Capture intermediate tensors at one target layer for pedagogy.

    Captures:
        - token embeddings (input to first block)
        - attn input (= output of attn_norm) at target layer
        - q, k, v after projection + RoPE at target layer
        - raw attention scores (before mask)
        - masked scores (after causal mask)
        - softmax attention weights
        - attention output (before out_proj)
        - FFN input (= output of ffn_norm) and FFN output at target layer
    """

    def __init__(self, model: NanoLLM, target_layer: int = 0):
        self.model = model
        self.target_layer = target_layer
        self.store = {}
        self._hooks = []
        self._install()

    def _install(self):
        # Embedding output (after emb_dropout)
        def emb_hook(module, inp, out):
            self.store["embeddings"] = out.detach().cpu()
        self._hooks.append(self.model.emb_dropout.register_forward_hook(emb_hook))

        target_block = self.model.blocks[self.target_layer]

        # Attention: manually recompute to capture intermediates
        def attn_pre_hook(module, inp):
            x_norm = inp[0]  # output of attn_norm
            self.store["attn_input"] = x_norm.detach().cpu()

            B, T, C = x_norm.shape
            qkv = module.qkv_proj(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, module.n_heads, module.d_head).transpose(1, 2)
            k = k.view(B, T, module.n_heads, module.d_head).transpose(1, 2)
            v = v.view(B, T, module.n_heads, module.d_head).transpose(1, 2)

            q_rot = module.rope(q, T, start_pos=0)
            k_rot = module.rope(k, T, start_pos=0)

            self.store["q"] = q_rot.detach().cpu()
            self.store["k"] = k_rot.detach().cpu()
            self.store["v"] = v.detach().cpu()

            scale = 1.0 / math.sqrt(module.d_head)
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
            self.store["scores_raw"] = scores.detach().cpu()

            masked = scores.masked_fill(
                module.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            self.store["scores_masked"] = masked.detach().cpu()

            weights = F.softmax(masked, dim=-1)
            self.store["attn_weights"] = weights.detach().cpu()

            attn_out = torch.matmul(weights, v)  # (B, nh, T, d_head)
            self.store["attn_out"] = attn_out.detach().cpu()

        self._hooks.append(target_block.attn.register_forward_pre_hook(attn_pre_hook))

        # FFN: capture input (= ffn_norm output) and output
        def ffn_pre_hook(module, inp):
            self.store["ffn_input"] = inp[0].detach().cpu()
        self._hooks.append(target_block.ffn.register_forward_pre_hook(ffn_pre_hook))

        def ffn_post_hook(module, inp, out):
            self.store["ffn_output"] = out.detach().cpu()
        self._hooks.append(target_block.ffn.register_forward_hook(ffn_post_hook))

    def remove(self):
        for h in self._hooks:
            h.remove()


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def load_model(checkpoint_path, device):
    """Load model+tokenizer from a checkpoint. Fall back to random init if missing."""
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
        tokenizer.load(config.tokenizer_path)
        config.vocab_size = tokenizer.vocab_size
        model = NanoLLM(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded epoch {ckpt.get('epoch', '?')}")
        return model, tokenizer, config

    print(f"WARNING: {checkpoint_path} not found — using random weights.")
    print("         Visualizations will look like noise. Train first for real patterns.")
    config = NanoLLMConfig(vocab_size=260)
    tokenizer = BPETokenizer(target_vocab_size=260)
    model = NanoLLM(config).to(device).eval()
    return model, tokenizer, config


def token_labels(tokenizer, token_ids, max_len=10):
    """Human-readable labels for each token ID, truncated for display."""
    out = []
    for tid in token_ids:
        s = tokenizer.decode_token(tid)
        s = s.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "_")
        if len(s) > max_len:
            s = s[:max_len] + ".."
        out.append(s)
    return out


def slide_header(plt, fig, slide_num, title, subtitle=""):
    """Draw a consistent header strip at the top of every slide."""
    fig.suptitle(
        f"Slide {slide_num:02d}  —  {title}",
        fontsize=14, fontweight="bold", y=0.98,
    )
    if subtitle:
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=10,
                 color="#444", style="italic")


# ═══════════════════════════════════════════════════════════════
# Slide implementations
# ═══════════════════════════════════════════════════════════════

def slide_01_tokenization(plt, tokenizer, text, token_ids, out_path):
    """Text -> bytes -> merged IDs, as three aligned rows."""
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")

    raw_bytes = list(text.encode("utf-8"))
    n_bytes = len(raw_bytes)
    n_toks = len(token_ids)

    slide_header(plt, fig, 1, "Tokenization",
                 f"{len(text)} chars  ->  {n_bytes} bytes  ->  "
                 f"{n_toks} BPE tokens  (compression {n_bytes/max(n_toks,1):.2f}x)")

    # Row 1: original characters
    x_char = 0.05
    for ch in text:
        ax.text(x_char, 0.78, repr(ch)[1:-1], ha="left", va="center",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#e6f2ff", edgecolor="#4a90d9"))
        x_char += 0.9 / max(len(text), 1)
    ax.text(0.01, 0.78, "chars:", ha="left", va="center",
            fontsize=10, fontweight="bold")

    # Row 2: bytes (as hex)
    x_b = 0.05
    for b in raw_bytes[:40]:  # cap at 40 for display
        ax.text(x_b, 0.50, f"{b:02x}", ha="left", va="center",
                fontsize=9, family="monospace",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#fff4e6", edgecolor="#d98c4a"))
        x_b += 0.9 / min(len(raw_bytes), 40)
    if len(raw_bytes) > 40:
        ax.text(0.96, 0.50, "...", ha="left", va="center", fontsize=10)
    ax.text(0.01, 0.50, "bytes:", ha="left", va="center",
            fontsize=10, fontweight="bold")

    # Row 3: token IDs (with decoded label)
    x_t = 0.05
    labels = token_labels(tokenizer, token_ids, max_len=8)
    for tid, lab in zip(token_ids, labels):
        is_merge = tid >= NUM_BASE
        color = "#d4edda" if is_merge else "#f8d7da"
        edge = "#28a745" if is_merge else "#dc3545"
        ax.text(x_t, 0.18, f"{tid}\n{lab}", ha="left", va="center",
                fontsize=8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor=color, edgecolor=edge))
        x_t += 0.9 / max(n_toks, 1)
    ax.text(0.01, 0.18, "tokens:", ha="left", va="center",
            fontsize=10, fontweight="bold")

    # Legend
    ax.text(0.05, 0.02,
            "Green = BPE merge token (learned)  |  Red = single byte (base vocab)",
            fontsize=9, color="#555", style="italic")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def _heatmap(ax, data, labels_y=None, labels_x=None, cmap="Blues", vmin=None, vmax=None):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if labels_y is not None:
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=8)
    if labels_x is not None:
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    return im


def slide_02_embeddings(plt, store, labels, out_path):
    emb = store["embeddings"][0].numpy()  # (T, d_model)
    T, d = emb.shape
    show_d = min(32, d)

    fig, ax = plt.subplots(figsize=(12, 4 + 0.2 * T))
    slide_header(plt, fig, 2, "Token embeddings",
                 f"Each token ID is looked up in an Embedding matrix to produce a "
                 f"{d}-dim vector. Showing first {show_d} dims.")

    im = _heatmap(ax, emb[:, :show_d], labels_y=labels, cmap="RdBu_r",
                  vmin=-abs(emb).max(), vmax=abs(emb).max())
    ax.set_xlabel(f"Embedding dimension (first {show_d} of {d})")
    ax.set_ylabel("Token")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_03_qkv(plt, store, labels, head, out_path):
    # q/k/v: (B, n_heads, T, d_head)
    q = store["q"][0, head].numpy()
    k = store["k"][0, head].numpy()
    v = store["v"][0, head].numpy()
    T, d_head = q.shape

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5 + 0.15 * T))
    slide_header(plt, fig, 3, f"Q, K, V projections (layer head {head})",
                 f"Each token x is projected into three {d_head}-dim vectors via "
                 f"separate learned matrices. Q asks questions, K advertises content, V carries info.")

    vmax = max(abs(q).max(), abs(k).max(), abs(v).max())
    for ax, data, name in zip(axes, [q, k, v], ["Q (query)", "K (key)", "V (value)"]):
        im = _heatmap(ax, data, labels_y=labels, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"d_head = {d_head}")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_04_scores_raw(plt, store, labels, head, out_path):
    scores = store["scores_raw"][0, head].numpy()  # (T, T)
    fig, ax = plt.subplots(figsize=(8, 7))
    slide_header(plt, fig, 4, "Attention scores (raw)",
                 "Q . K^T / sqrt(d_head)   — pairwise similarity BEFORE causal mask. "
                 "Upper triangle is still visible here.")

    vmax = abs(scores).max()
    im = _heatmap(ax, scores, labels_y=labels, labels_x=labels,
                  cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Key  (token j)")
    ax.set_ylabel("Query  (token i)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_05_scores_masked(plt, store, labels, head, out_path):
    masked = store["scores_masked"][0, head].numpy()
    # -inf won't render; clip for display only
    display = masked.copy()
    finite = display[~(display == float("-inf"))]
    vmax = abs(finite).max() if finite.size else 1.0
    display[display == float("-inf")] = float("nan")

    import numpy as np
    fig, ax = plt.subplots(figsize=(8, 7))
    slide_header(plt, fig, 5, "Causal mask applied",
                 "Upper triangle set to -inf so it contributes zero weight after softmax. "
                 "Token i can only 'see' tokens 0..i.")

    cmap = plt.cm.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")  # grey for masked positions
    im = ax.imshow(display, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Key (token j)")
    ax.set_ylabel("Query (token i)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_06_attn_weights(plt, store, labels, head, out_path):
    w = store["attn_weights"][0, head].numpy()  # (T, T)
    fig, ax = plt.subplots(figsize=(8, 7))
    slide_header(plt, fig, 6, "Softmax -> attention weights",
                 "Each ROW is a probability distribution (sums to 1). "
                 "Entry (i,j) = how much token i attends to token j.")

    im = _heatmap(ax, w, labels_y=labels, labels_x=labels, cmap="viridis", vmin=0, vmax=w.max())
    ax.set_xlabel("Key (token j)")
    ax.set_ylabel("Query (token i)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate argmax per row
    import numpy as np
    for i, row in enumerate(w):
        j = int(np.argmax(row))
        ax.text(j, i, "*", ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_07_value_sum(plt, store, labels, head, query_pos, out_path):
    """Show the weighted value sum for one query position: output_i = sum_j w_ij * v_j."""
    w = store["attn_weights"][0, head, query_pos].numpy()   # (T,)
    v = store["v"][0, head].numpy()                          # (T, d_head)
    out = (w[:, None] * v).sum(axis=0)                       # (d_head,)
    T, d_head = v.shape

    fig, axes = plt.subplots(2, 1, figsize=(12, 6),
                             gridspec_kw={"height_ratios": [1, 2]})
    slide_header(plt, fig, 7, f"Weighted value sum  (query = '{labels[query_pos]}')",
                 "output = sum_j  attn_weight(i,j)  *  V[j]")

    # Top: attention weights as bar chart
    axes[0].bar(range(T), w, color="#08519c")
    axes[0].set_xticks(range(T))
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("attn weight")
    axes[0].set_title("Weights  w_ij  for i = " + labels[query_pos],
                      fontsize=10)
    axes[0].grid(True, axis="y", alpha=0.3)

    # Bottom: V matrix stacked with output row highlighted
    import numpy as np
    full = np.vstack([v, out[None, :]])
    full_labels = labels + ["= OUTPUT"]
    vmax = abs(full).max()
    im = axes[1].imshow(full, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_yticks(range(len(full_labels)))
    axes[1].set_yticklabels(full_labels, fontsize=8)
    axes[1].axhline(T - 0.5, color="black", linewidth=2)
    axes[1].set_xlabel(f"d_head = {d_head}")
    axes[1].set_title("V rows (top) + output row (bottom) for this query",
                      fontsize=10)
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_08_all_heads(plt, store, labels, out_path):
    w = store["attn_weights"][0].numpy()   # (n_heads, T, T)
    n_heads = w.shape[0]

    cols = min(n_heads, 3)
    rows = (n_heads + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = axes.flatten() if n_heads > 1 else [axes]

    slide_header(plt, fig, 8, f"All {n_heads} attention heads side by side",
                 "Each head learns a different kind of relationship "
                 "(e.g. previous-token, syntactic, semantic).")

    for h in range(n_heads):
        ax = axes[h]
        im = ax.imshow(w[h], aspect="auto", cmap="viridis", vmin=0, vmax=w.max())
        ax.set_title(f"Head {h}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_09_ffn_delta(plt, store, labels, out_path):
    pre = store["ffn_input"][0].numpy()    # (T, d_model)
    post = store["ffn_output"][0].numpy()  # (T, d_model)
    T, d = pre.shape
    show_d = min(32, d)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4 + 0.2 * T))
    slide_header(plt, fig, 9, "Feed-forward network  (before / delta / after)",
                 "After attention routes information, the FFN computes on it "
                 "position-wise. This is where factual knowledge lives.")

    vmax = max(abs(pre).max(), abs(post).max())
    delta = post - pre
    dmax = abs(delta).max()

    for ax, data, title, vm in zip(
        axes, [pre, delta, post],
        ["FFN input (post-norm)", "Delta (output - input)", "FFN output"],
        [vmax, dmax, vmax],
    ):
        im = _heatmap(ax, data[:, :show_d], labels_y=labels,
                      cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"first {show_d} dims")
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_10_logits_topk(plt, model, tokenizer, idx, device, temperature, top_k, top_p, out_path):
    with torch.no_grad():
        logits, _ = model(idx)  # (1, vocab_size)

    logits = logits.squeeze(0).cpu()
    probs_raw = F.softmax(logits, dim=-1)

    # Apply temperature + top-k + top-p to show the filtered distribution
    tl = logits / max(temperature, 1e-8)
    if top_k > 0:
        k = min(top_k, tl.size(-1))
        topk_vals, _ = torch.topk(tl, k)
        tl = tl.masked_fill(tl < topk_vals[-1], float("-inf"))
    probs_filtered = F.softmax(tl, dim=-1)

    top_n = 20
    top_vals, top_ids = torch.topk(probs_raw, top_n)
    filt_vals = probs_filtered[top_ids]

    labels = [tokenizer.decode_token(int(tid)).replace("\n", "\\n").replace(" ", "_")
              for tid in top_ids.tolist()]
    labels = [l[:10] + ".." if len(l) > 12 else l for l in labels]

    fig, ax = plt.subplots(figsize=(12, 5))
    slide_header(plt, fig, 10, "Next-token distribution (top 20)",
                 f"Temperature={temperature}, top_k={top_k}, top_p={top_p}.  "
                 "Blue = raw softmax.  Orange = after temp+top_k+top_p filtering.")

    import numpy as np
    x = np.arange(top_n)
    ax.bar(x - 0.2, top_vals.numpy(), 0.4, label="raw", color="#08519c")
    ax.bar(x + 0.2, filt_vals.numpy(), 0.4, label="sampled-from", color="#e6550d")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("probability")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_11_sampling_rollout(plt, model, tokenizer, idx, device, temperature, top_k, top_p, out_path):
    """Generate 10 tokens step-by-step, recording top-5 candidates + chosen at each step."""
    steps = []  # list of (chosen_id, top5_ids, top5_probs)
    cur = idx.clone()
    import numpy as np

    with torch.no_grad():
        for _ in range(10):
            crop = cur[:, -model.config.max_seq_len:]
            logits, _ = model(crop)
            logits_sq = logits.squeeze(0).cpu()

            # For display: top-5 from raw distribution
            probs_disp = F.softmax(logits_sq, dim=-1)
            top5_vals, top5_ids = torch.topk(probs_disp, 5)

            # For sampling: use the shared helper with temp/top_k/top_p
            chosen = _sample_from_logits(logits, temperature, top_k, top_p)
            chosen_id = int(chosen.item())

            steps.append((chosen_id, top5_ids.tolist(), top5_vals.tolist()))
            cur = torch.cat([cur, chosen], dim=1)

    n_steps = len(steps)
    fig, ax = plt.subplots(figsize=(14, 6))
    slide_header(plt, fig, 11, "Sampling rollout — 10 decode steps",
                 "At each step: top-5 candidates (with probability), the CHOSEN token is highlighted.")

    ax.set_xlim(-0.5, n_steps - 0.5)
    ax.set_ylim(-0.5, 5.5)
    ax.invert_yaxis()
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels([f"step {i+1}" for i in range(n_steps)], fontsize=9)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"rank {r+1}" for r in range(5)], fontsize=9)
    ax.grid(True, alpha=0.3)

    for s, (chosen_id, top5_ids, top5_probs) in enumerate(steps):
        for r, (tid, p) in enumerate(zip(top5_ids, top5_probs)):
            label = tokenizer.decode_token(tid).replace("\n", "\\n").replace(" ", "_")
            if len(label) > 8:
                label = label[:8] + ".."
            is_chosen = (tid == chosen_id)
            facecolor = "#e6550d" if is_chosen else "#e6f2ff"
            textcolor = "white" if is_chosen else "black"
            ax.text(s, r, f"{label}\n{p:.2f}", ha="center", va="center",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor=facecolor,
                              edgecolor="#08519c" if is_chosen else "#4a90d9",
                              linewidth=2 if is_chosen else 1),
                    color=textcolor,
                    fontweight="bold" if is_chosen else "normal")

    # Show the resulting text at the bottom
    generated = tokenizer.decode(cur[0].tolist())
    ax.text(0.5, 1.08, f'Generated: "{generated[:120]}"',
            transform=ax.transAxes, ha="center", fontsize=10,
            style="italic", color="#333")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


# ═══════════════════════════════════════════════════════════════
# Raschka-book alignment slides (12-16)
# ═══════════════════════════════════════════════════════════════

def slide_12_positional_rope(plt, model, out_path):
    """RoPE cos/sin frequency table + the effect of position on a single Q vector.

    Raschka Ch 2 devotes substantial space to positional encoding. Our model uses
    RoPE instead of the sinusoidal scheme in the book, but the underlying idea is
    the same: position modifies the Q/K vectors in a way the model can decode.
    """
    import numpy as np
    rope = model.blocks[0].attn.rope
    cos_table = rope.cos_cached.detach().cpu().numpy()   # (max_seq_len, d_head/2)
    sin_table = rope.sin_cached.detach().cpu().numpy()

    max_pos = min(64, cos_table.shape[0])
    d_half = cos_table.shape[1]
    cos_show = cos_table[:max_pos]
    sin_show = sin_table[:max_pos]

    fig = plt.figure(figsize=(14, 6))
    slide_header(plt, fig, 12, "Positional encoding — RoPE",
                 "Low-index dims rotate slowly (long-range), high-index dims rotate "
                 "quickly (short-range). Q and K get rotated by pos x theta_i.")

    # Left: cos table heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(cos_show, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax1.set_title("cos(pos . theta_i)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("frequency index i")
    ax1.set_ylabel("position")
    plt.colorbar(im, ax=ax1, shrink=0.7)

    # Middle: sin table heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    im = ax2.imshow(sin_show, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax2.set_title("sin(pos . theta_i)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("frequency index i")
    plt.colorbar(im, ax=ax2, shrink=0.7)

    # Right: rotation of a single unit Q at several positions
    ax3 = fig.add_subplot(1, 3, 3)
    pair_i = 0  # use lowest-freq pair for a visible rotation
    positions = [0, 2, 4, 8, 16, 32]
    positions = [p for p in positions if p < max_pos]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(positions)))
    for p, color in zip(positions, colors):
        c, s = cos_table[p, pair_i], sin_table[p, pair_i]
        # Rotate the unit vector (1, 0)
        x, y = c, s
        ax3.arrow(0, 0, x, y, head_width=0.04, length_includes_head=True,
                  color=color, linewidth=2)
        ax3.text(x * 1.1, y * 1.1, f"pos {p}", color=color, fontsize=9,
                 fontweight="bold")
    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-1.3, 1.3)
    ax3.set_aspect("equal")
    ax3.axhline(0, color="#bbb", linewidth=0.5)
    ax3.axvline(0, color="#bbb", linewidth=0.5)
    ax3.grid(True, alpha=0.3)
    ax3.set_title(f"Unit (1,0) rotated by pos (dim pair {pair_i})",
                  fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_13_scaling_rationale(plt, out_path):
    """Show why attention divides by sqrt(d_k).

    With fixed-variance q, k ~ N(0,1), the dot product q.k has variance d_k.
    Without /sqrt(d_k), softmax saturates as d_k grows -> vanishing gradients.
    """
    import numpy as np
    torch.manual_seed(0)

    dims = [8, 64, 256, 1024]
    rows = 2
    cols = len(dims)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 6.5))
    slide_header(plt, fig, 13, "Why divide by sqrt(d_head)?",
                 "Top row: softmax(q.k)   Bottom row: softmax(q.k / sqrt(d)). "
                 "As d grows, unscaled softmax collapses to one-hot -> near-zero gradients.")

    T = 10  # sequence length
    for col, d in enumerate(dims):
        q = torch.randn(T, d)
        k = torch.randn(T, d)
        raw = (q @ k.T).numpy()
        scaled = (q @ k.T / math.sqrt(d)).numpy()

        soft_raw = np.exp(raw - raw.max(axis=-1, keepdims=True))
        soft_raw = soft_raw / soft_raw.sum(axis=-1, keepdims=True)
        soft_scaled = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
        soft_scaled = soft_scaled / soft_scaled.sum(axis=-1, keepdims=True)

        ax_top = axes[0, col]
        im = ax_top.imshow(soft_raw, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax_top.set_title(f"d = {d}  (no /sqrt(d))", fontsize=10, fontweight="bold")
        max_p = soft_raw.max()
        ax_top.set_xlabel(f"max entry = {max_p:.2f}", fontsize=9)

        ax_bot = axes[1, col]
        im = ax_bot.imshow(soft_scaled, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax_bot.set_title(f"d = {d}  (with /sqrt(d))", fontsize=10, fontweight="bold")
        max_p = soft_scaled.max()
        ax_bot.set_xlabel(f"max entry = {max_p:.2f}", fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_14_temperature_effect(plt, model, tokenizer, idx, out_path):
    """Same logits, three temperatures. Shows how T sharpens or flattens the
    next-token distribution — Raschka Ch 5.3.
    """
    import numpy as np
    with torch.no_grad():
        logits, _ = model(idx)
    logits = logits.squeeze(0).cpu()

    top_n = 15
    _, top_ids = torch.topk(logits, top_n)
    labels = [tokenizer.decode_token(int(tid)).replace("\n", "\\n").replace(" ", "_")
              for tid in top_ids.tolist()]
    labels = [l[:10] + ".." if len(l) > 12 else l for l in labels]

    temps = [0.3, 1.0, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    slide_header(plt, fig, 14, "Temperature effect on next-token distribution",
                 "T<1 sharpens (deterministic).  T=1 is raw softmax.  T>1 flattens (creative).")

    for ax, T in zip(axes, temps):
        probs = F.softmax(logits / T, dim=-1)[top_ids].numpy()
        ax.bar(range(top_n), probs, color="#08519c")
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        top_prob = probs.max()
        ax.set_title(f"T = {T}   |  max p = {top_prob:.2f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("probability")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_15_greedy_vs_sample(plt, model, tokenizer, idx, device, out_path):
    """Side-by-side generated text: greedy (T~0) vs temperature=1.0 vs temperature=1.5.

    Demonstrates Raschka Ch 5.3: greedy looks repetitive; sampling looks varied.
    """
    n_new = 40
    torch.manual_seed(42)

    with torch.no_grad():
        greedy = model.generate_fast(idx, max_new_tokens=n_new,
                                     temperature=0.01, top_k=1, top_p=1.0)
        mid = model.generate_fast(idx, max_new_tokens=n_new,
                                  temperature=1.0, top_k=40, top_p=0.9)
        hot = model.generate_fast(idx, max_new_tokens=n_new,
                                  temperature=1.5, top_k=40, top_p=0.95)

    texts = [
        ("Greedy  (T ~= 0, top_k=1)", tokenizer.decode(greedy[0].tolist())),
        ("Sample  (T = 1.0, top_k=40, top_p=0.9)", tokenizer.decode(mid[0].tolist())),
        ("Hot     (T = 1.5, top_k=40, top_p=0.95)", tokenizer.decode(hot[0].tolist())),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")
    slide_header(plt, fig, 15, "Decoding strategies on the same prompt",
                 "Greedy always picks argmax -> repetitive.  Sampling introduces variety.")

    y = 0.82
    for title, text in texts:
        ax.text(0.02, y, title, fontsize=11, fontweight="bold", color="#08519c",
                family="monospace")
        y -= 0.06
        # Wrap at ~80 chars
        text_disp = text.replace("\n", " // ")[:350]
        # Simple soft-wrap
        wrapped = []
        line = ""
        for word in text_disp.split(" "):
            if len(line) + len(word) > 90:
                wrapped.append(line)
                line = word
            else:
                line = f"{line} {word}".strip()
        if line:
            wrapped.append(line)
        for ln in wrapped[:3]:
            ax.text(0.04, y, ln, fontsize=10, family="monospace", color="#222")
            y -= 0.045
        y -= 0.03

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


def slide_16_param_breakdown(plt, model, out_path):
    """Pie chart of where the model's parameters live.

    Raschka Ch 4 ends with a parameter-count table. This is the visual version.
    """
    buckets = {
        "token_emb / lm_head (tied)": 0,
        "attn: QKV proj":            0,
        "attn: out proj":            0,
        "FFN: gate+up+down":         0,
        "RMSNorm":                   0,
        "other":                     0,
    }
    seen = set()
    for name, p in model.named_parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n = p.numel()
        if "token_emb" in name or "lm_head" in name:
            buckets["token_emb / lm_head (tied)"] += n
        elif "qkv_proj" in name:
            buckets["attn: QKV proj"] += n
        elif "out_proj" in name:
            buckets["attn: out proj"] += n
        elif "gate_proj" in name or "up_proj" in name or "down_proj" in name:
            buckets["FFN: gate+up+down"] += n
        elif "norm" in name.lower() or "weight" in name and p.dim() == 1:
            buckets["RMSNorm"] += n
        else:
            buckets["other"] += n

    total = sum(buckets.values())
    # Drop zero buckets
    items = [(k, v) for k, v in buckets.items() if v > 0]
    labels = [f"{k}\n{v/1e6:.2f}M ({v/total*100:.1f}%)" for k, v in items]
    sizes = [v for _, v in items]
    colors = ["#08519c", "#3182bd", "#6baed6", "#e6550d", "#fd8d3c", "#bcbddc"]

    fig, ax = plt.subplots(figsize=(10, 7))
    slide_header(plt, fig, 16, f"Parameter breakdown  ({total/1e6:.2f}M total)",
                 "Where the model's weights live. FFN usually dominates — "
                 "it's where factual knowledge is stored.")

    wedges, _ = ax.pie(
        sizes, labels=labels, startangle=90,
        colors=colors[:len(items)],
        wedgeprops=dict(linewidth=2, edgecolor="white"),
        textprops=dict(fontsize=10),
    )
    ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {out_path}")


# ═══════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════

def teach(checkpoint_path, text, output_dir, layer, head, query_pos,
          temperature, top_k, top_p):
    plt = _get_plt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer, config = load_model(checkpoint_path, device)

    # Tokenize
    token_ids = tokenizer.encode(text, add_special=False)
    if len(token_ids) < 2:
        print("ERROR: Prompt tokenizes to fewer than 2 tokens. Use a longer prompt.")
        sys.exit(1)
    if len(token_ids) > config.max_seq_len:
        token_ids = token_ids[:config.max_seq_len]
        text = tokenizer.decode(token_ids)

    labels = token_labels(tokenizer, token_ids, max_len=8)
    idx = torch.tensor([token_ids], device=device)

    # Resolve negative query_pos (Python-style: -1 = last token)
    if query_pos < 0:
        query_pos = len(token_ids) - 1
    query_pos = min(max(0, query_pos), len(token_ids) - 1)
    layer = min(max(0, layer), config.n_layers - 1)
    head = min(max(0, head), config.n_heads - 1)

    print(f"\nPrompt: {text!r}")
    print(f"Tokens ({len(token_ids)}): {labels}")
    print(f"Target: layer={layer}, head={head}, query_pos={query_pos} ('{labels[query_pos]}')")

    # Capture & run forward
    capture = ForwardCapture(model, target_layer=layer)
    with torch.no_grad():
        model(idx)
    capture.remove()

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nWriting slides to {output_dir}/ ...")

    p = lambda name: os.path.join(output_dir, name)

    slide_01_tokenization(plt, tokenizer, text, token_ids, p("01_tokenization.png"))
    slide_02_embeddings(plt, capture.store, labels, p("02_embeddings.png"))
    slide_03_qkv(plt, capture.store, labels, head, p("03_qkv.png"))
    slide_04_scores_raw(plt, capture.store, labels, head, p("04_scores_raw.png"))
    slide_05_scores_masked(plt, capture.store, labels, head, p("05_scores_masked.png"))
    slide_06_attn_weights(plt, capture.store, labels, head, p("06_attn_weights.png"))
    slide_07_value_sum(plt, capture.store, labels, head, query_pos, p("07_value_sum.png"))
    slide_08_all_heads(plt, capture.store, labels, p("08_all_heads.png"))
    slide_09_ffn_delta(plt, capture.store, labels, p("09_ffn_delta.png"))
    slide_10_logits_topk(plt, model, tokenizer, idx, device,
                         temperature, top_k, top_p, p("10_logits_topk.png"))
    slide_11_sampling_rollout(plt, model, tokenizer, idx, device,
                              temperature, top_k, top_p, p("11_sampling_rollout.png"))

    # Raschka-book alignment slides
    slide_12_positional_rope(plt, model, p("12_positional_rope.png"))
    slide_13_scaling_rationale(plt, p("13_scaling_rationale.png"))
    slide_14_temperature_effect(plt, model, tokenizer, idx, p("14_temperature_effect.png"))
    slide_15_greedy_vs_sample(plt, model, tokenizer, idx, device, p("15_greedy_vs_sample.png"))
    slide_16_param_breakdown(plt, model, p("16_param_breakdown.png"))

    print(f"\nAll 16 slides saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="NanoLLM — step-by-step teaching slides")
    parser.add_argument("--text", default="The cat sat on the")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--output_dir", default="teaching_plots")
    parser.add_argument("--layer", type=int, default=0,
                        help="Which layer to visualize in detail")
    parser.add_argument("--head", type=int, default=0,
                        help="Which head (within the chosen layer) to focus on")
    parser.add_argument("--query_pos", type=int, default=-1,
                        help="Query position for the value-sum slide (default: last token)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    teach(
        checkpoint_path=args.checkpoint,
        text=args.text,
        output_dir=args.output_dir,
        layer=args.layer,
        head=args.head,
        query_pos=args.query_pos,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()
