"""Visualise attention patterns from a trained LLM101.

Generates:
  1. Attention heatmaps per layer/head (which tokens attend to which)
  2. Attention rollout (effective attention through all layers combined)
  3. Token-level analysis (what a specific token attends to)

Great for webinar slides — shows the model learning syntax and semantics.

Usage:
    python visualise.py
    python visualise.py --text "To be or not to be that is the question"
    python visualise.py --checkpoint checkpoints/epoch_015.pt
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F

from config import NanoLLMConfig, require_cuda
from tokenizer import BPETokenizer
from model import NanoLLM


# ── Hook-based attention extraction ─────────────────────────

class AttentionCapture:
    """Captures attention weights from all layers via forward hooks."""

    def __init__(self, model: NanoLLM):
        self.attention_maps = []
        self._hooks = []
        # Register hooks on every attention module
        for block in model.blocks:
            hook = block.attn.register_forward_hook(self._make_hook())
            self._hooks.append(hook)

    def _make_hook(self):
        maps = self.attention_maps

        def hook_fn(module, input, output):
            # Re-compute attention weights (they aren't returned by default)
            x = input[0]
            B, T, C = x.shape
            qkv = module.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, T, module.n_heads, module.d_head).transpose(1, 2)
            k = k.view(B, T, module.n_heads, module.d_head).transpose(1, 2)

            q = module.rope(q, T)
            k = module.rope(k, T)

            scale = 1.0 / math.sqrt(module.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores.masked_fill(
                module.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            attn_weights = F.softmax(scores, dim=-1)
            maps.append(attn_weights.detach().cpu())

        return hook_fn

    def clear(self):
        self.attention_maps = []

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def compute_attention_rollout(attention_maps):
    """Compute attention rollout — effective attention through all layers.

    Abnar & Zuidema (2020): multiply attention matrices across layers
    to see the total attention flow from input to output.
    """
    # Start with identity (each token fully attends to itself)
    result = None
    for attn in attention_maps:
        # Average across heads for this layer
        attn_avg = attn.mean(dim=1)  # (B, T, T)
        # Add residual connection effect (identity matrix)
        attn_with_residual = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.size(-1))
        # Normalize rows
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)
        # Chain: multiply through layers
        if result is None:
            result = attn_with_residual
        else:
            result = torch.matmul(attn_with_residual, result)
    return result


def plot_attention_heatmap(weights, token_labels, title, save_path):
    """Plot a single attention heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(weights, cmap="Blues", aspect="auto")

    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(token_labels, fontsize=9)

    ax.set_xlabel("Key (attends TO)", fontsize=11)
    ax.set_ylabel("Query (attends FROM)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def visualise(
    checkpoint_path: str,
    text: str,
    output_dir: str = "attention_plots",
):
    device = require_cuda()

    # Load model
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config: NanoLLMConfig = ckpt["config"]

    tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
    tokenizer.load(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size

    model = NanoLLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Tokenise input
    token_ids = tokenizer.encode(text, add_special=False)
    token_labels = [tokenizer.decode_token(tid) for tid in token_ids]
    # Truncate labels for display
    token_labels = [t[:12] for t in token_labels]

    idx = torch.tensor([token_ids], device=device)
    print(f"Input: \"{text}\"")
    print(f"Tokens ({len(token_ids)}): {token_labels}")

    # Capture attention
    capture = AttentionCapture(model)
    with torch.no_grad():
        model(idx)
    capture.remove_hooks()

    os.makedirs(output_dir, exist_ok=True)

    # ── Plot individual layer/head attention maps ───────────
    n_layers = len(capture.attention_maps)
    n_heads = capture.attention_maps[0].size(1)

    print(f"\nGenerating attention heatmaps ({n_layers} layers × {n_heads} heads)...")

    # Plot a selection (all heads for first, middle, last layer)
    layers_to_plot = [0, n_layers // 2, n_layers - 1]
    for layer_idx in layers_to_plot:
        attn = capture.attention_maps[layer_idx][0]  # (n_heads, T, T)
        for head_idx in range(n_heads):
            weights = attn[head_idx].numpy()
            plot_attention_heatmap(
                weights,
                token_labels,
                f"Layer {layer_idx}, Head {head_idx}",
                os.path.join(output_dir, f"layer{layer_idx}_head{head_idx}.png"),
            )

    # ── Plot head-averaged attention per layer ──────────────
    print("\nGenerating head-averaged attention maps...")
    for layer_idx in range(n_layers):
        attn = capture.attention_maps[layer_idx][0]  # (n_heads, T, T)
        avg_weights = attn.mean(dim=0).numpy()
        plot_attention_heatmap(
            avg_weights,
            token_labels,
            f"Layer {layer_idx} — Head Average",
            os.path.join(output_dir, f"layer{layer_idx}_avg.png"),
        )

    # ── Plot attention rollout ──────────────────────────────
    print("\nComputing attention rollout...")
    rollout = compute_attention_rollout(capture.attention_maps)
    rollout_weights = rollout[0].numpy()
    plot_attention_heatmap(
        rollout_weights,
        token_labels,
        "Attention Rollout (effective attention through all layers)",
        os.path.join(output_dir, "rollout.png"),
    )

    print(f"\nAll plots saved to {output_dir}/")
    print(f"Total files: {len(os.listdir(output_dir))}")


def main():
    parser = argparse.ArgumentParser(description="LLM101 Attention Visualisation")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--text", default="To be or not to be that is the question")
    parser.add_argument("--output_dir", default="attention_plots")
    args = parser.parse_args()

    visualise(args.checkpoint, args.text, args.output_dir)


if __name__ == "__main__":
    main()
