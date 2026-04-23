"""Collect tensor data for the animated transformer visualization.

Runs a single hooked forward pass across ALL layers and collects:
  - attention weights (post-softmax, post-mask) for every layer and head
  - hidden state L2 norms after each layer's residual connection
  - tensor shape strings at each stage

Returns a JSON-serializable dict ready to inject into the HTML template.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from model import NanoLLM


def collect_viz_data(
    model: NanoLLM,
    tokenizer: BPETokenizer,
    text: str,
    config: NanoLLMConfig,
) -> dict:
    """Run a hooked forward pass and collect visualization data.

    Args:
        model: The NanoLLM model (eval mode, any device).
        tokenizer: Trained BPE tokenizer.
        text: Input text to visualize.
        config: Model config (for d_model, n_layers, etc.).

    Returns:
        JSON-serializable dict matching the animation data structure.
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    # ── Tokenize ──
    token_ids = tokenizer.encode(text, add_special=False)
    token_ids = token_ids[:config.max_seq_len]
    token_labels = [tokenizer.decode_token(tid) for tid in token_ids]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    B, T = input_ids.shape

    # ── Storage for hooked data ──
    hooks = []
    hidden_norms = {}   # layer_idx -> float
    attn_data = {}      # layer_idx -> {weights, shapes}
    ffn_shapes = {}     # layer_idx -> {attn_out, ffn_out, output}

    # ── Hook 1: TransformerBlock output — capture hidden-state norm ──
    def _make_block_hook(layer_idx):
        def hook(module, inp, out):
            # out is (x, new_kv) tuple from TransformerBlock.forward
            x = out[0]
            norm = x.detach().float().norm(dim=-1).mean().item()
            hidden_norms[layer_idx] = norm
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.register_forward_hook(_make_block_hook(i)))

    # ── Hook 2: CausalSelfAttention pre-hook — recompute attention weights ──
    #
    # We use a pre-hook on the attention module to capture the normed input,
    # then manually compute Q, K, V, RoPE, masking, and softmax to get the
    # attention weight matrix. This avoids modifying model.py.
    def _make_attn_hook(layer_idx):
        def hook(module, inp):
            x_norm = inp[0]
            b, t, c = x_norm.shape

            qkv = module.qkv_proj(x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(b, t, module.n_heads, module.d_head).transpose(1, 2)
            k = k.view(b, t, module.n_heads, module.d_head).transpose(1, 2)
            v = v.view(b, t, module.n_heads, module.d_head).transpose(1, 2)

            q_rot = module.rope(q, t, start_pos=0)
            k_rot = module.rope(k, t, start_pos=0)

            scale = 1.0 / math.sqrt(module.d_head)
            scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * scale
            masked = scores.masked_fill(
                module.causal_mask[:, :, :t, :t] == 0, float("-inf")
            )
            weights = F.softmax(masked, dim=-1)

            attn_data[layer_idx] = {
                "weights": weights.detach().cpu(),
                "shapes": {
                    "input": str(tuple(x_norm.shape)),
                    "q": str(tuple(q.shape)),
                    "k": str(tuple(k.shape)),
                    "v": str(tuple(v.shape)),
                },
            }
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.attn.register_forward_pre_hook(_make_attn_hook(i)))

    # ── Hook 3: FeedForward output — capture shapes ──
    def _make_ffn_hook(layer_idx):
        def hook(module, inp, out):
            ffn_shapes[layer_idx] = {
                "attn_out": str(tuple(inp[0].shape)),
                "ffn_out": str(tuple(out.shape)),
                "output": str(tuple(out.shape)),
            }
        return hook

    for i, block in enumerate(model.blocks):
        hooks.append(block.ffn.register_forward_hook(_make_ffn_hook(i)))

    # ── Run the forward pass ──
    with torch.no_grad():
        model(input_ids)

    # ── Clean up hooks ──
    for h in hooks:
        h.remove()

    if was_training:
        model.train()

    # ── Assemble JSON-serializable output ──
    all_layers_data = []
    for i in range(config.n_layers):
        ad = attn_data[i]
        weights_tensor = ad["weights"][0]  # Remove batch dim: (n_heads, T, T)

        attn_weights_list = []
        for head in range(config.n_heads):
            head_matrix = weights_tensor[head].tolist()
            attn_weights_list.append(head_matrix)

        shapes = {**ad["shapes"], **ffn_shapes.get(i, {})}

        all_layers_data.append({
            "layer_idx": i,
            "attn_weights": attn_weights_list,
            "hidden_norm": hidden_norms.get(i, 0.0),
            "shapes": shapes,
        })

    return {
        "tokens": token_labels,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_model": config.d_model,
        "d_head": config.d_head,
        "layers": all_layers_data,
    }
