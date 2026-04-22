"""NanoLLM — Webinar Console (Gradio UI).

A single browser app with four tabs:
  1. Generate      — live token streaming; toggle KV cache on/off to see speed
  2. Teach         — render the 16 step-by-step slides for a chosen prompt
  3. Attention     — per-head heatmap + rollout for the current prompt
  4. Benchmark     — side-by-side generate() vs generate_fast() tok/s

All four tabs reuse the existing teach.py / visualise.py plumbing — no logic
is duplicated. The model is loaded once at startup.

Run:
    python app.py
    python app.py --share           # phone-home for a public URL (webinar mode)
    python app.py --port 7861
    bash run.sh ui
"""

from __future__ import annotations
import argparse
import os
import tempfile
import time

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from model import NanoLLM, _sample_from_logits
from teach import (
    ForwardCapture, token_labels, _get_plt,
    slide_01_tokenization, slide_02_embeddings, slide_03_qkv,
    slide_04_scores_raw, slide_05_scores_masked, slide_06_attn_weights,
    slide_07_value_sum, slide_08_all_heads, slide_09_ffn_delta,
    slide_10_logits_topk, slide_11_sampling_rollout,
    slide_12_positional_rope, slide_13_scaling_rationale,
    slide_14_temperature_effect, slide_15_greedy_vs_sample,
    slide_16_param_breakdown,
)
from visualise import AttentionCapture, compute_attention_rollout, plot_attention_heatmap


# ═══════════════════════════════════════════════════════════════
# Model loading (singleton — load once, share across handlers)
# ═══════════════════════════════════════════════════════════════

_MODEL: NanoLLM | None = None
_TOKENIZER: BPETokenizer | None = None
_CONFIG: NanoLLMConfig | None = None
_STATUS: str = ""  # Banner text shown in the header


def _load_model(checkpoint_path: str = "checkpoints/best.pt") -> None:
    """Load model + tokenizer once. Falls back to random weights if no checkpoint."""
    global _MODEL, _TOKENIZER, _CONFIG, _STATUS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
        tokenizer.load(config.tokenizer_path)
        config.vocab_size = tokenizer.vocab_size
        model = NanoLLM(config).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        epoch = ckpt.get("epoch", "?")
        val_loss = ckpt.get("val_loss", ckpt.get("loss", "?"))
        _STATUS = (
            f"**Loaded:** `{checkpoint_path}` (epoch {epoch}, val_loss={val_loss}) "
            f"· **Device:** `{device}` · **Vocab:** {config.vocab_size} "
            f"· **Params:** {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
        )
    else:
        # Fall back to a random-initialised tiny model so the UI is explorable
        # even before training finishes.
        config = NanoLLMConfig(vocab_size=260)  # Just the byte vocab
        tokenizer = BPETokenizer(target_vocab_size=260)
        model = NanoLLM(config).to(device).eval()
        _STATUS = (
            f"⚠️ **No checkpoint at `{checkpoint_path}`** — using random weights. "
            f"Generation will be garbage. Run `bash run.sh train` first for real output. "
            f"· **Device:** `{device}`"
        )

    _MODEL = model
    _TOKENIZER = tokenizer
    _CONFIG = config


def _require_loaded():
    """Assert the model has been loaded. All handlers depend on this."""
    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call _load_model() first.")
    return _MODEL, _TOKENIZER, _CONFIG


# ═══════════════════════════════════════════════════════════════
# Tab 1: Generate (with token-by-token streaming)
# ═══════════════════════════════════════════════════════════════

def generate_stream(prompt, max_new_tokens, temperature, top_k, top_p, use_cache):
    """Yield the accumulated decoded text after each new token.

    Gradio will update the Textbox once per yield, producing the classic
    typewriter effect that makes autoregressive generation tangible to
    students: they SEE each token appearing one at a time.
    """
    model, tokenizer, config = _require_loaded()
    if not prompt:
        yield "(empty prompt)"
        return

    ids = tokenizer.encode(prompt, add_special=False)
    if not ids:
        yield "(prompt encoded to zero tokens)"
        return

    # Respect context window
    if len(ids) >= config.max_seq_len:
        ids = ids[-(config.max_seq_len - 1):]

    idx = torch.tensor([ids], device=config.device)
    out_ids = list(ids)
    yield tokenizer.decode(out_ids)  # initial state

    t0 = time.time()

    with torch.no_grad():
        if use_cache:
            # Prefill + incremental decode with KV cache
            logits, cache = model(idx)
            for step in range(int(max_new_tokens)):
                if cache[0][0].size(2) >= config.max_seq_len:
                    break
                next_tok = _sample_from_logits(logits, temperature, int(top_k), top_p)
                out_ids.append(int(next_tok.item()))
                yield tokenizer.decode(out_ids)
                logits, cache = model(next_tok, past_kv=cache)
        else:
            # Reference path: re-process the whole context every step
            cur = idx
            for step in range(int(max_new_tokens)):
                crop = cur[:, -config.max_seq_len:]
                logits, _ = model(crop)
                next_tok = _sample_from_logits(logits, temperature, int(top_k), top_p)
                cur = torch.cat([cur, next_tok], dim=1)
                out_ids.append(int(next_tok.item()))
                yield tokenizer.decode(out_ids)

    # Final frame with timing (use the known step count, not the loop var,
    # which may not be in scope if max_new_tokens was 0).
    elapsed = time.time() - t0
    n_new = len(out_ids) - len(ids)
    if n_new > 0:
        tps = n_new / max(elapsed, 1e-6)
        final_text = tokenizer.decode(out_ids) + \
            f"\n\n— {n_new} tokens in {elapsed:.2f}s ({tps:.1f} tok/s, " + \
            ("with cache" if use_cache else "no cache") + ")"
        yield final_text


# ═══════════════════════════════════════════════════════════════
# Tab 2: Teach (render 16 slides for any prompt)
# ═══════════════════════════════════════════════════════════════

def render_teach(prompt, layer, head, query_pos, temperature, top_k, top_p):
    """Render the 16-slide walkthrough for a user-chosen prompt + layer + head."""
    model, tokenizer, config = _require_loaded()
    plt_ = _get_plt()

    if not prompt:
        return [], "Please enter a prompt."

    token_ids = tokenizer.encode(prompt, add_special=False)
    if len(token_ids) < 2:
        return [], f"Prompt tokenizes to only {len(token_ids)} token(s). Need ≥ 2."
    if len(token_ids) > config.max_seq_len:
        token_ids = token_ids[:config.max_seq_len]
        prompt = tokenizer.decode(token_ids)

    # Resolve indices
    query_pos = int(query_pos)
    if query_pos < 0:
        query_pos = len(token_ids) - 1
    query_pos = min(max(0, query_pos), len(token_ids) - 1)
    layer = min(max(0, int(layer)), config.n_layers - 1)
    head = min(max(0, int(head)), config.n_heads - 1)

    labels = token_labels(tokenizer, token_ids, max_len=8)
    idx = torch.tensor([token_ids], device=config.device)

    # Run forward with hooks
    capture = ForwardCapture(model, target_layer=layer)
    with torch.no_grad():
        model(idx)
    capture.remove()

    # Write PNGs to a fresh temp dir (avoid caching old renders)
    outdir = tempfile.mkdtemp(prefix="nanollm_ui_slides_")
    p = lambda n: os.path.join(outdir, n)

    slide_01_tokenization(plt_, tokenizer, prompt, token_ids, p("01_tokenization.png"))
    slide_02_embeddings(plt_, capture.store, labels, p("02_embeddings.png"))
    slide_03_qkv(plt_, capture.store, labels, head, p("03_qkv.png"))
    slide_04_scores_raw(plt_, capture.store, labels, head, p("04_scores_raw.png"))
    slide_05_scores_masked(plt_, capture.store, labels, head, p("05_scores_masked.png"))
    slide_06_attn_weights(plt_, capture.store, labels, head, p("06_attn_weights.png"))
    slide_07_value_sum(plt_, capture.store, labels, head, query_pos, p("07_value_sum.png"))
    slide_08_all_heads(plt_, capture.store, labels, p("08_all_heads.png"))
    slide_09_ffn_delta(plt_, capture.store, labels, p("09_ffn_delta.png"))
    slide_10_logits_topk(plt_, model, tokenizer, idx, config.device,
                         temperature, int(top_k), top_p, p("10_logits_topk.png"))
    slide_11_sampling_rollout(plt_, model, tokenizer, idx, config.device,
                              temperature, int(top_k), top_p, p("11_sampling_rollout.png"))
    slide_12_positional_rope(plt_, model, p("12_positional_rope.png"))
    slide_13_scaling_rationale(plt_, p("13_scaling_rationale.png"))
    slide_14_temperature_effect(plt_, model, tokenizer, idx, p("14_temperature_effect.png"))
    slide_15_greedy_vs_sample(plt_, model, tokenizer, idx, config.device, p("15_greedy_vs_sample.png"))
    slide_16_param_breakdown(plt_, model, p("16_param_breakdown.png"))

    # Build gallery entries: (image_path, caption)
    captions = [
        "01 · Tokenization — text → bytes → BPE → token IDs",
        "02 · Embeddings — token IDs → dense vectors",
        f"03 · Q, K, V — projections for layer head {head}",
        f"04 · Attention scores (raw, before mask)",
        "05 · Causal mask applied (upper triangle → −∞)",
        "06 · Softmax → attention weights (rows sum to 1)",
        f"07 · Weighted value sum for query '{labels[query_pos]}'",
        "08 · All heads side-by-side (this layer)",
        "09 · FFN: before vs delta vs after",
        "10 · Next-token distribution (top 20)",
        "11 · Sampling rollout — 10 decode steps",
        "12 · RoPE positional encoding",
        "13 · Why /√d_head (scaling rationale)",
        "14 · Temperature effect on distribution",
        "15 · Greedy vs sampling output",
        "16 · Parameter breakdown",
    ]
    gallery = list(zip(sorted(os.listdir(outdir)), captions))
    gallery = [(os.path.join(outdir, f), c) for f, c in gallery]

    status = (
        f"Rendered 16 slides for prompt '{prompt[:40]}{'…' if len(prompt)>40 else ''}' "
        f"({len(token_ids)} tokens) · layer={layer} · head={head} · "
        f"query_pos={query_pos} ('{labels[query_pos]}')"
    )
    return gallery, status


# ═══════════════════════════════════════════════════════════════
# Tab 3: Attention (one head + rollout)
# ═══════════════════════════════════════════════════════════════

def render_attention(prompt, layer, head):
    """Render a single head's heatmap + the attention-rollout heatmap."""
    model, tokenizer, config = _require_loaded()

    if not prompt:
        return None, None, "Please enter a prompt."

    token_ids = tokenizer.encode(prompt, add_special=False)
    if len(token_ids) < 2:
        return None, None, f"Prompt tokenizes to only {len(token_ids)} token(s). Need ≥ 2."
    if len(token_ids) > config.max_seq_len:
        token_ids = token_ids[:config.max_seq_len]

    labels = [tokenizer.decode_token(t).replace("\n", "\\n")[:10] for t in token_ids]
    idx = torch.tensor([token_ids], device=config.device)

    cap = AttentionCapture(model)
    with torch.no_grad():
        model(idx)
    cap.remove_hooks()

    n_layers = len(cap.attention_maps)
    n_heads = cap.attention_maps[0].size(1)
    layer = min(max(0, int(layer)), n_layers - 1)
    head = min(max(0, int(head)), n_heads - 1)

    outdir = tempfile.mkdtemp(prefix="nanollm_ui_attn_")
    head_path = os.path.join(outdir, "head.png")
    rollout_path = os.path.join(outdir, "rollout.png")

    # Single head heatmap
    weights = cap.attention_maps[layer][0, head].numpy()
    plot_attention_heatmap(
        weights, labels, f"Layer {layer}, Head {head}", head_path
    )

    # Rollout across all layers
    rollout = compute_attention_rollout(cap.attention_maps)
    plot_attention_heatmap(
        rollout[0].numpy(), labels,
        f"Attention rollout across {n_layers} layers",
        rollout_path,
    )

    info = (
        f"{len(token_ids)} tokens · layer {layer}/{n_layers-1} · "
        f"head {head}/{n_heads-1}"
    )
    return head_path, rollout_path, info


# ═══════════════════════════════════════════════════════════════
# Tab 4: Benchmark (generate vs generate_fast)
# ═══════════════════════════════════════════════════════════════

def run_bench(prompt, n_tokens):
    """Time both generation paths and return a bar chart + summary text."""
    model, tokenizer, config = _require_loaded()
    n_tokens = int(n_tokens)

    prompt = prompt or "The "
    ids = tokenizer.encode(prompt, add_special=False) or [0]
    idx = torch.tensor([ids], device=config.device)

    # Warm up once so kernel-compile cost isn't charged to the first path
    with torch.no_grad():
        model.generate(idx, max_new_tokens=4, temperature=0.8, top_k=40, top_p=0.9)
        model.generate_fast(idx, max_new_tokens=4, temperature=0.8, top_k=40, top_p=0.9)
    if config.device.type == "cuda":
        torch.cuda.synchronize()

    # No-cache path
    t0 = time.time()
    with torch.no_grad():
        model.generate(idx, max_new_tokens=n_tokens)
    if config.device.type == "cuda":
        torch.cuda.synchronize()
    t_slow = time.time() - t0

    # KV-cache path
    t0 = time.time()
    with torch.no_grad():
        model.generate_fast(idx, max_new_tokens=n_tokens)
    if config.device.type == "cuda":
        torch.cuda.synchronize()
    t_fast = time.time() - t0

    slow_tps = n_tokens / t_slow
    fast_tps = n_tokens / t_fast
    speedup = t_slow / t_fast if t_fast > 0 else float("inf")

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        ["generate()\n(no cache)", "generate_fast()\n(KV cache)"],
        [slow_tps, fast_tps],
        color=["#9ecae1", "#08519c"],
    )
    ax.set_ylabel("tokens / second", fontsize=11)
    ax.set_title(
        f"Generation throughput — {n_tokens} tokens · speedup {speedup:.2f}×",
        fontsize=12, fontweight="bold",
    )
    for bar, val in zip(bars, [slow_tps, fast_tps]):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.02,
                f"{val:.1f} tok/s", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    outpath = os.path.join(tempfile.mkdtemp(prefix="nanollm_ui_bench_"), "bench.png")
    fig.savefig(outpath, dpi=130)
    plt.close(fig)

    summary = (
        f"generate()        : {t_slow*1000:8.1f} ms   ·   {slow_tps:6.1f} tok/s\n"
        f"generate_fast()   : {t_fast*1000:8.1f} ms   ·   {fast_tps:6.1f} tok/s\n"
        f"Speedup           : {speedup:.2f}x"
    )
    return outpath, summary


# ═══════════════════════════════════════════════════════════════
# UI layout
# ═══════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    """Assemble the 4-tab Gradio app. Returns the Blocks object (unlaunched)."""
    # Determine UI ranges from the loaded config
    n_layers = _CONFIG.n_layers if _CONFIG is not None else 6
    n_heads = _CONFIG.n_heads if _CONFIG is not None else 6
    max_seq = _CONFIG.max_seq_len if _CONFIG is not None else 256

    with gr.Blocks(title="NanoLLM — Webinar Console",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            "# NanoLLM — Webinar Console\n"
            "A ~15M-parameter GPT-style transformer, built from scratch. "
            "Four tabs: **Generate** (live streaming), **Teach** (16 explanatory slides), "
            "**Attention** (per-head heatmaps), **Benchmark** (KV-cache speedup)."
        )
        gr.Markdown(_STATUS)

        # ── Tab 1: Generate ──
        with gr.Tab("Generate"):
            gr.Markdown(
                "**Tip:** Toggle the KV cache off to see the slower no-cache path "
                "(the reference implementation used in most tutorials)."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(
                        label="Prompt", value="To be or not to be, ",
                        lines=2, max_lines=4,
                    )
                    with gr.Row():
                        gen_max = gr.Slider(8, max_seq - 1, 100, step=1,
                                            label="max_new_tokens")
                    with gr.Row():
                        gen_temp = gr.Slider(0.05, 2.0, 0.8, step=0.05,
                                             label="temperature")
                        gen_topk = gr.Slider(0, 200, 40, step=1, label="top_k")
                        gen_topp = gr.Slider(0.05, 1.0, 0.9, step=0.05,
                                             label="top_p (nucleus)")
                    gen_cache = gr.Checkbox(True, label="Use KV cache (generate_fast)")
                    gen_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    gen_out = gr.Textbox(
                        label="Output (streaming)", lines=16, show_copy_button=True,
                    )
            gen_btn.click(
                generate_stream,
                inputs=[gen_prompt, gen_max, gen_temp, gen_topk, gen_topp, gen_cache],
                outputs=gen_out,
            )

        # ── Tab 2: Teach ──
        with gr.Tab("Teach"):
            gr.Markdown(
                "Render the 16 step-by-step slides on-the-fly for any prompt. "
                "Changing `layer` / `head` / `query_pos` re-renders slides 02–09 for "
                "that specific slice of the model. Aligned with Raschka's "
                "*Build a Large Language Model (From Scratch)* — see `REFERENCES.md`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    teach_prompt = gr.Textbox(
                        label="Prompt", value="The cat sat on the", lines=2,
                    )
                    teach_layer = gr.Slider(0, n_layers - 1, 0, step=1, label="layer")
                    teach_head = gr.Slider(0, n_heads - 1, 0, step=1, label="head")
                    teach_qpos = gr.Number(value=-1, precision=0,
                                           label="query_pos (-1 = last token)")
                    with gr.Row():
                        teach_temp = gr.Slider(0.05, 2.0, 0.8, step=0.05, label="temp")
                        teach_topk = gr.Slider(0, 200, 40, step=1, label="top_k")
                        teach_topp = gr.Slider(0.05, 1.0, 0.9, step=0.05, label="top_p")
                    teach_btn = gr.Button("Render 16 slides", variant="primary")
                    teach_status = gr.Markdown()
                with gr.Column(scale=2):
                    teach_gallery = gr.Gallery(
                        label="Slides", columns=2, height=700,
                        show_label=False, object_fit="contain",
                    )
            teach_btn.click(
                render_teach,
                inputs=[teach_prompt, teach_layer, teach_head, teach_qpos,
                        teach_temp, teach_topk, teach_topp],
                outputs=[teach_gallery, teach_status],
            )

        # ── Tab 3: Attention ──
        with gr.Tab("Attention"):
            gr.Markdown(
                "Heatmap view: one head's attention matrix + the attention rollout "
                "across all layers (Abnar & Zuidema 2020). "
                "Rows = query position, columns = key position. "
                "Bright cells = strong attention."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    attn_prompt = gr.Textbox(
                        label="Prompt", value="To be or not to be", lines=2,
                    )
                    attn_layer = gr.Slider(0, n_layers - 1, n_layers // 2,
                                           step=1, label="layer")
                    attn_head = gr.Slider(0, n_heads - 1, 0, step=1, label="head")
                    attn_btn = gr.Button("Render attention", variant="primary")
                    attn_status = gr.Markdown()
                with gr.Column(scale=2):
                    with gr.Row():
                        attn_head_img = gr.Image(label="Single head", type="filepath")
                        attn_rollout_img = gr.Image(label="Rollout (all layers)", type="filepath")
            attn_btn.click(
                render_attention,
                inputs=[attn_prompt, attn_layer, attn_head],
                outputs=[attn_head_img, attn_rollout_img, attn_status],
            )

        # ── Tab 4: Benchmark ──
        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Compare `generate()` (no cache, O(T²) per step) vs "
                "`generate_fast()` (KV cache, O(T) per step). "
                "Speedup grows with sequence length."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bench_prompt = gr.Textbox(
                        label="Prompt", value="The ", lines=1,
                    )
                    bench_n = gr.Slider(20, max_seq - 10, 100, step=10,
                                        label="tokens to generate")
                    bench_btn = gr.Button("Run benchmark", variant="primary")
                    bench_summary = gr.Code(
                        label="Result", language=None, interactive=False,
                    )
                with gr.Column(scale=2):
                    bench_img = gr.Image(label="Throughput", type="filepath")
            bench_btn.click(
                run_bench,
                inputs=[bench_prompt, bench_n],
                outputs=[bench_img, bench_summary],
            )

        gr.Markdown(
            "\n---\n"
            "Built on PyTorch · GPT-style decoder · RMSNorm + RoPE + SwiGLU + "
            "Pre-Norm · weight-tied · combined QKV · bf16-capable. "
            "Code: `model.py` (800 lines)."
        )

    return demo


# ═══════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NanoLLM — Gradio webinar console")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true",
                        help="Create a public URL via Gradio's share tunnel")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 to expose on LAN)")
    args = parser.parse_args()

    print("NanoLLM Webinar Console — starting...")
    _load_model(args.checkpoint)
    print(f"  {_STATUS}")

    demo = build_ui()
    # show_api=False works around a gradio-client schema-introspection bug
    # (TypeError on schema traversal in gradio 4.44.x). The UI still works,
    # we just skip the auto-generated API docs page.
    demo.queue().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
