"""LLM101 — Webinar Console (Gradio UI).

A single browser app with seven tabs:
  1. Generate      — live token streaming; toggle KV cache on/off to see speed
  2. Train Reports — render the 16 step-by-step slides for a chosen prompt
  3. Attention     — per-head heatmap + rollout for the current prompt
  4. Benchmark     — side-by-side generate() vs generate_fast() tok/s
  5. Train         — interactive training with LR / dropout / warmup sliders
  6. Effects       — how each hyperparameter shapes the loss curve
  7. Build Steps   — structural diagrams (no trained model needed)

All tabs reuse the existing teach.py / visualise.py / build_viz.py /
effect_viz.py plumbing — no logic is duplicated. The model is loaded once
at startup.

Run:
    python app.py
    python app.py --share           # phone-home for a public URL (webinar mode)
    python app.py --port 7861
    bash run.sh ui
"""

from __future__ import annotations
import argparse
import math
import os
import socket
import tempfile
import time
import warnings

# Silence Gradio 6.0 deprecation warnings that Gradio 5.x emits for APIs that
# still work fine in 5.x (theme= on Blocks, show_api= on launch). Targeted
# pattern avoids swallowing unrelated warnings.
warnings.filterwarnings(
    "ignore",
    message=r".*will be removed in Gradio 6\.0.*",
    category=DeprecationWarning,
)

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from config import NanoLLMConfig, require_cuda
from tokenizer import BPETokenizer
from model import NanoLLM, _sample_from_logits
from train import train_iter, save_loss_curve
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
from visualize_anim import collect_viz_data
import build_viz
import effect_viz


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

    device = require_cuda()  # hard-require CUDA (set NANOLLM_ALLOW_CPU=1 to bypass)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt["config"]
        tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
        tokenizer.load(config.tokenizer_path)
        config.vocab_size = tokenizer.vocab_size
        model = NanoLLM(config).to(device)
        # Checkpoints saved from torch.compile'd models have keys prefixed
        # with "_orig_mod." — strip that prefix so they load into a plain model.
        sd = ckpt["model_state_dict"]
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd)
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
# Tab 4: Train Reports (render 16 slides for any prompt)
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
    try:
        fig.tight_layout()
    except (ValueError, Exception):
        pass

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
# Tab 5: Train (streams training progress into the UI)
# ═══════════════════════════════════════════════════════════════

# Module-level lock so we can't accidentally start two trainings at once.
_TRAINING_LOCK = False


def _build_live_curve(history: list[tuple[int, float]],
                      epochs: list[tuple[int, float, float]]) -> str | None:
    """Render an in-progress loss curve PNG. Returns the path or None."""
    if not history:
        return None
    outdir = tempfile.mkdtemp(prefix="nanollm_train_curve_")
    path = os.path.join(outdir, "curve.png")
    save_loss_curve(history, epochs, path)
    return path


def train_stream(max_epochs_override: int, batch_size_override: int,
                 learning_rate_override: float, dropout_override: float,
                 warmup_steps_override: int,
                 checkpoint_path: str = "checkpoints/best.pt"):
    """Streaming training handler. Yields (log_text, plot_path, status, gen_samples).

    The Gradio UI updates all four outputs once per yielded frame. On completion
    we reload the global _MODEL so the other tabs pick up the new checkpoint.

    `warmup_steps_override=0` means "keep the config default and let auto-scale
    run". Any non-zero value is treated as an explicit user choice and bypasses
    auto-scale (same semantics as `--warmup-steps` on the CLI).
    """
    global _TRAINING_LOCK, _MODEL, _TOKENIZER, _CONFIG

    if _TRAINING_LOCK:
        yield ("Training already in progress — wait for it to finish or restart the app.",
               None, "busy", "")
        return

    _TRAINING_LOCK = True
    log_lines: list[str] = []
    train_history: list[tuple[int, float]] = []
    epoch_history: list[tuple[int, float, float]] = []
    samples_text = ""
    last_plot_path = None

    def as_text():
        # Tail to last 300 lines to keep the textbox responsive
        tail = log_lines[-300:] if len(log_lines) > 300 else log_lines
        return "\n".join(tail)

    try:
        cfg = NanoLLMConfig()
        cfg.max_epochs = int(max_epochs_override)
        cfg.batch_size = int(batch_size_override)
        cfg.learning_rate = float(learning_rate_override)
        cfg.dropout = float(dropout_override)
        if int(warmup_steps_override) > 0:
            cfg.warmup_steps = int(warmup_steps_override)
            setattr(cfg, "_warmup_explicit", True)

        yield (f"Starting training  "
               f"(epochs={cfg.max_epochs}  "
               f"batch={cfg.batch_size}  "
               f"lr={cfg.learning_rate:.0e}  "
               f"dropout={cfg.dropout}  "
               f"warmup={'auto' if not getattr(cfg, '_warmup_explicit', False) else cfg.warmup_steps})",
               None, "running", "")

        last_emit = time.time()
        step_counter = 0

        for evt in train_iter(cfg):
            t = evt["type"]

            if t == "log":
                log_lines.append(evt["msg"])

            elif t == "step":
                step_counter += 1
                train_history.append((evt["global_step"], evt["loss"]))
                # Update the live loss curve every 10 steps
                if step_counter % 10 == 0 and train_history:
                    last_plot_path = _build_live_curve(train_history, epoch_history)
                # Emit a log line only every log_interval steps (same cadence as CLI)
                if evt["batch_idx"] % cfg.log_interval == 0:
                    ppl = math.exp(min(evt["loss"], 20)) if evt["loss"] < 20 else float("inf")
                    log_lines.append(
                        f"  Epoch {evt['epoch']}/{cfg.max_epochs} | "
                        f"Step {evt['batch_idx']}/{evt['total_batches']} | "
                        f"Loss {evt['loss']:.4f} | PPL {ppl:.1f} | "
                        f"LR {evt['lr']:.2e} | {evt['tps']:,.0f} tok/s"
                    )

            elif t == "epoch":
                epoch_history.append((evt["epoch"], evt["train_loss"], evt["val_loss"]))
                log_lines.append("")
                log_lines.append(
                    f"  Epoch {evt['epoch']}/{evt['max_epochs']} done  |  "
                    f"train={evt['train_loss']:.4f}  val={evt['val_loss']:.4f}  "
                    f"(train_ppl={evt['train_ppl']:.1f}  val_ppl={evt['val_ppl']:.1f})  "
                    f"time={evt['elapsed']:.1f}s"
                )
                if evt["samples"]:
                    preview = evt["samples"][0].replace("\n", " / ")[:200]
                    samples_text = (f"Epoch {evt['epoch']} generation sample:\n"
                                    f"  \"{preview}\"")
                # Re-render loss curve now that we have new epoch data
                last_plot_path = _build_live_curve(train_history, epoch_history)

            elif t == "best":
                log_lines.append(f"  * New best val loss {evt['val_loss']:.4f} "
                                 f"saved to {evt['path']}")

            elif t == "done":
                log_lines.append("")
                log_lines.append("=" * 50)
                log_lines.append(f"Training complete.  Best val loss: {evt['best_val_loss']:.4f}")
                log_lines.append(f"Loss curve: {evt['curve_path']}")
                log_lines.append("=" * 50)
                last_plot_path = evt["curve_path"] or last_plot_path

            elif t == "error":
                log_lines.append(f"ERROR: {evt['msg']}")
                yield (as_text(), last_plot_path, "error", samples_text)
                return

            # Throttle UI updates — flush immediately for infrequent events
            # (log, epoch, best, done, error) but rate-limit high-frequency
            # step events to every 0.5s so we don't choke the SSE stream.
            now = time.time()
            if (t in ("log", "epoch", "best", "done", "error")
                or now - last_emit >= 0.5):
                yield (as_text(), last_plot_path, "running", samples_text)
                last_emit = now

        # Training finished — reload the singleton so other tabs see the new model
        try:
            _load_model(checkpoint_path)
            log_lines.append("")
            log_lines.append(f"Model reloaded.  Status: {_STATUS}")
        except Exception as e:
            log_lines.append(f"  (note: couldn't auto-reload model: {e})")

        yield (as_text(), last_plot_path, "done", samples_text)

    finally:
        _TRAINING_LOCK = False


# ═══════════════════════════════════════════════════════════════
# Tab 6: Effects (how each hyperparameter shapes training)
# ═══════════════════════════════════════════════════════════════

def render_effect_panel(param: str) -> tuple[str, str]:
    """Return (markdown_caption, image_path) for one effect selection."""
    outdir = globals().setdefault(
        "_EFFECT_DIR",
        tempfile.mkdtemp(prefix="nanollm_effects_"),
    )
    img, caption = effect_viz.render(param, outdir)
    return caption, img


# ═══════════════════════════════════════════════════════════════
# Tab 7: Build Steps (interactive tutorial)
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Tab 7: Visualize (animated forward pass)
# ═══════════════════════════════════════════════════════════════

_VIZ_TEMPLATE = None  # Lazy-loaded


def _get_viz_template() -> str:
    """Load the HTML template once."""
    global _VIZ_TEMPLATE
    if _VIZ_TEMPLATE is None:
        template_path = os.path.join(os.path.dirname(__file__), "templates", "visualize.html")
        with open(template_path, "r", encoding="utf-8") as f:
            _VIZ_TEMPLATE = f.read()
    return _VIZ_TEMPLATE


def render_visualization(prompt: str) -> str:
    """Collect tensors and return the animated HTML visualization.

    Gradio's gr.HTML does NOT execute <script> tags (static HTML only).
    We wrap the visualization in an <iframe srcdoc="..."> which creates
    a sandboxed document that does execute scripts.
    """
    import json
    import html as html_lib
    model, tokenizer, config = _require_loaded()
    data = collect_viz_data(model, tokenizer, prompt, config)
    template = _get_viz_template()
    json_str = json.dumps(data)
    inner_html = template.replace("{{VIZ_DATA_JSON}}", json_str)
    # Wrap in a full HTML document for the iframe
    doc = (
        '<!DOCTYPE html><html><head><meta charset="UTF-8">'
        '<style>body{margin:0;background:#0f172a;}</style></head>'
        f'<body>{inner_html}</body></html>'
    )
    # Escape for srcdoc attribute (HTML entity encoding)
    escaped = html_lib.escape(doc, quote=True)
    iframe = (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:750px;border:none;border-radius:8px;" '
        f'sandbox="allow-scripts"></iframe>'
    )
    return iframe


# ═══════════════════════════════════════════════════════════════
# Tab 8: Build Steps (12-step build tour)
# ═══════════════════════════════════════════════════════════════

# Step content: (title, body markdown, image-key or None)
# image-key resolves via build_viz.render_all() for structural diagrams,
# or via a small helper for the teach.py reused slides (step 2, 6, 8).

_BUILD_STEPS: list[tuple[str, str, str | None]] = [
    (
        "1 · Pin the hyperparameters  ·  `config.py`",
        "Start here, not at `model.py`. A single dataclass owning every tunable "
        "keeps the rest of the code uncluttered and reproducibility mechanical.\n\n"
        "**Non-obvious choice:** `vocab_size` starts at **0**. Every script that "
        "builds the model must set `config.vocab_size = tokenizer.vocab_size` "
        "*before* instantiation. This avoids hardcoding a vocab size and keeps "
        "the tokenizer and model coupled through a single variable.\n\n"
        "Derived values live in `@property`, with assertions that catch "
        "misconfiguration at access time rather than during a forward pass.",
        "config",
    ),
    (
        "2 · Byte-level BPE tokenizer  ·  `tokenizer.py`",
        "BPE from scratch in ~200 lines. Byte-level means the base vocab is "
        "always 256 (every possible byte), so any UTF-8 text round-trips "
        "losslessly — no `<UNK>` explosion on new characters.\n\n"
        "**Vocabulary layout:**\n"
        "```\n"
        "0-3     : <PAD>, <BOS>, <EOS>, <UNK>    (4 specials)\n"
        "4-259   : raw bytes 0x00-0xFF           (256 base)\n"
        "260+    : learned BPE merges\n"
        "```\n\n"
        "**Gotcha — test corpus variety:** A corpus of `\"hello world. \" * 100` "
        "collapses into a **single** token after ~10 merges. BPE tests need real "
        "variation. The visual below shows one prompt tokenized (this is slide 01 "
        "of the Train Reports tab, reused here).",
        "teach_01",  # rendered from teach.py on demand
    ),
    (
        "3 · Sliding-window dataset  ·  `dataset.py`",
        "Turns a token stream into `(input_ids, targets)` pairs where "
        "`targets = input_ids shifted +1`. This is the core causal-LM "
        "training invariant.\n\n"
        "**Default stride is `seq_len // 2`** (50% overlap). Non-overlap is a "
        "cleaner evaluation metric but halves the sample count; the overlap is "
        "slightly redundant for training but improves boundary next-token "
        "prediction.\n\n"
        "Tested by an invariant: `target[:-1] == input[1:]` — "
        "`tests/test_dataset.py::test_target_is_input_shifted_by_one`.",
        "sliding",
    ),
    (
        "4 · The Transformer block  ·  `model.py`",
        "Build bottom-up: **RMSNorm → RoPE → Attention → SwiGLU → Block**. "
        "Each component has a test in `tests/test_model.py`.\n\n"
        "**Pre-Norm** (normalize *before* the sublayer, add residual *after*) "
        "is what keeps deep stacks stable without a learning-rate-warmup "
        "knife-edge. Modern LLMs all use this pattern.\n\n"
        "**Weight tying:** `self.lm_head.weight = self.token_emb.weight` — "
        "not a copy, the SAME tensor. Test with `is` not `torch.equal`.\n\n"
        "**GPT-2 residual-projection init:** scale by `1/√(2·n_layers)` to keep "
        "activation variance stable as depth grows. Easy to miss; preserve it.",
        "block",
    ),
    (
        "5 · Training loop  ·  `train.py`",
        "The ingredients, in descending order of impact:\n\n"
        "1. **bf16 mixed precision** on Ampere+ — 2× speedup, basically free\n"
        "2. **AdamW** with weight-decay groups — decay on 2D weights only\n"
        "3. **Linear warmup → cosine decay** to 10% of peak LR (200 warmup steps)\n"
        "4. **Grad clip 1.0** — cheap insurance against loss spikes\n"
        "5. **Val split is sequential 90/10** — random split leaks via overlapping "
        "windows\n"
        "6. **\"Best\" checkpoint on *val* loss**, not train loss. Otherwise the "
        "\"best\" is usually right before overfitting starts.\n\n"
        "At the end, we dump `loss_curve.png` — students can SEE the gap between "
        "train and val open up. That IS overfitting.",
        "training",
    ),
    (
        "6 · Autoregressive generation (naïve)",
        "Forward the full context, sample from last-position logits, append, "
        "repeat. The sampling filter (temperature → top-k → top-p) is shared "
        "between `generate()`, `generate_fast()`, and `teach.py`'s sampling "
        "rollout — one source of truth, `_sample_from_logits()`.\n\n"
        "The `scatter` at the end of top-p looks fragile but is correct: "
        "`sorted_idx` is a permutation over ALL vocab positions, so every slot "
        "is written.\n\n"
        "**This is slide 11 of the Train Reports tab** — watch 10 decode steps with the "
        "top-5 candidates at each step and which one was sampled.",
        "teach_11",  # sampling rollout
    ),
    (
        "7 · KV cache  ·  `generate_fast()`",
        "Without a cache, every decode step recomputes Q, K, V for the entire "
        "context. With a cache, you keep K, V from prior steps and only compute "
        "them for the **new token**. Each step becomes "
        "`O(total_len)` instead of `O(total_len²)`.\n\n"
        "**Two invariants that are easy to miss:**\n\n"
        "- **RoPE offset.** The new token's RoPE angle must match what it would "
        "have been in a single-pass forward — so `start_pos = past_len`, not 0.\n"
        "- **Causal mask.** Applied during **prefill** (T>1), skipped during "
        "**decode** (T=1).\n\n"
        "**Correctness test:** `prefill(prompt)` then 1-step decode must match "
        "a full forward on `prompt`. `test_kv_cache_matches_full_pass` asserts "
        "`max |Δlogit| < 1e-4`. This is the test that catches a broken RoPE "
        "offset.\n\n"
        "**Keep both paths.** Deleting `generate()` loses the pedagogical A/B "
        "comparison. The Benchmark tab literally shows both numbers side by side.",
        "kv_cache",
    ),
    (
        "8 · Teaching hooks  ·  `teach.py`",
        "Pattern worth stealing: capture intermediate tensors via "
        "**forward hooks** without modifying `model.py` at all.\n\n"
        "```python\n"
        "block.attn.register_forward_pre_hook(attn_hook)\n"
        "def attn_hook(module, inp):\n"
        "    x = inp[0]\n"
        "    # Recompute Q, K, V, scores, weights from x\n"
        "    store['attn_weights'] = ...\n"
        "```\n\n"
        "The alternative — sprinkling `return_attention=True` kwargs through "
        "the production code — poisons `model.py`. Hooks keep teaching "
        "instrumentation strictly orthogonal to the model.\n\n"
        "**Re-render on-the-fly** (don't cache PNGs to disk and serve those): "
        "the whole point is that changing the prompt flips the attention. "
        "That's why this UI calls `teach.py` functions directly.\n\n"
        "The visual below is slide 08 — all attention heads of one layer "
        "side-by-side, each learning a different pattern.",
        "teach_08",  # all heads
    ),
    (
        "9 · Attention visualisation  ·  `visualise.py`",
        "Two things worth rendering:\n\n"
        "1. **Per-head heatmap** — `attn_weights[layer][0, head]`, shape `(T, T)`, "
        "viridis colormap, row = query, column = key.\n"
        "2. **Attention rollout** (Abnar & Zuidema 2020) — multiply attention "
        "matrices across all layers to see *effective* attention flow. More "
        "interpretable than any single layer.\n\n"
        "Both are in the **Attention** tab. Pick any prompt + layer + head to "
        "see the heatmap live.",
        None,  # use the Attention tab directly
    ),
    (
        "10 · Gradio web UI  ·  `app.py`  (this page!)",
        "A single browser app as the webinar entrypoint. All handlers import "
        "and call existing functions from `teach.py` and `visualise.py` — no "
        "logic re-implemented.\n\n"
        "**Two Gradio patterns worth learning:**\n\n"
        "1. **Streaming via `yield`.** A generator handler gives you per-token "
        "typewriter output for free.\n"
        "2. **Module-level singletons + `monkeypatch` for tests.** Load the "
        "model once at startup; tests inject a tiny model via "
        "`monkeypatch.setattr(app, \"_MODEL\", tiny_model)`. No I/O, "
        "sub-second UI smoke tests.\n\n"
        "**Version pin:** `gradio>=5,<6`. 4.44.x has a schema-introspection "
        "bug that crashes `launch()`. Upgrade fixes it.",
        "ui_layout",
    ),
    (
        "11 · Test suite  ·  `tests/`",
        "48 tests (42 unit + 6 UI smoke), CPU-only, ~18 s end-to-end. The "
        "highest-value tests are the ones that guard **invariants** rather "
        "than shapes:\n\n"
        "- `test_kv_cache_matches_full_pass` → RoPE `start_pos` invariant\n"
        "- `test_causal_mask_no_future_leakage` → perturbing token T-1 "
        "leaves earlier logits unchanged\n"
        "- `test_weight_tying` → `is`-check: `lm_head.weight` and "
        "`token_emb.weight` are the SAME tensor\n"
        "- `test_training_step_reduces_loss_on_tiny_overfit` → end-to-end "
        "backward/optimizer actually moves parameters\n\n"
        "Invariant tests survive refactoring. Shape tests get rewritten every "
        "time somebody touches the code.",
        "test_matrix",
    ),
    (
        "12 · Ship it  ·  `run.sh`",
        "Commands unified through one bash dispatcher:\n\n"
        "```bash\n"
        "bash run.sh setup       # venv + torch + TinyShakespeare\n"
        "bash run.sh verify      # model.py __main__ sanity check\n"
        "bash run.sh train       # full training (~5-15 min on RTX 4080)\n"
        "bash run.sh generate --fast\n"
        "bash run.sh benchmark   # generate vs generate_fast\n"
        "bash run.sh teach       # 16 static PNGs\n"
        "bash run.sh visualise   # attention heatmaps + rollout\n"
        "bash run.sh ui          # this console\n"
        "bash run.sh test        # pytest suite\n"
        "```\n\n"
        "Explicit subcommands > magic auto-detection. "
        "A new user running `bash run.sh` with no arg gets a helpful usage message.\n\n"
        "**Further reading:** `BUILDING.md` (this page, as markdown), "
        "`REFERENCES.md` (slide ↔ Raschka book), "
        "`nanollm-guide.md` (RNN → Transformer theory).",
        None,
    ),
]


_BUILD_IMG_CACHE: dict[str, str] = {}


def _render_build_image(key: str) -> str | None:
    """Resolve a step's image-key to a PNG path. Idempotent & cached."""
    if key is None:
        return None
    if key in _BUILD_IMG_CACHE:
        return _BUILD_IMG_CACHE[key]

    outdir = tempfile.mkdtemp(prefix="nanollm_build_") \
        if "_BUILD_DIR" not in globals() else globals()["_BUILD_DIR"]
    globals()["_BUILD_DIR"] = outdir

    # Structural diagrams from build_viz
    diagram_paths = build_viz.render_all(outdir)
    if key in diagram_paths:
        _BUILD_IMG_CACHE[key] = diagram_paths[key]
        return diagram_paths[key]

    # Reused slides from teach.py (rendered with a fixed demo prompt)
    if key.startswith("teach_"):
        path = _render_teach_reuse(key, outdir)
        if path is not None:
            _BUILD_IMG_CACHE[key] = path
        return path

    return None


def _render_teach_reuse(key: str, outdir: str) -> str | None:
    """Render one of the teach.py slides using a fixed demo prompt."""
    try:
        model, tokenizer, config = _require_loaded()
    except RuntimeError:
        return None

    demo_prompt = "The cat sat on the"
    token_ids = tokenizer.encode(demo_prompt, add_special=False)
    if len(token_ids) < 2:
        return None
    if len(token_ids) > config.max_seq_len:
        token_ids = token_ids[:config.max_seq_len]
    labels = token_labels(tokenizer, token_ids, max_len=8)
    idx = torch.tensor([token_ids], device=config.device)

    plt_ = _get_plt()

    # Run capture once; reuse for all teach_* slides in this session
    if "_BUILD_CAPTURE" not in globals():
        cap = ForwardCapture(model, target_layer=0)
        with torch.no_grad():
            model(idx)
        cap.remove()
        globals()["_BUILD_CAPTURE"] = cap
        globals()["_BUILD_LABELS"] = labels
        globals()["_BUILD_DEMO_IDS"] = token_ids
    cap = globals()["_BUILD_CAPTURE"]
    labels = globals()["_BUILD_LABELS"]
    token_ids = globals()["_BUILD_DEMO_IDS"]

    path = os.path.join(outdir, f"{key}.png")

    if key == "teach_01":
        slide_01_tokenization(plt_, tokenizer, demo_prompt, token_ids, path)
    elif key == "teach_08":
        slide_08_all_heads(plt_, cap.store, labels, path)
    elif key == "teach_11":
        slide_11_sampling_rollout(plt_, model, tokenizer, idx, config.device,
                                  0.8, 40, 0.9, path)
    else:
        return None

    return path


def render_step_panel(step_idx: int) -> tuple[str, str | None]:
    """Return (markdown, image_path) for a given step index."""
    if not (0 <= step_idx < len(_BUILD_STEPS)):
        return "Unknown step", None
    title, body, img_key = _BUILD_STEPS[step_idx]
    md = f"### {title}\n\n{body}"
    img = _render_build_image(img_key) if img_key else None
    return md, img


# ═══════════════════════════════════════════════════════════════
# UI layout
# ═══════════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    """Assemble the 4-tab Gradio app. Returns the Blocks object (unlaunched)."""
    # Determine UI ranges from the loaded config
    n_layers = _CONFIG.n_layers if _CONFIG is not None else 6
    n_heads = _CONFIG.n_heads if _CONFIG is not None else 6
    max_seq = _CONFIG.max_seq_len if _CONFIG is not None else 256

    with gr.Blocks(title="LLM101 — Webinar Console",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            "# LLM101 — Webinar Console\n"
            "A ~15M-parameter GPT-style transformer, built from scratch. "
            "Tabs follow the build → train → explore workflow: "
            "**Build Steps** → **Train** → **Effects** → **Train Reports** → "
            "**Attention** → **Visualize** → **Generate** → **Benchmark**."
        )
        gr.Markdown(_STATUS)

        # ── Tab 1: Build Steps (orientation — how the project is structured) ──
        with gr.Tab("Build Steps"):
            gr.Markdown(
                "A 12-step tour of how this project was actually built, in the order "
                "a new builder should follow. Complements the standalone `BUILDING.md` "
                "with inline visualisations. Use ◀ / ▶ to walk through, or pick any "
                "step from the list."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    build_radio = gr.Radio(
                        choices=[s[0].split("·")[0].strip() for s in _BUILD_STEPS],
                        value=_BUILD_STEPS[0][0].split("·")[0].strip(),
                        label="Step", interactive=True,
                    )
                    with gr.Row():
                        build_prev = gr.Button("◀ Previous", size="sm")
                        build_next = gr.Button("Next ▶", size="sm", variant="primary")
                with gr.Column(scale=3):
                    build_md = gr.Markdown(_BUILD_STEPS[0][1])  # eager first step
                    build_img = gr.Image(
                        label="Visualisation", type="filepath",
                        show_label=False, height=500,
                    )

            def _idx_from_choice(choice: str) -> int:
                for i, s in enumerate(_BUILD_STEPS):
                    if s[0].split("·")[0].strip() == choice:
                        return i
                return 0

            def _img_or_hidden(img):
                """Return the image or hide the component if None."""
                if img is None:
                    return gr.update(value=None, visible=False)
                return gr.update(value=img, visible=True)

            def on_select(choice):
                md, img = render_step_panel(_idx_from_choice(choice))
                return md, _img_or_hidden(img)

            def on_prev(choice):
                i = max(0, _idx_from_choice(choice) - 1)
                new_choice = _BUILD_STEPS[i][0].split("·")[0].strip()
                md, img = render_step_panel(i)
                return new_choice, md, _img_or_hidden(img)

            def on_next(choice):
                i = min(len(_BUILD_STEPS) - 1, _idx_from_choice(choice) + 1)
                new_choice = _BUILD_STEPS[i][0].split("·")[0].strip()
                md, img = render_step_panel(i)
                return new_choice, md, _img_or_hidden(img)

            build_radio.change(on_select, inputs=build_radio,
                               outputs=[build_md, build_img])
            build_prev.click(on_prev, inputs=build_radio,
                             outputs=[build_radio, build_md, build_img])
            build_next.click(on_next, inputs=build_radio,
                             outputs=[build_radio, build_md, build_img])

            # Eager-render the first step's image on load so the initial view
            # isn't empty
            demo.load(
                lambda: on_select(_BUILD_STEPS[0][0].split("·")[0].strip()),
                inputs=None,
                outputs=[build_md, build_img],
            )

        # ── Tab 2: Train (train the model) ──
        with gr.Tab("Train"):
            gr.Markdown(
                "**Trigger training from the UI and watch progress live.** "
                "Training writes `checkpoints/best.pt` and `checkpoints/loss_curve.png`. "
                "When it finishes, the Generate / Train Reports / Attention tabs pick up the "
                "new model automatically — no restart. "
                "Needs `data/corpus.txt` (run `bash run.sh setup` first)."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    train_epochs = gr.Slider(
                        1, 30, value=10, step=1, label="max_epochs",
                        info="How many full passes through the dataset. Too few = underfit, too many = overfit. Watch the val_loss to find the sweet spot.",
                    )
                    train_batch = gr.Slider(
                        8, 128, value=64, step=8, label="batch_size",
                        info="Sequences processed in parallel per step. Larger = smoother gradients but more memory. Lower (8-16) if you hit OOM.",
                    )
                    train_lr = gr.Slider(
                        1e-5, 1e-3, value=3e-4, step=1e-5, label="learning_rate",
                        info="Peak learning rate after warmup. Controls step size during gradient descent. Too high = unstable, too low = slow convergence. 3e-4 is the AdamW standard.",
                    )
                    train_dropout = gr.Slider(
                        0.0, 0.4, value=0.1, step=0.05, label="dropout",
                        info="Randomly zeroes this fraction of activations during training. Regularization that prevents overfitting — the model can't rely on any single neuron.",
                    )
                    train_warmup = gr.Slider(
                        0, 500, value=0, step=10, label="warmup_steps",
                        info="Steps where LR ramps from 0 to peak before cosine decay. Prevents early gradient explosion. 0 = auto-scale to 10% of total steps.",
                    )
                    train_btn = gr.Button("Start training", variant="primary", size="lg")
                    train_status = gr.Textbox(
                        label="Status", value="idle", interactive=False,
                    )
                    train_samples = gr.Textbox(
                        label="Latest generation sample",
                        lines=4, interactive=False,
                    )
                with gr.Column(scale=2):
                    train_log = gr.Textbox(
                        label="Training log (streaming)",
                        value="(press Start to begin)",
                        lines=22, max_lines=30, show_copy_button=True,
                        interactive=False,
                    )
                    train_plot = gr.Image(
                        label="Loss curve (live)", type="filepath", height=340,
                    )

            train_btn.click(
                train_stream,
                inputs=[train_epochs, train_batch, train_lr,
                        train_dropout, train_warmup],
                outputs=[train_log, train_plot, train_status, train_samples],
            )

        # ── Tab 3: Effects (hyperparameter reference) ──
        with gr.Tab("Effects"):
            gr.Markdown(
                "**How does each training hyperparameter shape the loss curve?** "
                "The plots below are schematic (synthetic data, based on the "
                "standard behavior observed in the ML literature and on small "
                "transformers). Use them as a reference when choosing settings "
                "in the **Train** tab."
            )
            with gr.Row():
                with gr.Column(scale=1, min_width=220):
                    effect_radio = gr.Radio(
                        choices=effect_viz.PARAMS,
                        value=effect_viz.PARAMS[0],
                        label="Parameter",
                        interactive=True,
                        info="Select a hyperparameter to see its schematic effect on training and validation loss curves.",
                    )
                    effect_caption = gr.Markdown()
                with gr.Column(scale=2):
                    effect_img = gr.Image(
                        label="How it shapes training", type="filepath",
                        show_label=False, height=540,
                    )

            def on_effect_change(param):
                caption, img = render_effect_panel(param)
                return caption, img

            effect_radio.change(
                on_effect_change, inputs=effect_radio,
                outputs=[effect_caption, effect_img],
            )
            # Eager-render the first plot so the panel isn't blank on load
            demo.load(
                lambda: render_effect_panel(effect_viz.PARAMS[0]),
                inputs=None,
                outputs=[effect_caption, effect_img],
            )

        # ── Tab 4: Train Reports (step-by-step forward pass) ──
        with gr.Tab("Train Reports"):
            gr.Markdown(
                "### 16-slide visual walkthrough of a single forward pass\n\n"
                "Each slide reveals one stage of how the model processes your prompt — "
                "from raw text through tokenization, embeddings, Q/K/V projections, "
                "attention scores, causal masking, softmax, residual connections, "
                "the FFN layer, and finally the output logits and sampling.\n\n"
                "**How to use:** Enter a prompt and click *Render*. Adjust "
                "`layer` / `head` / `query_pos` to see how different parts of the "
                "model attend to different tokens. Slides 02–09 update when you "
                "change these controls.\n\n"
                "Aligned with Raschka's "
                "*Build a Large Language Model (From Scratch)* — see `REFERENCES.md`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    teach_prompt = gr.Textbox(
                        label="Prompt", value="The cat sat on the", lines=2,
                        info="The text the model will process. Try short phrases to see clear patterns.",
                    )
                    teach_layer = gr.Slider(
                        0, n_layers - 1, 0, step=1, label="layer",
                        info=f"Which transformer block (0–{n_layers-1}). Early layers capture syntax, later layers capture semantics.",
                    )
                    teach_head = gr.Slider(
                        0, n_heads - 1, 0, step=1, label="head",
                        info=f"Which attention head (0–{n_heads-1}). Each head learns to attend to different token relationships.",
                    )
                    teach_qpos = gr.Number(
                        value=-1, precision=0,
                        label="query_pos (-1 = last token)",
                        info="Which token position is 'asking the question'. -1 means the last token (most common for generation).",
                    )
                    with gr.Row():
                        teach_temp = gr.Slider(
                            0.05, 2.0, 0.8, step=0.05, label="temp",
                            info="Sampling temperature. Lower = more deterministic, higher = more creative.",
                        )
                        teach_topk = gr.Slider(
                            0, 200, 40, step=1, label="top_k",
                            info="Only consider the top-k most likely tokens. 0 = disabled.",
                        )
                        teach_topp = gr.Slider(
                            0.05, 1.0, 0.9, step=0.05, label="top_p",
                            info="Nucleus sampling: keep tokens until cumulative probability reaches top_p.",
                        )
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

        # ── Tab 5: Attention (per-head heatmaps) ──
        with gr.Tab("Attention"):
            gr.Markdown(
                "### Attention heatmaps — what tokens attend to what\n\n"
                "Self-attention is the core mechanism that lets each token 'look at' "
                "every earlier token to decide what information to carry forward. "
                "This tab visualizes those attention weights directly.\n\n"
                "**Left (Single head):** One attention head's weight matrix after "
                "softmax + causal masking. Rows = query token (the one 'looking'), "
                "columns = key token (the one being 'looked at'). "
                "Bright = strong attention. The lower triangle is zero because "
                "causal masking prevents tokens from attending to future positions.\n\n"
                "**Right (Rollout):** Attention rollout across all layers "
                "(Abnar & Zuidema 2020) — approximates how information flows from "
                "input tokens to the final output across the *entire* model depth, "
                "not just one layer.\n\n"
                "**What the sliders do:**\n"
                "- **layer:** The model has {n_layers} stacked transformer blocks. "
                "Layer 0 (first) typically learns surface patterns — positional attention, "
                "punctuation, local word relationships. Higher layers learn increasingly "
                "abstract features — syntax, semantic roles, long-range dependencies. "
                "Changing the layer shows you a completely different stage of processing.\n"
                "- **head:** Each layer has {n_heads} parallel attention heads. "
                "Each head independently learns to focus on different types of relationships "
                "— one head might track subject-verb agreement, another might attend to "
                "the previous word, another to sentence boundaries. Changing the head shows "
                "you a different 'lens' on the same layer.\n\n"
                "**Try:** Compare layer 0 head 0 vs layer {n_layers_m1} head 0 to see "
                "how attention evolves from shallow to deep."
                .format(n_layers=n_layers, n_heads=n_heads, n_layers_m1=n_layers - 1)
            )
            with gr.Row():
                with gr.Column(scale=1):
                    attn_prompt = gr.Textbox(
                        label="Prompt", value="To be or not to be", lines=2,
                        info="Shorter prompts produce clearer heatmaps (5-10 tokens ideal).",
                    )
                    attn_layer = gr.Slider(
                        0, n_layers - 1, n_layers // 2, step=1, label="layer",
                        info=f"Layer 0 = surface patterns, layer {n_layers-1} = deep semantics. Changes the left heatmap.",
                    )
                    attn_head = gr.Slider(
                        0, n_heads - 1, 0, step=1, label="head",
                        info=f"Each of the {n_heads} heads learns different attention patterns. Changes the left heatmap.",
                    )
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

        # ── Tab 6: Visualize (animated forward pass) ──
        with gr.Tab("Visualize"):
            gr.Markdown(
                "### Animated forward pass — watch data flow through the transformer\n\n"
                "Enter a prompt and click **Visualize** to see a synchronized animation of:\n"
                "- **Left:** Architecture diagram — each layer lights up as data flows through\n"
                "- **Center:** 6×6 attention grid — every head's attention pattern, revealed layer by layer\n"
                "- **Right:** Activation flow — hidden state magnitude at each layer\n\n"
                "Click any cell in the attention grid to see the full-size heatmap and tensor shapes. "
                "Use the playback controls (Play/Pause/Step) to control the animation speed."
            )
            viz_prompt = gr.Textbox(
                label="Prompt", value="To be or not to be",
                lines=1, max_lines=2,
                info="Text to process. Shorter prompts (5-10 tokens) produce clearer visualizations.",
            )
            viz_btn = gr.Button("Visualize forward pass", variant="primary")
            viz_html = gr.HTML(label="Visualization")

            viz_btn.click(
                render_visualization,
                inputs=[viz_prompt],
                outputs=[viz_html],
            )

        # ── Tab 7: Generate (interactive text generation) ──
        with gr.Tab("Generate"):
            gr.Markdown(
                "### Interactive text generation\n\n"
                "The model predicts one token at a time, appending each prediction to "
                "the prompt and feeding it back in. The three sampling parameters "
                "(`temperature`, `top_k`, `top_p`) control *how* the next token is "
                "chosen from the model's probability distribution — they don't change "
                "the model itself, just how creative vs. deterministic the output is.\n\n"
                "**Tip:** Toggle the KV cache off to see the slower no-cache path "
                "(the reference implementation used in most tutorials). "
                "With cache on, each new token only computes attention against the "
                "cached K/V from previous tokens — O(T) instead of O(T²)."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    gen_prompt = gr.Textbox(
                        label="Prompt", value="To be or not to be, ",
                        lines=2, max_lines=4,
                        info="Starting text the model continues from. The model has only seen Shakespeare during training.",
                    )
                    with gr.Row():
                        gen_max = gr.Slider(
                            8, max_seq - 1, 100, step=1,
                            label="max_new_tokens",
                            info="How many tokens to generate. Longer = slower but more text. Limited by the 256-token context window.",
                        )
                    with gr.Row():
                        gen_temp = gr.Slider(
                            0.05, 2.0, 0.8, step=0.05, label="temperature",
                            info="Scales the logits before softmax. <1 = sharper (more repetitive), >1 = flatter (more random). 0.8 is a good default.",
                        )
                        gen_topk = gr.Slider(
                            0, 200, 40, step=1, label="top_k",
                            info="Keep only the top-k most probable tokens, zero out the rest. 0 = disabled (consider all tokens). Prevents rare gibberish.",
                        )
                        gen_topp = gr.Slider(
                            0.05, 1.0, 0.9, step=0.05, label="top_p (nucleus)",
                            info="Keep the smallest set of tokens whose cumulative probability exceeds top_p. Adapts dynamically — broad when uncertain, narrow when confident.",
                        )
                    gen_cache = gr.Checkbox(
                        True, label="Use KV cache (generate_fast)",
                        info="ON: reuses past K/V vectors (fast, O(T) per step). OFF: recomputes full attention each step (slow, O(T²)) — educational comparison.",
                    )
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

        # ── Tab 8: Benchmark (cache vs no-cache comparison) ──
        with gr.Tab("Benchmark"):
            gr.Markdown(
                "### KV cache speedup — why caching matters\n\n"
                "Compares two generation strategies side by side:\n"
                "- **`generate()`** — no cache. Every new token recomputes attention "
                "over *all* previous tokens from scratch. Cost: O(T²) per step.\n"
                "- **`generate_fast()`** — KV cache. Stores each layer's K and V "
                "tensors from previous steps and only computes the new token's "
                "attention. Cost: O(T) per step.\n\n"
                "The speedup grows with sequence length — at 200 tokens the cached "
                "version can be 5-10x faster. This is how production LLMs (GPT, LLaMA, "
                "Claude) achieve fast inference."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    bench_prompt = gr.Textbox(
                        label="Prompt", value="The ", lines=1,
                        info="Short prompt to seed generation. The benchmark measures the generation speed, not prompt processing.",
                    )
                    bench_n = gr.Slider(
                        20, max_seq - 10, 100, step=10,
                        label="tokens to generate",
                        info="More tokens = bigger speedup gap between cached and uncached. Try 200 for dramatic results.",
                    )
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

def find_free_port(start_port: int, host: str = "127.0.0.1",
                   attempts: int = 20) -> int:
    """Return the first free port at or above `start_port`, probing by bind.

    Gradio's own retry only tries the exact port you pass. This helper walks
    the next N ports so `bash run.sh ui` Just Works even when a prior instance
    is still holding 7860.
    """
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port in {start_port}..{start_port + attempts - 1} on {host}"
    )


def main():
    parser = argparse.ArgumentParser(description="LLM101 — Gradio webinar console")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--port", type=int, default=7860,
                        help="Preferred starting port (will auto-fallback if busy)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public URL via Gradio's share tunnel")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (use 0.0.0.0 to expose on LAN)")
    args = parser.parse_args()

    print("LLM101 Webinar Console — starting...")
    _load_model(args.checkpoint)
    print(f"  {_STATUS}")

    # Auto-fallback to the next free port so a running instance on 7860 doesn't
    # crash a fresh `bash run.sh ui` call.
    port = find_free_port(args.port, host=args.host)
    if port != args.port:
        print(f"  Port {args.port} is busy — using {port} instead")

    demo = build_ui()
    # show_api=False works around a gradio-client schema-introspection bug
    # (TypeError on schema traversal in gradio 4.44.x). The UI still works,
    # we just skip the auto-generated API docs page.
    demo.queue().launch(
        server_name=args.host,
        server_port=port,
        share=args.share,
        show_error=True,
        inbrowser=True,
        show_api=False,
    )


if __name__ == "__main__":
    main()
