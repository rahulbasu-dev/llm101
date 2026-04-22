"""LLM101 — Webinar Console (Gradio UI).

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
import build_viz


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
# Tab 5: Build Steps (interactive tutorial)
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
        "of the Teach tab, reused here).",
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
        "**This is slide 11 of the Teach tab** — watch 10 decode steps with the "
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

        # ── Tab 5: Build Steps ──
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

            def on_select(choice):
                md, img = render_step_panel(_idx_from_choice(choice))
                return md, img

            def on_prev(choice):
                i = max(0, _idx_from_choice(choice) - 1)
                new_choice = _BUILD_STEPS[i][0].split("·")[0].strip()
                md, img = render_step_panel(i)
                return new_choice, md, img

            def on_next(choice):
                i = min(len(_BUILD_STEPS) - 1, _idx_from_choice(choice) + 1)
                new_choice = _BUILD_STEPS[i][0].split("·")[0].strip()
                md, img = render_step_panel(i)
                return new_choice, md, img

            build_radio.change(on_select, inputs=build_radio,
                               outputs=[build_md, build_img])
            build_prev.click(on_prev, inputs=build_radio,
                             outputs=[build_radio, build_md, build_img])
            build_next.click(on_next, inputs=build_radio,
                             outputs=[build_radio, build_md, build_img])

            # Eager-render the first step's image on load so the initial view
            # isn't empty
            demo.load(
                lambda: render_step_panel(0),
                inputs=None,
                outputs=[build_md, build_img],
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
