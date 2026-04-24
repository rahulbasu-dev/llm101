# LLM101 — Build a Language Model From Scratch

A ~15M-parameter GPT-style decoder-only Transformer in ~800 lines of PyTorch.
Every component is written explicitly — RMSNorm, RoPE, SwiGLU, causal self-attention,
weight tying — so you can see exactly how modern LLMs work inside.

Aligned with Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
— see [REFERENCES.md](REFERENCES.md) for the chapter cross-walk.

## Quick Start

```bash
bash run.sh setup      # venv + PyTorch + TinyShakespeare corpus
bash run.sh verify     # shape test + KV-cache equivalence check
bash run.sh train      # train (~5 min GPU, ~30 min CPU)
bash run.sh generate --fast   # interactive generation with KV cache
bash run.sh ui         # Gradio web console → http://127.0.0.1:7860
```

`run.sh` auto-detects your hardware (NVIDIA GPU / AMD ROCm / CPU + Intel NPU)
and installs the matching PyTorch build. No manual configuration needed.

### Jupyter Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulbasu-dev/llm101/blob/main/LLM101_From_Scratch.ipynb)

**`LLM101_From_Scratch.ipynb`** covers the complete pipeline in 10 sections:
config → tokenizer → dataset → model architecture → forward pass visualization →
training → generation → attention heatmaps → KV cache deep dive.
Works on both GPU and CPU (auto-detected).

## Architecture

### Model internals — one forward pass

```
idx : (B, T)                                                     [input token IDs]
  │
  ▼
token_emb (vocab→d_model)  ──── weight-tied with lm_head ────┐
  │  emb_dropout                                              │
  ▼                                                           │
x : (B, T, d_model=384)                                       │
  │                                                           │
  ├──────────────────── × n_layers=6 ────────────────────┐    │
  │ ┌────────────────── TransformerBlock ──────────────┐ │    │
  │ │  RMSNorm → CausalSelfAttention → + residual     │ │    │
  │ │  │  qkv_proj → Q,K,V : (B, 6, T, 64)           │ │    │
  │ │  │  RoPE(Q), RoPE(K)                            │ │    │
  │ │  │  scores = Q·Kᵀ / √d_head → mask → softmax   │ │    │
  │ │  │  out = attn_weights · V → out_proj            │ │    │
  │ │  RMSNorm → FeedForward (SwiGLU) → + residual    │ │    │
  │ │  │  silu(gate_proj(x)) ⊙ up_proj(x) → down_proj │ │    │
  │ └──────────────────────────────────────────────────┘ │    │
  └───────────────────────────────────────────────────────┘    │
  │                                                             │
  ▼                                                             │
norm_f (RMSNorm) → lm_head ◄── shared weight ──────────────────┘
  │
  ▼
logits : (B, T, vocab_size)
```

### KV-cache decoding

`generate_fast()` caches K,V per layer so each decode step is O(T) instead of O(T²):

1. **Prefill**: full forward on prompt → collect caches, apply causal mask, RoPE at `start_pos=0`
2. **Decode loop**: single-token forward, concat new K,V to cache, RoPE at `start_pos=past_len`, no causal mask needed

Equivalence verified by `test_kv_cache_matches_full_pass` (max |Δlogit| < 1e-4).

### Design choices

| Component | Choice | Classical alternative |
|---|---|---|
| Normalization | **RMSNorm** | LayerNorm |
| Positional encoding | **RoPE** | Sinusoidal / learned |
| FFN activation | **SwiGLU** | ReLU / GELU |
| Norm placement | **Pre-Norm** | Post-Norm |
| Output projection | **Weight-tied** with embedding | Separate weights |
| Attention projection | **Combined QKV** | Separate Q, K, V |

## Web UI

`bash run.sh ui` launches a Gradio console with 6 tabs:

| Tab | Purpose |
|-----|---------|
| **Train** | Interactive training with hyperparameter sliders, live loss curve |
| **Train Reports** | 16-step forward pass walkthrough with pin-and-compare + PPTX export |
| **Attention** | Per-head heatmap + attention rollout across all layers |
| **Visualize** | Animated three-panel view: architecture + all-heads grid + activation norms |
| **Generate** | Token-by-token streaming with temperature/top_k/top_p controls |
| **Benchmark** | Side-by-side generate() vs generate_fast() speed comparison |

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters in one dataclass + hardware detection |
| `tokenizer.py` | Byte-level BPE tokenizer (from scratch, ~200 lines) |
| `model.py` | Full Transformer: RMSNorm, RoPE, Attention, SwiGLU, NanoLLM |
| `dataset.py` | Sliding-window dataset for causal LM |
| `train.py` | Training loop: torch.compile, mixed precision, warmup + cosine LR |
| `generate.py` | Interactive generation with top-k + nucleus sampling |
| `teach.py` | Hook-based 16-slide forward-pass walkthrough |
| `visualise.py` | Attention heatmaps and rollout analysis |
| `visualize_anim.py` | Tensor collection for the animated visualization tab |
| `app.py` | Gradio web console (6 tabs) |
| `tests/` | 55-test pytest suite (CPU-only, ~15s) |
| `run.sh` | One-command dispatcher for all workflows |
| `LLM101_From_Scratch.ipynb` | Comprehensive Jupyter notebook (48 cells, 10 sections) |

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 384 | Hidden dimension |
| n_layers | 6 | Transformer blocks |
| n_heads | 6 | Attention heads (d_head=64) |
| d_ff | 1536 | FFN intermediate (4× d_model) |
| max_seq_len | 256 | Context window |
| vocab_size | ~4096 | BPE on TinyShakespeare |
| batch_size | 64 | Adjustable via CLI/UI |
| learning_rate | 3e-4 | AdamW with cosine decay |

## Commands

```bash
bash run.sh setup       # venv + PyTorch (auto-detects GPU) + corpus
bash run.sh verify      # GPU check + model shapes + tokenizer + training time estimate
bash run.sh train       # full training (--max-epochs 3 for quick test)
bash run.sh generate    # interactive generation (add --fast for KV cache)
bash run.sh benchmark   # generate() vs generate_fast() timing
bash run.sh teach       # 16 teaching slides → teaching_plots/
bash run.sh visualise   # attention heatmaps → attention_plots/
bash run.sh test        # pytest suite (CPU, no GPU needed)
bash run.sh ui          # Gradio web console
```

## License

Educational use. Built for learning, not production.
