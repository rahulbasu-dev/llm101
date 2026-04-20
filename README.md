# NanoLLM — Build a Language Model From Scratch

A complete, annotated GPT-style Transformer language model in ~800 lines of PyTorch.
Every component is written from scratch with detailed comments explaining **what** and **why**.

**Built to teach.** Ships with a 16-slide step-by-step visualization of one prompt
flowing through the network, a KV-cache speedup demo, and a train/val loss curve.
Aligned with Sebastian Raschka's
[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
— see [REFERENCES.md](REFERENCES.md) for the slide ↔ chapter cross-walk.

## Architecture

```
Token IDs ──→ Embedding ──→ [TransformerBlock × 6] ──→ RMSNorm ──→ LM Head ──→ Logits
                                    │
                         ┌──────────┴──────────┐
                         │   TransformerBlock   │
                         │                      │
                         │  RMSNorm             │
                         │  ↓                   │
                         │  Multi-Head Attention │ ← RoPE, Causal Mask
                         │  ↓ + residual        │
                         │  RMSNorm             │
                         │  ↓                   │
                         │  SwiGLU FFN          │
                         │  ↓ + residual        │
                         └──────────────────────┘
```

**Design choices match modern LLMs (LLaMA/Mistral/Qwen):**
- RMSNorm (not LayerNorm)
- RoPE (not sinusoidal or learned positions)
- SwiGLU activation (not ReLU/GELU)
- Pre-Norm (not Post-Norm)
- Weight tying (embedding ↔ lm_head)
- Combined QKV projection

**~15M parameters** — trains in 5–15 minutes on RTX 4080.

## Quick Start

```bash
# 1. Setup (installs PyTorch, downloads TinyShakespeare)
bash run.sh setup

# 2. Verify shapes + KV-cache equivalence
bash run.sh verify

# 3. Train (saves loss_curve.png, picks best checkpoint on VAL loss)
bash run.sh train

# 4. Generate text interactively (use --fast for KV-cache path)
bash run.sh generate --fast

# 5. Show the 10-100x KV-cache speedup
bash run.sh benchmark

# 6. 16-slide step-by-step forward-pass walkthrough
bash run.sh teach

# 7. Attention heatmaps + rollout
bash run.sh visualise
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All hyperparameters in one dataclass |
| `tokenizer.py` | Byte-level BPE tokenizer (from scratch) |
| `model.py` | Full Transformer: RMSNorm, RoPE, Attention, SwiGLU, NanoLLM |
| `dataset.py` | Sliding-window dataset for causal LM |
| `train.py` | Training loop: mixed-precision, warmup, cosine LR, checkpointing |
| `generate.py` | Interactive generation with top-k + nucleus sampling |
| `visualise.py` | Attention heatmaps and rollout analysis |
| `teach.py` | **16-slide forward-pass walkthrough** (tokenization → sampling) |
| `run.sh` | One-command setup/train/generate/visualise/teach/benchmark |
| `REFERENCES.md` | Slide-to-Raschka-chapter cross-walk |
| `docs/superpowers/specs/` | Design docs for enhancements |

## Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| d_model | 384 | Enough capacity for language patterns |
| n_layers | 6 | Deep enough for interesting attention patterns |
| n_heads | 6 | 64-dim per head (standard) |
| d_ff | 1536 | 4× d_model (standard ratio) |
| max_seq_len | 256 | Short but sufficient for demo |
| vocab_size | ~4096 | BPE on small corpus |
| batch_size | 64 | Fills RTX 4080 VRAM well |
| learning_rate | 3e-4 | Standard for small models |

## What to Show in a Webinar

1. **model.py** — walk through each class, explain the architecture evolution
2. **`run.sh teach`** — step through 16 annotated slides (tokenization → embedding
   → Q/K/V → causal mask → softmax → value-weighted sum → multi-head → FFN →
   RoPE → temperature → greedy-vs-sample → parameter breakdown)
3. **Training loss curve** (`checkpoints/loss_curve.png`) — train vs val,
   watch loss drop from ~8.3 (random) to ~3–4 without overfitting
4. **`run.sh benchmark`** — before/after KV-cache speedup, the "inference
   optimization" money-shot
5. **Generation samples** (`run.sh generate --fast`) — interactive demo with
   KV cache live
6. **Attention heatmaps** (`run.sh visualise`) — per-head patterns + rollout
7. **Scaling table** — compare NanoLLM to LLaMA-2 7B to GPT-4

## New features (beyond Raschka)

- **KV cache + `generate_fast()`** — decode step is O(total_len) instead of
  O(total_len²). Equivalence to the reference `generate()` is validated by
  a smoke test in `model.py:__main__` (checks `max |Δlogit| < 1e-4`).
- **Val split** — sequential 90/10, best checkpoint picked on *val* loss.
- **Loss curve PNG** — `checkpoints/loss_curve.png` with train per-step,
  train per-epoch, val per-epoch, log-scale y.
- **16 teaching slides** — `teach.py` hooks intermediate tensors to render
  one prompt's full journey through the model.

## License

Educational use. Built for learning, not production.
