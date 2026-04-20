# NanoLLM — Webinar-Ready Enhancements

**Date:** 2026-04-20
**Scope:** Smallest, most-visible enhancements to make NanoLLM webinar-ready.
**Status:** Approved. Implementation in progress.

## Goals

1. Fast, honest, pedagogical. Nothing fancy.
2. Every addition should produce a visible demo moment in a webinar.
3. No existing file is rewritten; additions are additive where possible.

## Deliverables

### 1. KV cache (model.py)

Keep the existing `generate()` untouched as a "no cache" reference path.
Add a parallel `generate_fast()` that reuses cached K,V across decode steps.

**Changes:**
- `RotaryPositionEmbedding.forward(x, seq_len, start_pos=0)` — slice RoPE at the absolute position so cached decode uses the correct angle for each new token.
- `CausalSelfAttention.forward(x, past_kv=None) → (out, new_kv)` — if `past_kv` is supplied, concatenate cached K,V with the freshly computed ones; causal mask applies only on prefill (T>1).
- `TransformerBlock.forward(x, past_kv=None) → (out, new_kv)` — threads through.
- `NanoLLM.forward(idx, targets=None, past_kv=None)` — returns `(logits, loss)` during training (unchanged) and `(logits, list[new_kv])` during inference. Existing generate() discards the second item.
- `NanoLLM.generate_fast(idx, max_new_tokens, ...)` — NEW. Prefill on full prompt, then decode one token at a time with cache.

**Correctness check:** `generate_fast(prompt)` and `generate(prompt)` must produce identical logits given the same RNG seed. A simple smoke test in `__main__` validates this.

**Rationale:** For a 256-token context, generate_fast() is ~10-50× faster than generate() — a visible webinar "wow" moment.

### 2. Validation split (train.py)

Sequential 90/10 split of the tokenized corpus (first 90% train, last 10% val).
Random split would leak via overlapping sliding windows.

**Changes:**
- Split `tokens` before building `TextDataset`.
- Build two datasets, two dataloaders. Val loader has `shuffle=False`.
- After each epoch, compute val loss in `model.eval()` + `torch.no_grad()`.
- "Best" checkpoint selection moves from train loss → val loss.
- Epoch summary reports both train PPL and val PPL.

**Rationale:** Demonstrating val loss keeps training honest — the audience can see when overfitting starts.

### 3. Loss curve PNG (train.py)

Accumulate per-step train losses and per-epoch (train_avg, val_avg) tuples.
At training end, save `checkpoints/loss_curve.png`:
- X: global step (per-step trace) and epoch boundaries (for val).
- Y: log-scale loss.
- Blue line: train per-step loss (light, underlaid).
- Blue dots: train per-epoch average.
- Red dots: val per-epoch loss.

**Rationale:** The classic "loss went from 8.3 → 3.2" slide.

### 4. Benchmark demo (generate.py, run.sh)

- `generate.py --benchmark` — generates 100 tokens with `generate()` then `generate_fast()`, prints tokens/sec for each and the speedup factor.
- `generate.py --fast` — use the cached path for interactive generation.
- `run.sh benchmark` — one-command webinar button.

**Rationale:** One slide: "Without KV cache: N tok/s · With KV cache: M tok/s · Speedup: Kx".

### 5. Step-by-step forward-pass walkthrough (teach.py — NEW)

New file `teach.py` generates 11 annotated PNG slides in `teaching_plots/` showing one short prompt traversing the model.

Slides:

1. **Tokenization** — text → bytes → BPE merges → token IDs
2. **Embeddings** — (T, d_model) heatmap, first 32 dims
3. **Q, K, V projections** — three side-by-side heatmaps, layer 0 head 0
4. **Attention scores (raw)** — Q·Kᵀ/√d before masking
5. **Causal mask applied** — same matrix, upper triangle grey
6. **Softmax → attention weights** — rows sum to 1, annotated
7. **Weighted value sum** — bar chart for one output token position
8. **All heads side-by-side** — 6 small heatmaps, layer 0
9. **Residual + FFN** — before/after FFN heatmap (32 dims)
10. **Logits → top-k bars** — next-token distribution with top 20 tokens
11. **Sampling rollout** — 10 decode steps, top-5 candidates each, chosen highlighted

**Implementation approach:** Forward hooks (same pattern as `visualise.py`) capture intermediate tensors — no changes to `model.py` forward logic. Matplotlib produces each slide with a consistent header strip (slide #, title, caption).

**Run:** `bash run.sh teach` or `bash run.sh teach "custom prompt"`.

### 6. run.sh additions

- `benchmark` — invokes `python generate.py --benchmark`
- `teach` — invokes `python teach.py` with optional prompt arg

## Out of scope (parked, not TODO)

- Tests (pytest), resume-from-checkpoint, faster BPE, config CLI overrides
- Dead code cleanup in `model.py:283-285` (two cancelling lines)
- Any change to tokenizer.py, dataset.py, visualise.py, config.py

## Risks

- **RoPE offset bug.** Easiest way to get the cache wrong — the new token's RoPE angle must match what it would have been in a single-pass forward. Mitigated by the equivalence smoke test in `model.py`'s `__main__`.
- **Val sliding windows.** If the val tokens end up smaller than `max_seq_len + 1`, there are zero samples — guard with a clear error message.
