# References — What LLM101 Covers From the Book

This project follows the pedagogy of **Sebastian Raschka's
["Build a Large Language Model (From Scratch)"](https://github.com/rasbt/LLMs-from-scratch)**
and adapts it with a few modern substitutions (RMSNorm / RoPE / SwiGLU instead of
LayerNorm / sinusoidal / GELU) so students see both the classical design and what
today's LLaMA-family models actually use.

Each teach.py slide maps to a chapter of the book:

| Slide | File | Concept | Raschka chapter |
|------:|------|---------|----------------|
| 01 | `01_tokenization.png`       | Text → bytes → BPE merges → token IDs | **Ch 2.5** — BPE tokenization |
| 02 | `02_embeddings.png`         | Token IDs → embedding vectors          | **Ch 2.6** — Token embeddings |
| 12 | `12_positional_rope.png`    | Position encoding (RoPE variant)       | **Ch 2.7** — Positional embeddings |
| 03 | `03_qkv.png`                | Q, K, V projections                    | **Ch 3.4** — Self-attention with trainable weights |
| 04 | `04_scores_raw.png`         | Raw Q·Kᵀ similarity matrix             | **Ch 3.3 → 3.4** — Simplified → trainable attention |
| 13 | `13_scaling_rationale.png`  | Why `/√d_k`                            | **Ch 3.4.2** — Scaled dot-product |
| 05 | `05_scores_masked.png`      | Causal mask → upper-triangle blocked   | **Ch 3.5** — Hiding future words |
| 06 | `06_attn_weights.png`       | Softmax → row-probability distribution | **Ch 3.4** — Attention weights |
| 07 | `07_value_sum.png`          | Weighted sum of V vectors              | **Ch 3.4** — Context vector |
| 08 | `08_all_heads.png`          | Multi-head attention side-by-side      | **Ch 3.6** — Multi-head attention |
| 09 | `09_ffn_delta.png`          | Position-wise FFN (SwiGLU variant)     | **Ch 4.3** — GELU + FeedForward block |
| 16 | `16_param_breakdown.png`    | Parameter count by component           | **Ch 4.6** — Parameter counting |
| 10 | `10_logits_topk.png`        | Top-k next-token distribution          | **Ch 5.3.1** — Top-k sampling |
| 14 | `14_temperature_effect.png` | Temperature sharpens / flattens        | **Ch 5.3.1** — Temperature scaling |
| 15 | `15_greedy_vs_sample.png`   | Greedy vs sampling output text         | **Ch 5.3** — Decoding strategies |
| 11 | `11_sampling_rollout.png`   | Autoregressive generation step-by-step | **Ch 5.3** — generate_text_simple |

## Modern substitutions (and why)

| Book uses (GPT-2 era) | LLM101 uses (LLaMA era) | Rationale |
|-----------------------|--------------------------|-----------|
| LayerNorm             | **RMSNorm**              | Fewer ops, same stability — adopted by LLaMA / Mistral / Qwen |
| Learned positional embeddings | **RoPE** (Rotary) | Better length generalization; relative position for free |
| GELU + standard FFN   | **SwiGLU FFN**           | Gated activation; used in LLaMA-2 / Mistral |
| Post-Norm             | **Pre-Norm**             | More stable training, no LR warmup knife-edge |

Slides 09 and 12 demonstrate the substitution visually. Chapter 4 of the book
(LayerNorm, GELU) still applies conceptually — only the inner formula differs.

## Topics from the book intentionally NOT covered

The book goes on to fine-tuning (chapters 6–7). LLM101 stops at pre-training
because the webinar target is understanding the transformer, not production
fine-tuning. The pre-training loop (Ch 5.2) is covered by `train.py` with
additions not in the book: bf16 mixed precision, val split with best-on-val
checkpointing, and the loss-curve PNG.

## Beyond the book

Two elements the book doesn't visualize, but LLM101 does:

- **KV cache** — `generate_fast()` demonstrates the inference optimization that
  makes modern LLMs deploy-able. The `benchmark` command shows the speedup.
- **Attention rollout** — `visualise.py` computes effective attention across
  layers (Abnar & Zuidema 2020), a complementary interpretability tool.
