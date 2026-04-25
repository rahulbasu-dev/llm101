# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**LLM101** — a ~15M-parameter, from-scratch GPT-style decoder-only Transformer in
~800 lines of PyTorch. Every component is written explicitly (RMSNorm, RoPE,
SwiGLU FFN, causal self-attention with a combined QKV projection, weight tying)
to be read and taught. The project is aligned with Sebastian Raschka's
*Build a Large Language Model (From Scratch)* — see `REFERENCES.md` for the
slide ↔ chapter cross-walk.

**This is an educational codebase.** Pedagogy trumps brevity; see "What NOT to
simplify" below.

> Note: `C:\GitHub\CLAUDE.md` (the repo-parent CLAUDE.md) describes a different
> project ("NLP Knowledge Base" / Flask). It does not apply here. This file is
> authoritative for `me/llm101/`.

## Accelerator detection

`run.sh` auto-detects the available hardware at startup:

| Detected | PyTorch build | Pip index |
|---|---|---|
| NVIDIA GPU (`nvidia-smi`) | cu121 (~2 GB) | `whl/cu121` |
| AMD GPU (`rocminfo`) | ROCm (~2 GB) | `whl/rocm6.2` |
| CPU only / Intel NPU | CPU-only (~200 MB) | `whl/cpu` |

When no CUDA GPU is found, `run.sh` automatically exports `NANOLLM_ALLOW_CPU=1`
so all commands work without manual flags. Intel NPUs (Core Ultra / Meteor Lake)
are detected and reported in the banner but not yet usable by PyTorch — the CPU
fallback is used.

The application entry points (`app.py`, `train.py`, `generate.py`, `teach.py`,
`visualise.py`) call `config.require_cuda()` at startup. If CUDA is not available
**and** `NANOLLM_ALLOW_CPU` is not set, they print a helpful message and
`sys.exit(2)`. **Tests are exempted**: the autouse `_seed_everything` fixture in
`tests/conftest.py` sets `NANOLLM_ALLOW_CPU=1` so the suite stays CPU-portable.

When running Python directly (not through `run.sh`) on a non-CUDA machine:
```bash
NANOLLM_ALLOW_CPU=1 python train.py
```

When a new entry point is added, call `require_cuda()` at the top of its `main()`.
Don't replicate the `"cuda" if torch.cuda.is_available() else "cpu"` ternary —
that's the pre-gate pattern we deliberately moved away from.

## Commands

All standard workflows go through `run.sh`. User runs via WSL2 (Linux bash on
Windows), so assume an RTX 4080 GPU and bash/Unix paths in any shell command:

```bash
bash run.sh setup      # venv + torch(cu121) + TinyShakespeare → data/corpus.txt
bash run.sh verify     # runs model.py + tokenizer.py smoke tests
bash run.sh train      # full training; saves checkpoints/ + loss_curve.png
bash run.sh generate            # interactive, uses reference generate() (no cache)
bash run.sh generate --fast     # interactive, uses generate_fast() (KV cache)
bash run.sh benchmark           # side-by-side tok/s: generate vs generate_fast
bash run.sh teach               # 16 slides → teaching_plots/
bash run.sh teach "custom prompt"
bash run.sh visualise           # attention heatmaps + rollout → attention_plots/
bash run.sh ui                  # Gradio webinar console at http://127.0.0.1:7860
bash run.sh ui --share          # public Gradio URL (webinar mode)
```

Direct Python invocations (for targeted work, preferred over shelling through run.sh):

```bash
python model.py                 # shape test + KV-cache equivalence check
python tokenizer.py             # BPE round-trip smoke test
python train.py                 # full training (data/corpus.txt must exist)
python generate.py --benchmark
python teach.py --layer 3 --head 2 --query_pos 5
```

**Test suite:** 42 tests under `tests/` covering config, tokenizer, dataset,
model (including KV-cache multi-step equivalence and causal-mask leakage), and
end-to-end integration. All run CPU-only on mock synthetic data in ~4s.

```bash
bash run.sh test                                # full suite
python -m pytest tests/test_model.py -v         # one module
python -m pytest tests/ -k "kv_cache"           # one pattern
python -m pytest tests/ -x                      # stop on first failure
```

`model.py:__main__` also has a quick smoke block (shape + KV-cache equivalence
+ generate_fast) that runs via `python model.py` or `bash run.sh verify`.

When editing attention, RoPE, or the forward signature, run
`tests/test_model.py` first — that's where equivalence is guarded. If you add
new behavior, add the test there too.

`pylint` / `black` / other linters mentioned in the parent `C:\GitHub\CLAUDE.md`
are **not configured here** — don't invoke them.

On Windows, the codebase uses some box-drawing and arrow characters in print
statements that cp1252 can't encode. Prefix Python runs with
`PYTHONIOENCODING=utf-8` when running directly from a Windows shell.

## Architecture — what requires reading multiple files

### Data flow end-to-end

```
data/corpus.txt
   └─ tokenizer.BPETokenizer.train()  → tokenizer.json (vocab + merges)
      └─ tokenizer.encode(corpus)     → List[int]  (one pass, no special tokens)
         └─ sequential 90/10 split    → train_tokens, val_tokens   (train.py)
            └─ dataset.TextDataset    → sliding windows (stride = seq_len/2 by default)
               └─ DataLoader          → (input_ids, targets)  where targets = input_ids shifted +1
                  └─ NanoLLM(input_ids, targets)  → (logits, cross_entropy_loss)
```

Every module imports from `config.NanoLLMConfig`. `config.vocab_size` starts at
**0** and is assigned at runtime *after* the tokenizer trains or loads. Any code
that constructs `NanoLLM` directly must set `config.vocab_size` first (see
`generate.py:load_model` and `teach.py:load_model` for the pattern). Do not
hard-code a vocab size in config.py.

### Model pipeline (model.py, one pass)

```
idx (B, T) ─→ token_emb ─→ emb_dropout ─→ x
                                           │
            ┌──────────────────────────────┘
            ▼  (× n_layers TransformerBlocks)
    ┌─ attn_norm (RMSNorm) ─→ CausalSelfAttention ─→ + residual ─┐
    │                                                             │
    └─ ffn_norm  (RMSNorm) ─→ FeedForward (SwiGLU) ─→ + residual ─┘
                                           │
                                           ▼
                                        norm_f (RMSNorm)
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼ training                          ▼ inference
                 lm_head(x)                          lm_head(x[:, -1])
                 cross_entropy                       return logits only
```

Weight tying: `self.lm_head.weight = self.token_emb.weight` (shared tensor).
`sum(p.numel() for p in model.parameters())` already deduplicates by tensor
identity, so it counts weight-tied params once. Any param-counting code should
either trust this or explicitly dedupe by `id(p)` (see `teach.py:slide_16_param_breakdown`).

### KV cache — the important invariant

`NanoLLM.forward` accepts an optional `past_kv` (list of `(k, v)` tuples, one per
layer). The shape is `(B, n_heads, past_T, d_head)`. When `past_kv` is supplied:

- `RotaryPositionEmbedding.forward(x, T, start_pos=past_len)` — the new token's
  RoPE angle must come from the absolute position, **not** position 0. Getting
  this wrong is the single most likely way to break the cache.
- The causal mask is **not** applied on the decode step (T=1 means the single
  query is always the newest position; all cached K,V are strictly ≤ it).
- `CausalSelfAttention` returns `(output, (k, v))` where `(k, v)` is the
  *concatenated* cache for next step.
- Inference return value from `NanoLLM.forward` is always `(logits, list[(k, v)])`
  in inference mode — never `None`. Callers that discard the cache use
  `logits, _ = self(idx)`.

The `__main__` smoke test in `model.py` validates this with a prefill vs.
step-by-step equivalence check. **Run it after touching attention.**

`generate()` (no cache) is kept intentionally — do not delete it to "clean up"
the duplication with `generate_fast()`. The webinar depends on the A/B comparison.

### app.py — Gradio UI reuses everything else

`app.py` is the Gradio webinar console (8 tabs: Build Steps / Train / Effects /
Train Reports / Attention / Visualize / Generate / Benchmark). It does **not**
re-implement generation or visualization — it imports and calls the functions in
`model.py`, `teach.py`, `visualise.py`, `visualize_anim.py`, `build_viz.py`, and
`effect_viz.py`. The model is
loaded once into module-level singletons (`_MODEL`, `_TOKENIZER`, `_CONFIG`) at
startup so the first click feels instant. Handlers use `_require_loaded()` to
access them. Tests inject test fixtures by monkeypatching those three module
globals (see `tests/test_app.py`).

- **Train tab** streams training progress via `train.train_iter()`, a generator
  that yields structured dicts (`{"type": "step", ...}` / `{"type": "log", ...}`).
  Exposes sliders for `learning_rate`, `dropout`, and `warmup_steps`.
- **Effects tab** uses `effect_viz.py` to render schematic charts showing how
  each hyperparameter (epochs, batch size, LR, dropout, warmup) shapes the loss.
- **Build Steps tab** uses `build_viz.py` to render structural diagrams (sliding
  windows, transformer block, KV cache flow, test matrix) — pure matplotlib,
  no trained model needed.
- **Visualize tab** uses `visualize_anim.py` to collect all attention weights
  and hidden norms in a single hooked forward pass, then renders an animated
  HTML/JS three-panel visualization via `gr.HTML`.

Gradio is a lazy dep: `bash run.sh ui` installs it on-demand if missing, so
`bash run.sh setup` stays lean for users who only want CLI training.

### teach.py — hook-based, not modification-based

`teach.py` captures intermediate tensors via `register_forward_hook` /
`register_forward_pre_hook` on the model. It never modifies `model.py` forward
logic. When adding a new slide:

1. If it needs a new intermediate, extend `ForwardCapture._install()`.
2. If it operates only on already-captured state, just write a new `slide_*`
   function and wire it into `teach()`.
3. Target layer/head/query_pos are CLI args (`--layer 0 --head 0 --query_pos -1`).
   Negative `query_pos` resolves to the last token.

The `_sample_from_logits` helper in `model.py` is shared between `generate()`,
`generate_fast()`, and the sampling rollout slide — one source of truth for
temperature / top-k / top-p.

## What NOT to simplify

This code is read by students. The following are load-bearing pedagogy, not
redundancy:

- **Both `generate()` and `generate_fast()` exist** — webinar A/B demo.
- **Verbose step-by-step comments in `model.py`** (e.g. "Step 1: Project to Q, K, V")
  — teaching scaffolding.
- **Explicit RMSNorm / RoPE / SwiGLU classes** rather than PyTorch built-ins —
  the point is to show how they work. Don't swap in `nn.LayerNorm`, SDPA, etc.
- **The `scatter` / `sorted_logits` nucleus-sampling pattern in `_sample_from_logits`**
  looks fragile but is correct — `torch.sort` returns a permutation over *all*
  vocab positions, so every slot is overwritten. Leave it alone unless you
  rewrite the whole sampler.

## Design docs and memory

Approved design docs live in `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md`.
The webinar-ready enhancements (KV cache, val split, loss curve, teach.py) are
documented in `2026-04-20-nanollm-webinar-ready-design.md`.

## Quick reference

| Hyperparameter defaults | File | Notes |
|---|---|---|
| `d_model=384, n_layers=6, n_heads=6` | `config.py` | ~11M params before vocab; ~15M with vocab ~4096 |
| `max_seq_len=256` | `config.py` | Also sizes RoPE cos/sin tables and causal_mask buffer |
| `target_vocab_size=4096` | `config.py` | BPE merge target; 260 base (4 special + 256 bytes) + 3836 merges |
| `max_epochs=15, warmup_steps=200` | `config.py` | Linear warmup → cosine decay to 10% peak LR |
| bf16 autodetect | `config.amp_dtype` | Falls back to fp16 on non-Ampere GPUs |
