# Building LLM101 — Step by Step

A chronicle of how this project actually came together, in the order a new builder
should follow. Unlike `nanollm-guide.md` (which is the *theory*, RNN → Transformer)
and `REFERENCES.md` (which maps features to Raschka's book), this page captures the
**construction journey**: what to build first, which decisions matter early, and
the gotchas you only find after running the code.

**Target reader:** Someone who understands self-attention in principle and wants
to replicate this repo from an empty directory. About 4-8 hours of focused work
end-to-end.

---

## Prerequisites

- Python 3.10+ (we use `match`-statement-friendly typing)
- A GPU with ≥ 8 GB VRAM is ideal; CPU works for everything except full training
- 2-3 GB free disk (PyTorch wheels are chunky)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy pytest
```

On **Windows without WSL2**, add `PYTHONIOENCODING=utf-8` when running Python directly
— several files print box-drawing characters and arrows that cp1252 chokes on.

---

## Step 1 — Pin the hyperparameters (`config.py`)

Start here, not at `model.py`. A single dataclass owning every tunable keeps the
rest of the code uncluttered and makes reproducibility mechanical.

```python
@dataclass
class NanoLLMConfig:
    vocab_size: int = 0          # assigned at runtime after tokenizer trains
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int = 1536
    max_seq_len: int = 256
    # ... training + generation hyperparams
```

**Non-obvious choice:** `vocab_size` starts at **0**. Every script that builds a
`NanoLLM` must set `config.vocab_size = tokenizer.vocab_size` *before* instantiation.
This avoids hardcoding a vocab size and keeps the tokenizer and model coupled through
a single variable.

Derived values go in `@property`:

```python
@property
def d_head(self) -> int:
    assert self.d_model % self.n_heads == 0
    return self.d_model // self.n_heads
```

Property-with-assertion is a clean way to catch misconfiguration at access time
rather than during a forward pass where the error is less clear.

---

## Step 2 — Byte-level BPE tokenizer (`tokenizer.py`)

BPE from scratch in ~200 lines. Byte-level means the base vocab is always 256
(every possible byte), so any UTF-8 text round-trips losslessly — no `<UNK>`
explosion on new characters.

**Vocabulary layout:**
```
0-3     : <PAD>, <BOS>, <EOS>, <UNK>    (4 special tokens)
4-259   : raw bytes 0x00-0xFF           (256 base tokens)
260+    : learned BPE merges            (up to target_vocab_size)
```

**Training loop (naïve but readable):**
```python
while len(merges) < target_merges:
    pair_counts = Counter((tokens[i], tokens[i+1]) for i in range(len(tokens)-1))
    best_pair = max(pair_counts, key=pair_counts.get)
    # replace every occurrence in `tokens` with the new merge id
```

**Gotcha — test corpus variety:** A corpus of `"hello world. " * 100` will
collapse into a **single** token after ~10 merges, because the whole phrase
repeats identically. BPE tests need real variation. Our `mock_corpus` fixture
rotates through three different sentences precisely to avoid this.

**Performance caveat (not fixed):** Encoding with this tokenizer is O(merges × N)
because `encode()` re-scans every merge rule on every call. For a 1 MB corpus
and 3800 merges that's ~4 billion ops. Fine for education. Parked as future work;
see `docs/superpowers/specs/2026-04-20-nanollm-webinar-ready-design.md` for what
was explicitly out-of-scope.

---

## Step 3 — Sliding-window dataset (`dataset.py`)

Turns a token stream into `(input_ids, targets)` pairs where `targets = input_ids
shifted +1`. This is the core causal-LM training invariant.

```python
for i in range(0, len(tokens) - seq_len - 1, stride):
    input_ids  = tokens[i     : i + seq_len]
    target_ids = tokens[i + 1 : i + seq_len + 1]
    self.samples.append((input_ids, target_ids))
```

**Non-obvious choice:** default stride = `seq_len // 2` (50 % overlap). Non-overlap
is a cleaner evaluation metric but halves your samples. The overlap is slightly
redundant for training but the cheap memorization improves next-token prediction
at the window boundaries.

**Verification:** `tests/test_dataset.py::test_target_is_input_shifted_by_one`
asserts `target[:-1] == input[1:]` — a simple property-based check that would
catch a subtle off-by-one.

---

## Step 4 — Model components (`model.py`)

**Build order matters.** If you build the components bottom-up and test each,
each failure has one suspect.

### 4.1 RMSNorm (5 lines)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight
```

Test: `rms(y) ≈ 1` after norm. See `test_model.py::test_rmsnorm_preserves_shape_and_scales`.

### 4.2 Rotary Position Embedding (~30 lines)

Precompute `cos` and `sin` tables at `__init__`, slice at `forward`:

```python
def forward(self, x, seq_len, start_pos=0):
    cos = self.cos_cached[start_pos:start_pos + seq_len]
    sin = self.sin_cached[start_pos:start_pos + seq_len]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)
```

**`start_pos` matters for the KV cache (Step 7).** When you generate the next
token with a cache, it sits at position `past_len`, not 0. Getting this wrong
is the single most likely bug in a KV-cache implementation.

Test: `test_rope_start_pos_produces_different_angles`.

### 4.3 Causal self-attention

Combined QKV projection (one matmul instead of three), multi-head reshape,
scaled dot-product, causal mask via `masked_fill(mask == 0, -inf)`:

```python
qkv = self.qkv_proj(x).chunk(3, dim=-1)    # Q, K, V all at once
q, k, v = [t.view(B, T, nh, dh).transpose(1, 2) for t in qkv]
q = self.rope(q, T); k = self.rope(k, T)
scores = (q @ k.transpose(-2, -1)) / sqrt(d_head)
scores.masked_fill_(self.causal_mask[:, :, :T, :T] == 0, -inf)
attn = F.softmax(scores, dim=-1) @ v
```

**Property-based correctness check:** perturbing token T-1 must not change
logits at positions 0..T-2. `test_causal_mask_no_future_leakage` asserts
exactly this. It catches subtle mask bugs that shape-only tests miss.

### 4.4 SwiGLU feed-forward

```python
gate = F.silu(self.gate_proj(x))
up   = self.up_proj(x)
out  = self.down_proj(gate * up)
```

Not ReLU. Not GELU. Not a single matrix. Three matrices because SwiGLU is a
*gated* activation — one path gates the other. We pick hidden_dim = `2 × d_ff / 3`
and round to a multiple of 8, so total params ≈ a standard ReLU FFN but with
a gating mechanism.

### 4.5 Transformer block = Pre-Norm + residual × 2

```python
x = x + self.attn(self.attn_norm(x))
x = x + self.ffn(self.ffn_norm(x))
```

Pre-Norm (normalize *before* the sublayer, add residual *after*) is what keeps
deep stacks stable without a learning-rate-warmup knife-edge. Modern LLMs all
use this pattern.

### 4.6 NanoLLM — putting it together

Token embedding → N blocks → final norm → LM head. The one non-obvious line:

```python
self.lm_head.weight = self.token_emb.weight    # weight tying (shared tensor)
```

This is not a copy — it's the **same tensor**. `sum(p.numel() for p in model.parameters())`
deduplicates by tensor identity, so it counts once. Verified by
`test_weight_tying` using `is` not `torch.equal`.

Residual init scaling (GPT-2 trick):
```python
for pn, p in self.named_parameters():
    if pn.endswith(("out_proj.weight", "down_proj.weight")):
        nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

Keeps activation variance stable as depth grows. Easy to miss. Preserve it.

---

## Step 5 — Training loop (`train.py`)

Order of things that matter, roughly in descending impact:

1. **bf16 mixed precision** on Ampere+ GPUs — 2× speedup, basically free.
2. **AdamW with weight-decay groups** — decay applies to 2D weights only,
   never to biases or norms. Standard.
3. **Linear warmup → cosine decay to 10 % of peak LR** — 200 warmup steps.
4. **Grad clip 1.0** — cheap insurance against loss spikes.
5. **Val split is sequential 90/10** — random split leaks through overlapping
   windows.
6. **"Best" checkpoint picked on *val* loss**, not train loss. Otherwise the
   best checkpoint is usually the one right before overfitting starts.

```python
for epoch in range(max_epochs):
    for inp, tgt in train_loader:
        with autocast(dtype=bf16):
            logits, loss = model(inp, tgt)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        optimizer.zero_grad(set_to_none=True)

    val_loss = evaluate(model, val_loader)   # eval mode, no grad
    if val_loss < best_val_loss:
        save_checkpoint("best.pt")
```

At the end, dump a `loss_curve.png` — train per-step (faint) + train/val per-epoch
(dots), log-y axis. Students can *see* the gap between train and val open up —
that IS overfitting.

---

## Step 6 — Generation (naïve version)

Autoregressive loop: forward full context, sample from last-position logits,
append, repeat. In ~20 lines:

```python
for _ in range(max_new_tokens):
    idx_crop = idx[:, -max_seq_len:]
    logits, _ = self(idx_crop)
    next_tok = _sample_from_logits(logits, temperature, top_k, top_p)
    idx = torch.cat([idx, next_tok], dim=1)
```

**The sampling filter is worth keeping intact:**

```python
# top-k: zero out everything below the k-th highest logit
topk_vals, _ = torch.topk(logits, k)
logits.masked_fill_(logits < topk_vals[:, [-1]], -inf)

# top-p (nucleus): sort, cumulative-softmax, mask beyond p
sorted_logits, sorted_idx = logits.sort(descending=True)
probs = F.softmax(sorted_logits, dim=-1)
mask = probs.cumsum(-1) - probs >= top_p
sorted_logits.masked_fill_(mask, -inf)
logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
```

The `scatter` at the end looks fragile but is correct: `sorted_idx` is a permutation
over ALL vocab positions, so every slot is written.

---

## Step 7 — The KV cache (`generate_fast`)

Without a cache, every decode step recomputes Q, K, V for the entire context.
With a cache, you keep K, V from prior steps and only compute them for the *new*
token. This makes each step O(total_len) instead of O(total_len²).

**Four edits needed:**

1. `RotaryPositionEmbedding.forward(x, seq_len, start_pos=0)` — index RoPE from
   the absolute position.
2. `CausalSelfAttention.forward(x, past_kv=None) → (out, new_kv)` — concatenate
   cached K,V with new K,V.
3. `TransformerBlock.forward(x, past_kv=None) → (out, new_kv)` — thread through.
4. `NanoLLM.forward(idx, targets=None, past_kv=None)` — collect per-layer caches;
   return `list[(k, v)]` in inference mode.

**Two invariants that are easy to miss:**

- **RoPE offset.** The new token's RoPE angle must match what it would have
  been during a single-pass forward — so `start_pos=past_len`, not 0.
- **Causal mask.** Applied during **prefill** (T > 1), skipped during **decode**
  (T = 1). When the only query is the newest position, all cached keys are
  strictly ≤ it — no mask needed.

**Correctness test:** prefill(prompt) then 1-step decode must match a full
forward on prompt. `test_kv_cache_matches_full_pass` asserts
`max |Δlogit| < 1e-4` — this is the test that catches a broken RoPE offset.

**Keep both paths.** Deleting `generate()` to "clean up" is tempting but loses
the pedagogical A/B comparison. The webinar benchmark tab literally shows both
numbers side-by-side.

---

## Step 8 — Teaching slides (`teach.py`)

Here's a pattern worth stealing: capture intermediate tensors via **forward
hooks** without modifying `model.py` at all.

```python
class ForwardCapture:
    def __init__(self, model, target_layer=0):
        self.store = {}
        block = model.blocks[target_layer]
        block.attn.register_forward_pre_hook(self._attn_hook)
        block.ffn.register_forward_pre_hook(lambda m, i: self.store.__setitem__("ffn_in", i[0]))
        # ...

    def _attn_hook(self, module, inp):
        # Recompute Q, K, V, scores, weights from the input and stash them
        ...
```

The alternative — sprinkling `return_attention=True` kwargs through the production
code — poisons `model.py`. Hooks keep teaching instrumentation strictly orthogonal
to the model.

**Slide inventory (16):** tokenization, embeddings, Q/K/V projections, raw scores,
causal mask applied, softmax weights, weighted value sum, all heads side-by-side,
FFN delta, top-k logits, sampling rollout, RoPE, scaling rationale, temperature
effect, greedy vs sampling, parameter breakdown. See `REFERENCES.md` for the
chapter mapping.

**Re-render on-the-fly** (don't cache PNGs to disk and serve those): the whole
point is that a student can change the prompt and see the attention flip.
This is why the Gradio UI (Step 10) calls `teach.py` functions directly rather
than loading `teaching_plots/*.png`.

---

## Step 9 — Attention visualisation (`visualise.py`)

Two things worth rendering:

1. **Per-head heatmap** — `attn_weights[layer][0, head]`, shape `(T, T)`, viridis
   colormap, row = query, column = key.
2. **Attention rollout** (Abnar & Zuidema 2020) — multiply attention matrices
   across all layers to see *effective* attention flow. More interpretable
   than any single layer.

```python
def compute_attention_rollout(attention_maps):
    result = None
    for attn in attention_maps:
        attn_avg = attn.mean(dim=1)  # average across heads
        # account for residual connections
        attn_with_residual = 0.5 * attn_avg + 0.5 * torch.eye(attn_avg.size(-1))
        attn_with_residual /= attn_with_residual.sum(-1, keepdim=True)
        result = attn_with_residual if result is None else attn_with_residual @ result
    return result
```

---

## Step 10 — Gradio UI (`app.py`)

A 4-tab browser console as the webinar entrypoint: **Generate · Teach ·
Attention · Benchmark**. All four handlers import and call existing functions
from `teach.py` and `visualise.py` — no logic is re-implemented.

**Two Gradio patterns worth learning:**

1. **Streaming via `yield`.** A generator handler gives you per-token typewriter
   output for free:
   ```python
   def generate_stream(prompt, ...):
       for step in range(max_new_tokens):
           next_tok = sample(...)
           out_ids.append(int(next_tok.item()))
           yield tokenizer.decode(out_ids)    # browser updates each yield
   ```
2. **Module-level singletons + `monkeypatch` for tests.** Load the model once
   at startup into `_MODEL`; tests inject a tiny model via
   `monkeypatch.setattr(app, "_MODEL", tiny_model)` — no I/O, sub-second tests.

**Version pin:** gradio `>=5,<6`. The 4.44.x line has a gradio-client schema
introspection bug (`TypeError: argument of type 'bool' is not iterable`) that
crashes `launch()`. Upgrade fixes it. Cost: one extra dep. Worth it — the public
`--share` URL turns the whole console into a Discord-shareable webinar demo.

---

## Step 11 — Tests (`tests/`)

42 unit + 6 UI smoke tests, CPU-only, ~18 s. The high-value ones:

| Test | What it guards |
|------|-----------------|
| `test_kv_cache_matches_full_pass` | The RoPE `start_pos` invariant |
| `test_kv_cache_multi_step_equivalence` | Cache growth is correct across N steps, not just step 1 |
| `test_causal_mask_no_future_leakage` | Perturbing token T-1 leaves earlier logits unchanged |
| `test_weight_tying` | `is`-check: `lm_head.weight` and `token_emb.weight` are the SAME tensor, not copies |
| `test_training_step_reduces_loss_on_tiny_overfit` | End-to-end backward/optimizer actually update weights |

**The pattern:** test invariants (causal-mask no-leak, cache equivalence), not
implementation details (tensor shapes are implied). Invariant tests survive
refactoring.

Run:
```bash
bash run.sh test                                # everything, ~18s
pytest tests/ -k "kv_cache"                     # just the KV-cache tests
pytest tests/test_model.py -v                   # one module, verbose
pytest tests/ -x                                # stop on first failure
```

---

## Step 12 — Running it all

Commands unified through one bash dispatcher (`run.sh`):

```bash
bash run.sh setup       # venv + torch + TinyShakespeare
bash run.sh verify      # model.py __main__ sanity check
bash run.sh train       # full training (~5-15 min on RTX 4080)
bash run.sh generate --fast
bash run.sh benchmark   # generate vs generate_fast
bash run.sh teach       # 16 static PNGs
bash run.sh visualise   # attention heatmaps + rollout
bash run.sh ui          # Gradio console (localhost:7860)
bash run.sh test        # pytest suite
```

Explicit subcommands > magic auto-detection. A new user running
`bash run.sh` with no arg gets a helpful usage message.

---

## Gotchas encountered (in build order)

| # | Gotcha | Fix |
|---|---|---|
| 1 | `cp1252` console can't print box-drawing / arrow characters | Add `PYTHONIOENCODING=utf-8` when running on raw Windows shell |
| 2 | BPE collapses a repetitive test corpus into one token | Use varied seed text; `mock_corpus` fixture rotates 3 sentences |
| 3 | KV cache silently wrong if RoPE uses `start_pos=0` at decode | Enforce with `test_kv_cache_matches_full_pass` (max diff < 1e-4) |
| 4 | "Best checkpoint" on train loss misleads when overfitting starts | Use val loss; sequential 90/10 split |
| 5 | Static PNGs in `teaching_plots/` don't update with new prompts | `teach.py` functions are importable; UI calls them live |
| 6 | Gradio 4.44.x crashes on launch (`additionalProperties` bool bug) | Pin `gradio>=5,<6` in `run.sh ui` |
| 7 | Forward hooks fire AFTER the return; we want to capture Q,K,V before softmax | Use `register_forward_pre_hook` and recompute, don't try to intercept mid-forward |
| 8 | Weight-tying bugs invisible to `torch.equal` | Test with `is` (tensor identity), not value equality |

---

## What to skip if you're pressed for time

If you only have 2 hours, build Steps 1-6 (config → tokenizer → dataset → model
→ training → naïve generate). That alone gives you a working decoder-only LM
that learns Shakespeare in under an hour on an RTX-class GPU.

If you have 4 hours, add Step 7 (KV cache) — the pedagogical payoff is large
because the speedup is dramatic and visible.

Step 10 (Gradio UI) is webinar-specific; skip unless you're actually demoing.

---

## Further reading

- **Theory:** `nanollm-guide.md` (RNN → LSTM → attention → Transformer)
- **Book map:** `REFERENCES.md` (slides ↔ Raschka's chapters)
- **Architecture diagrams:** `README.md` (end-to-end + block internals + KV flow)
- **Design record:** `docs/superpowers/specs/2026-04-20-nanollm-webinar-ready-design.md`
- **Agent guidance:** `CLAUDE.md` (for future Claude Code sessions)
