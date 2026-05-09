"""LLM101 — A complete GPT-style Transformer language model built from scratch.

Architecture choices mirror modern LLMs (LLaMA / Mistral / Qwen):
  • RMSNorm (not LayerNorm)
  • Rotary Position Embeddings (RoPE)
  • SwiGLU activation (not ReLU/GELU)
  • Pre-Norm (not Post-Norm)
  • Weight tying (embedding ↔ lm_head)
  • Combined QKV projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NanoLLMConfig


# ═══════════════════════════════════════════════════════════════
# 1. RMSNorm — Root Mean Square Layer Normalization
# ═══════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm (Zhang & Sennrich, 2019).

    Unlike LayerNorm, RMSNorm:
      - Does NOT subtract the mean (no re-centering)
      - Does NOT have a bias parameter
      - Only rescales by root-mean-square → fewer ops, same stability

    Used in: LLaMA, LLaMA-2, Mistral, Qwen, Gemma
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # Learnable scale γ

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rms = 1 / sqrt(mean(x²) + eps)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ═══════════════════════════════════════════════════════════════
# 2. Rotary Position Embeddings (RoPE)
# ═══════════════════════════════════════════════════════════════

class RotaryPositionEmbedding(nn.Module):
    """RoPE (Su et al., 2021) — encodes absolute position via rotation,
    but the dot product Q·K naturally captures RELATIVE position.

    For each pair of dimensions (2i, 2i+1), rotate by angle = pos × θ_i
    where θ_i = 10000^(-2i/d_head).

    Used in: LLaMA, Mistral, Qwen, PaLM, CodeLLaMA, Phi
    """

    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        # Frequency bands: θ_i = base^(-2i/d) for i in [0, d/2)
        freqs = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        # Position indices
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # Outer product: angles[pos, i] = pos × θ_i
        angles = torch.outer(t, freqs)  # (max_seq_len, d_head/2)
        # Cache cos and sin
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int, start_pos: int = 0) -> torch.Tensor:
        """Apply rotary embeddings to x.

        x shape: (batch, n_heads, seq_len, d_head)
        start_pos: absolute starting position (for KV-cache decode, this equals
                   the number of already-cached tokens).
        """
        cos = self.cos_cached[start_pos:start_pos + seq_len]  # (seq_len, d_head/2)
        sin = self.sin_cached[start_pos:start_pos + seq_len]
        # Broadcast to (1, 1, seq_len, d_head/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Split into even/odd dimension pairs
        x1 = x[..., 0::2]  # Even dims
        x2 = x[..., 1::2]  # Odd dims

        # 2D rotation: [x1, x2] → [x1·cos − x2·sin, x1·sin + x2·cos]
        out = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1)

        return out.flatten(-2)  # Interleave even/odd back together


# ═══════════════════════════════════════════════════════════════
# 3. Causal Multi-Head Self-Attention
# ═══════════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """Multi-Head Self-Attention with:
      - Combined QKV projection (single matmul, more efficient)
      - RoPE on Q and K
      - Causal (lower-triangular) mask
      - Scaled dot-product attention

    This is the core mechanism: each token computes a weighted combination
    of ALL preceding tokens' value vectors, where weights come from
    query-key compatibility scores.
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model

        # Single linear → split into Q, K, V (faster than 3 separate projections)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection: concat(heads) → d_model
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(config.d_head, config.max_seq_len)

        # Causal mask: lower triangular = token i can attend to tokens 0..i
        causal = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", causal.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor, past_kv=None):
        """Forward pass with optional KV cache.

        Args:
            x: (B, T, d_model) input activations for the NEW tokens only.
            past_kv: Optional (past_k, past_v) tuple from previous decode steps.
                     Shapes: (B, n_heads, past_T, d_head).

        Returns:
            out:    (B, T, d_model)
            new_kv: (k, v) with shapes (B, n_heads, past_T + T, d_head) for caching.
        """
        B, T, C = x.shape

        # ── Step 1: Project to Q, K, V ──
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # ── Step 2: Apply RoPE at the correct absolute position ──
        # During cached decode, the new token sits at position `past_len`, so
        # its RoPE angle must come from that row of the cos/sin tables.
        past_len = past_kv[0].size(2) if past_kv is not None else 0
        q = self.rope(q, T, start_pos=past_len)
        k = self.rope(k, T, start_pos=past_len)

        # ── Step 2b: Append to cache ──
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, nh, past_T + T, d_head)
            v = torch.cat([past_v, v], dim=2)
        new_kv = (k, v)

        # ── Step 3: Scaled dot-product attention ──
        scale = 1.0 / math.sqrt(self.d_head)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, nh, T, total_T)

        # ── Step 4: Causal mask — only needed during prefill ──
        # During cached decode, T=1 and the single query is the newest position;
        # all cached keys are strictly ≤ it, so no masking is required.
        if past_kv is None:
            attn_scores = attn_scores.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )

        # ── Step 5: Softmax → attention weights ──
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ── Step 6: Weighted sum of value vectors ──
        out = torch.matmul(attn_weights, v)  # (B, nh, T, d_head)

        # ── Step 7: Concatenate heads and project ──
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out)), new_kv


# ═══════════════════════════════════════════════════════════════
# 4. Feed-Forward Network with SwiGLU
# ═══════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """Position-wise FFN with SwiGLU activation (Shazeer, 2020).

    Standard FFN:   ReLU(x·W₁) · W₂
    SwiGLU FFN:     (Swish(x·W_gate) ⊙ x·W_up) · W_down

    SwiGLU has 3 weight matrices instead of 2, so we use 2/3 of d_ff
    for the hidden dim to keep total parameter count ≈ the same.

    This is where the model stores and retrieves factual knowledge.
    Attention routes information; FFN computes on it.
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        # Adjusted hidden dim for SwiGLU (3 matrices vs 2)
        hidden_dim = int(2 * config.d_ff / 3)
        # Round to multiple of 8 for GPU tensor core efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8

        self.gate_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) ⊙ up, then project down
        # silu(x) = x × sigmoid(x), also called "swish"
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ═══════════════════════════════════════════════════════════════
# 5. Transformer Block
# ═══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """Single Transformer decoder block (Pre-Norm variant).

    Data flow:
        x → RMSNorm → Self-Attention → + residual
          → RMSNorm → FFN            → + residual → output

    Pre-Norm (normalize BEFORE sublayer) is more stable than
    Post-Norm (normalize AFTER). Used in GPT-2+, LLaMA, all modern LLMs.
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, past_kv=None):
        """Forward pass, threading an optional KV cache through attention."""
        # Residual connection: x + sublayer(norm(x))
        attn_out, new_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# ═══════════════════════════════════════════════════════════════
# 6. NanoLLM — Complete Language Model
# ═══════════════════════════════════════════════════════════════

class NanoLLM(nn.Module):
    """Decoder-only Transformer language model.

    Pipeline:
        token_ids → Embedding → [TransformerBlock × N] → RMSNorm → Linear → logits

    Weight tying: The embedding matrix and the final lm_head share weights.
    This forces the model to use the same vector space for input tokens
    and output predictions — reduces parameters and improves quality.
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__()
        self.config = config

        # Token embedding: vocab_size → d_model
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization before output projection
        self.norm_f = RMSNorm(config.d_model)

        # Language model head: d_model → vocab_size
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (Press & Wolf, 2017)
        self.lm_head.weight = self.token_emb.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/√(2·n_layers) for stable deep training
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        # Deduplicate by tensor identity so tied weights are counted once.
        n_params = sum(p.numel() for p in {id(p): p for p in self.parameters()}.values())
        print(f"LLM101 initialised: {n_params:,} parameters ({n_params/1e6:.2f}M)")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv=None):
        """
        Args:
            idx:     (B, T) token indices (new tokens only when past_kv supplied)
            targets: (B, T) target indices (training mode)
            past_kv: Optional list of (k, v) tuples, one per layer, from prior decode steps.

        Returns:
            Training mode:  (logits, loss)                    — past_kv must be None
            Inference mode: (logits, new_caches)              — new_caches is a list
                                                                 of per-layer (k, v) tuples
                                                                 (existing callers that did
                                                                 `logits, _ = self(idx)`
                                                                 are unaffected.)
        """
        B, T = idx.shape
        past_len = past_kv[0][0].size(2) if past_kv is not None else 0
        total_len = past_len + T
        assert total_len <= self.config.max_seq_len, \
            f"Sequence length {total_len} exceeds max_seq_len {self.config.max_seq_len}"

        # Embed tokens (position is handled by RoPE inside attention)
        x = self.emb_dropout(self.token_emb(idx))  # (B, T, d_model)

        # Pass through all transformer blocks, collecting new caches
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv is not None else None
            x, new_kv = block(x, past_kv=layer_past)
            new_caches.append(new_kv)

        # Final norm
        x = self.norm_f(x)  # (B, T, d_model)

        if targets is not None:
            # Training mode: compute loss over all positions
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # ignore padding
            )
            return logits, loss
        # Inference mode: only compute logits for last token (efficient)
        logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)
        return logits, new_caches

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k + nucleus (top-p) sampling.

        At each step:
          1. Forward pass → logits for next token
          2. Apply temperature scaling (controls randomness)
          3. Filter to top-k candidates
          4. Further filter by cumulative probability (top-p / nucleus)
          5. Sample from filtered distribution
          6. Append to sequence and repeat
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            idx_crop = idx[:, -self.config.max_seq_len:]

            # Forward pass (inference mode — only last-token logits)
            logits, _ = self(idx_crop)  # (B, vocab_size)

            # Temperature: divide logits → higher T = more random
            logits = logits / max(temperature, 1e-8)

            # Top-k: zero out everything below the k-th highest logit
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                topk_vals, _ = torch.topk(logits, k)
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            # Top-p (nucleus): keep smallest set of tokens with cumprob ≥ p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Mask tokens beyond the nucleus
                mask = cumprobs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                # Scatter back to original positions
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    @torch.no_grad()
    def generate_fast(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Cached autoregressive generation.

        Same semantics as `generate()`, but each decode step reuses the K,V of
        all previously-processed tokens instead of recomputing them. This makes
        each step O(total_len) rather than O(total_len²).

        Pipeline:
          1. Prefill  — one forward pass on the whole prompt, produces the first
                        per-layer (k,v) cache AND the logits for the prompt's
                        last position.
          2. Decode   — for each new token: single-token forward with past_kv,
                        append new (k,v) to the cache, sample, repeat.
        """
        self.eval()
        B, prompt_len = idx.shape
        # Respect the context window: prefill can use at most max_seq_len tokens
        if prompt_len > self.config.max_seq_len:
            idx = idx[:, -self.config.max_seq_len:]
            prompt_len = self.config.max_seq_len

        # ── Prefill: process the entire prompt once ──
        logits, past_kv = self(idx)  # logits: (B, vocab), past_kv: list of (k,v)

        generated = idx
        for _ in range(max_new_tokens):
            # Respect context window: if the cache is full, stop.
            if past_kv[0][0].size(2) >= self.config.max_seq_len:
                break

            # Sample next token from the current logits
            next_token = _sample_from_logits(logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)

            # ── Decode step: single-token forward with cache ──
            logits, past_kv = self(next_token, past_kv=past_kv)

        return generated


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Temperature / top-k / top-p sampling. Shared between generate paths."""
    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        logits = logits.masked_fill(logits < topk_vals[:, [-1]], float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)  # (B, 1)


# ═══════════════════════════════════════════════════════════════
# Quick shape verification
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config = NanoLLMConfig(vocab_size=1000)  # Dummy vocab for test
    model = NanoLLM(config)
    model.eval()  # Disable dropout for deterministic comparisons

    # ── Test 1: training-mode forward ──
    batch = torch.randint(0, 1000, (2, 64))
    targets = torch.randint(0, 1000, (2, 64))
    logits, loss = model(batch, targets)
    print(f"Forward pass OK: logits {logits.shape}, loss {loss.item():.4f}")

    # ── Test 2: generate() smoke ──
    prompt = torch.randint(0, 1000, (1, 5))
    output = model.generate(prompt, max_new_tokens=10)
    print(f"generate() OK: {prompt.shape} → {output.shape}")

    # ── Test 3: KV-cache equivalence ──
    # Feeding a prompt all at once should produce the same next-token logits
    # as feeding the prompt piece-by-piece with a KV cache.
    prompt = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        full_logits, _ = model(prompt)  # (1, vocab_size) — last-position logits

        # Same prompt, chunked via cache: first T-1 as prefill, then 1-token decode
        pre_logits, cache = model(prompt[:, :-1])
        step_logits, _ = model(prompt[:, -1:], past_kv=cache)

    max_diff = (full_logits - step_logits).abs().max().item()
    ok = max_diff < 1e-4
    print(f"KV-cache equivalence: max |Δlogit| = {max_diff:.2e}  → {'OK' if ok else 'FAIL'}")

    # ── Test 4: generate_fast() smoke ──
    output_fast = model.generate_fast(prompt[:, :5], max_new_tokens=10)
    print(f"generate_fast() OK: (1, 5) → {tuple(output_fast.shape)}")
