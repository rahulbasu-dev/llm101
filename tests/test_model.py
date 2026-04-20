"""Tests for NanoLLM: shapes, weight tying, KV-cache equivalence, causal mask.

These are the highest-value tests — if they fail, the model is fundamentally
broken. In particular, the KV-cache equivalence and causal-mask leakage tests
guard against the two classes of bug that are easy to introduce when editing
attention.
"""

import math
import torch
import pytest

from config import NanoLLMConfig
from model import (
    NanoLLM,
    RMSNorm,
    RotaryPositionEmbedding,
    CausalSelfAttention,
    _sample_from_logits,
)


# ═══════════════════════════════════════════════════════════════
# Fundamentals
# ═══════════════════════════════════════════════════════════════

def test_parameter_count_positive(tiny_model):
    n = sum(p.numel() for p in tiny_model.parameters())
    assert n > 0


def test_weight_tying(tiny_model):
    """lm_head.weight and token_emb.weight must be the *same tensor*, not a copy."""
    assert tiny_model.lm_head.weight is tiny_model.token_emb.weight


# ═══════════════════════════════════════════════════════════════
# Forward pass shapes
# ═══════════════════════════════════════════════════════════════

def test_forward_training_mode_shapes(tiny_model, tiny_config):
    B, T = 2, 8
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    targets = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, loss = tiny_model(idx, targets)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert loss.dim() == 0  # scalar
    assert torch.isfinite(loss)


def test_forward_inference_mode_shapes(tiny_model, tiny_config):
    B, T = 2, 8
    idx = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, caches = tiny_model(idx)
    # Inference returns last-position logits only
    assert logits.shape == (B, tiny_config.vocab_size)
    # Cache: one (k, v) per layer, each (B, n_heads, T, d_head)
    assert isinstance(caches, list)
    assert len(caches) == tiny_config.n_layers
    for k, v in caches:
        assert k.shape == (B, tiny_config.n_heads, T, tiny_config.d_head)
        assert v.shape == (B, tiny_config.n_heads, T, tiny_config.d_head)


def test_forward_raises_on_too_long_sequence(tiny_model, tiny_config):
    idx = torch.randint(0, tiny_config.vocab_size,
                        (1, tiny_config.max_seq_len + 1))
    with pytest.raises(AssertionError, match="exceeds max_seq_len"):
        tiny_model(idx)


# ═══════════════════════════════════════════════════════════════
# KV-cache equivalence (the most important correctness test)
# ═══════════════════════════════════════════════════════════════

def test_kv_cache_matches_full_pass(tiny_model, tiny_config):
    """Feeding the prompt all-at-once should produce the same last-position
    logits as prefilling, then feeding the last token with a KV cache.
    """
    idx = torch.randint(0, tiny_config.vocab_size, (1, 12))
    with torch.no_grad():
        full_logits, _ = tiny_model(idx)
        pre_logits, cache = tiny_model(idx[:, :-1])
        step_logits, _ = tiny_model(idx[:, -1:], past_kv=cache)
    max_diff = (full_logits - step_logits).abs().max().item()
    assert max_diff < 1e-4, f"KV-cache diverges: max|Δ|={max_diff:.2e}"


def test_kv_cache_multi_step_equivalence(tiny_model, tiny_config):
    """Equivalence must hold across many decode steps, not just the first one.
    Tests the invariant that `start_pos` in RoPE tracks the true absolute
    position as the cache grows.
    """
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 6))
    n_steps = 6  # total cached length will be 12 (prompt + steps)
    assert 6 + n_steps <= tiny_config.max_seq_len

    # ── Reference: prefix-of-length-k for each k ──
    with torch.no_grad():
        # Fake "chosen" tokens so both paths process identical sequences
        choices = torch.randint(0, tiny_config.vocab_size, (1, n_steps))
        full_sequence = torch.cat([prompt, choices], dim=1)

        # Path A — feed whole prefix at each step, read last logit
        per_step_full = []
        for k in range(prompt.size(1), full_sequence.size(1) + 1):
            logits_k, _ = tiny_model(full_sequence[:, :k])
            per_step_full.append(logits_k)

        # Path B — prefill + cached decode
        per_step_cache = []
        logits, cache = tiny_model(prompt)
        per_step_cache.append(logits)
        for t in range(n_steps):
            nxt = choices[:, t:t+1]
            logits, cache = tiny_model(nxt, past_kv=cache)
            per_step_cache.append(logits)

    assert len(per_step_full) == len(per_step_cache) == n_steps + 1
    for i, (a, b) in enumerate(zip(per_step_full, per_step_cache)):
        diff = (a - b).abs().max().item()
        assert diff < 1e-4, f"Step {i}: max|Δ|={diff:.2e}"


# ═══════════════════════════════════════════════════════════════
# Causal mask (no-leak) test
# ═══════════════════════════════════════════════════════════════

def test_causal_mask_no_future_leakage(tiny_model, tiny_config):
    """The logits at position i must not depend on tokens at positions > i.

    We perturb the token at position T-1 and verify that logits at positions
    0..T-2 are unchanged. If the causal mask were broken, later-position
    perturbations would leak backward.
    """
    B, T = 1, 10
    idx1 = torch.randint(0, tiny_config.vocab_size, (B, T))
    idx2 = idx1.clone()
    # Change the last token to something definitely different
    idx2[0, -1] = (idx1[0, -1] + 1) % tiny_config.vocab_size

    with torch.no_grad():
        # Use training-style forward so we get logits at every position
        logits1, _ = tiny_model(idx1, targets=idx1)
        logits2, _ = tiny_model(idx2, targets=idx2)

    # Positions 0..T-2 must be identical between the two runs
    diff = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
    assert diff < 1e-5, \
        f"Causal mask leaks: changing last token shifted earlier logits by {diff:.2e}"


# ═══════════════════════════════════════════════════════════════
# Component-level tests
# ═══════════════════════════════════════════════════════════════

def test_rmsnorm_preserves_shape_and_scales():
    norm = RMSNorm(dim=16)
    x = torch.randn(3, 5, 16) * 10.0
    y = norm(x)
    assert y.shape == x.shape
    # After RMSNorm (with default weight=1), rms(y) should be ≈ 1 per position
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_rope_start_pos_produces_different_angles():
    """RoPE at start_pos=0 and start_pos=5 must produce different rotated
    vectors — this is what makes positional information actually work.
    """
    rope = RotaryPositionEmbedding(d_head=16, max_seq_len=32)
    x = torch.randn(1, 1, 1, 16)  # single position
    a = rope(x, seq_len=1, start_pos=0)
    b = rope(x, seq_len=1, start_pos=5)
    assert not torch.allclose(a, b), \
        "RoPE at pos=0 and pos=5 should yield different outputs"


def test_sampling_helper_shapes(tiny_config):
    """_sample_from_logits should return (B, 1) token IDs within vocab range."""
    B, V = 4, tiny_config.vocab_size
    logits = torch.randn(B, V)
    out = _sample_from_logits(logits, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (B, 1)
    assert out.min() >= 0
    assert out.max() < V


def test_sampling_greedy_picks_argmax(tiny_config):
    """With temperature→0 and top_k=1, sampling must be deterministic on argmax."""
    B, V = 4, tiny_config.vocab_size
    logits = torch.randn(B, V)
    out = _sample_from_logits(logits, temperature=0.001, top_k=1, top_p=1.0)
    expected = logits.argmax(dim=-1, keepdim=True)
    assert torch.equal(out, expected)


# ═══════════════════════════════════════════════════════════════
# generate() and generate_fast() behavior
# ═══════════════════════════════════════════════════════════════

def test_generate_produces_correct_shape(tiny_model, tiny_config):
    B, T = 1, 4
    prompt = torch.randint(0, tiny_config.vocab_size, (B, T))
    n_new = 5
    out = tiny_model.generate(prompt, max_new_tokens=n_new)
    assert out.shape == (B, T + n_new)
    # All tokens in vocab range
    assert (out >= 0).all() and (out < tiny_config.vocab_size).all()


def test_generate_fast_produces_correct_shape(tiny_model, tiny_config):
    B, T = 1, 4
    prompt = torch.randint(0, tiny_config.vocab_size, (B, T))
    n_new = 5
    out = tiny_model.generate_fast(prompt, max_new_tokens=n_new)
    assert out.shape == (B, T + n_new)
    assert (out >= 0).all() and (out < tiny_config.vocab_size).all()


def test_generate_fast_matches_generate_greedy(tiny_model, tiny_config):
    """With deterministic sampling (temp→0, top_k=1), generate and generate_fast
    must produce identical output token sequences.
    """
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    n_new = 6

    torch.manual_seed(123)
    a = tiny_model.generate(prompt, max_new_tokens=n_new,
                            temperature=0.001, top_k=1, top_p=1.0)
    torch.manual_seed(123)
    b = tiny_model.generate_fast(prompt, max_new_tokens=n_new,
                                 temperature=0.001, top_k=1, top_p=1.0)
    assert torch.equal(a, b), \
        f"generate() and generate_fast() diverge under greedy decoding:\n  {a}\n  {b}"


def test_generate_fast_respects_context_limit(tiny_model, tiny_config):
    """Requesting more tokens than fit in max_seq_len must stop, not crash."""
    prompt = torch.randint(0, tiny_config.vocab_size,
                           (1, tiny_config.max_seq_len - 2))
    out = tiny_model.generate_fast(prompt, max_new_tokens=50)
    # Should never exceed max_seq_len
    assert out.size(1) <= tiny_config.max_seq_len
