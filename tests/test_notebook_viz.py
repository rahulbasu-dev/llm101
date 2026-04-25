"""Tests for notebook_viz — pure render helpers for the new tabs.

Covers each public function with the smallest possible inputs (the tiny
fixtures in conftest.py). Asserts shapes / structure rather than pixel
content, since matplotlib output isn't byte-stable across versions.
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import notebook_viz as nv
from tokenizer import NUM_BASE, BYTE_OFFSET


# ───────────────────────────────────────────────────────────────
# Section 2 — Tokenizer
# ───────────────────────────────────────────────────────────────

def test_encode_breakdown_round_trips(trained_tokenizer):
    text = "the cat sat"
    b = nv.encode_breakdown(trained_tokenizer, text)
    assert b["input"] == text
    assert b["decoded"] == text
    assert b["round_trip_match"] is True
    assert isinstance(b["encoded"], list)
    assert all(isinstance(x, int) for x in b["encoded"])
    assert len(b["tokens"]) == len(b["encoded"])
    for tok in b["tokens"]:
        assert tok["kind"] in {"SPECIAL", "BYTE", "MERGE"}
        assert isinstance(tok["id"], int)
        assert isinstance(tok["label"], str)


def test_encode_breakdown_kind_classification(trained_tokenizer):
    """Kind labels must match the actual id ranges."""
    b = nv.encode_breakdown(trained_tokenizer, "the cat sat on the mat")
    for tok in b["tokens"]:
        if tok["id"] >= NUM_BASE:
            assert tok["kind"] == "MERGE"
        elif tok["id"] >= BYTE_OFFSET:
            assert tok["kind"] == "BYTE"
        else:
            assert tok["kind"] == "SPECIAL"


def test_format_breakdown_returns_text(trained_tokenizer):
    b = nv.encode_breakdown(trained_tokenizer, "hi")
    out = nv.format_breakdown(b)
    assert "Original:" in out
    assert "Round-trip match: True" in out
    assert "Token breakdown" in out


def test_draw_tokenizer_overview_returns_figure(trained_tokenizer):
    fig = nv.draw_tokenizer_overview(trained_tokenizer)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # composition + compression
    plt.close(fig)


def test_draw_tokenizer_overview_custom_samples(trained_tokenizer):
    fig = nv.draw_tokenizer_overview(trained_tokenizer, sample_texts=["hi", "the"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ───────────────────────────────────────────────────────────────
# Section 3 — Dataset / sliding windows
# ───────────────────────────────────────────────────────────────

def test_draw_window_view(trained_tokenizer, mock_tokens):
    fig = nv.draw_window_view(trained_tokenizer, mock_tokens,
                              seq_len=16, stride=8, sample_idx=0, n_show=8)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # shift + windows
    plt.close(fig)


def test_draw_window_view_clamps_out_of_range(trained_tokenizer, mock_tokens):
    """A sample_idx past the end of the corpus should clamp safely (no exception)."""
    fig = nv.draw_window_view(trained_tokenizer, mock_tokens,
                              seq_len=16, stride=8, sample_idx=9999, n_show=8)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_draw_window_view_rejects_invalid_args(trained_tokenizer, mock_tokens):
    with pytest.raises(ValueError):
        nv.draw_window_view(trained_tokenizer, [], seq_len=4, stride=2)
    with pytest.raises(ValueError):
        nv.draw_window_view(trained_tokenizer, mock_tokens, seq_len=0, stride=2)
    with pytest.raises(ValueError):
        nv.draw_window_view(trained_tokenizer, mock_tokens, seq_len=4, stride=0)


# ───────────────────────────────────────────────────────────────
# Section 4 — Components
# ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("summary_fn,marker", [
    (nv.rmsnorm_summary, "RMSNorm"),
    (nv.rope_summary, "RotaryPositionEmbedding"),
    (nv.attention_summary, "CausalSelfAttention"),
    (nv.swiglu_summary, "FeedForward"),
])
def test_component_summaries_mention_class(tiny_config, summary_fn, marker):
    out = summary_fn(tiny_config)
    assert marker in out
    assert isinstance(out, str)
    assert len(out.splitlines()) > 3


def test_rmsnorm_summary_reflects_config(tiny_config):
    """The summary must echo the configured d_model so it stays in sync if
    the user re-trains with a different config."""
    out = nv.rmsnorm_summary(tiny_config)
    assert str(tiny_config.d_model) in out


@pytest.mark.parametrize("draw_fn", [
    nv.draw_rmsnorm_dist,
    nv.draw_rope_demo,
    nv.draw_causal_mask,
    nv.draw_swiglu_breakdown,
])
def test_component_figures_render(tiny_config, draw_fn):
    fig = draw_fn(tiny_config)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


# ───────────────────────────────────────────────────────────────
# Section 9 — KV Cache deep dive
# ───────────────────────────────────────────────────────────────

def test_kv_cache_single_step_passes(tiny_model, tiny_config):
    r = nv.kv_cache_single_step(tiny_model, tiny_config, T=8)
    assert r["T"] == 8
    assert r["passed"] is True
    assert r["max_diff"] < 1e-3
    assert r["n_layers"] == tiny_config.n_layers
    assert r["k_shape"][1] == tiny_config.n_heads


def test_kv_cache_multi_step_passes(tiny_model, tiny_config):
    r = nv.kv_cache_multi_step(tiny_model, tiny_config, T=8)
    assert r["T"] == 8
    assert r["passed"] is True
    assert r["max_diff"] < 1e-3


def test_kv_cache_clamps_T_to_max_seq_len(tiny_model, tiny_config):
    """T > max_seq_len should be clamped, not crash."""
    r = nv.kv_cache_single_step(tiny_model, tiny_config,
                                T=tiny_config.max_seq_len + 100)
    assert r["T"] == tiny_config.max_seq_len


def test_format_kv_single_step_shows_pass_or_fail():
    r_pass = {"T": 8, "max_diff": 1e-7, "mean_diff": 1e-8, "passed": True,
              "n_layers": 2, "k_shape": (1, 2, 7, 16), "v_shape": (1, 2, 7, 16)}
    out = nv.format_kv_single_step(r_pass)
    assert "YES" in out
    r_fail = {**r_pass, "max_diff": 1e-1, "passed": False}
    assert "NO" in nv.format_kv_single_step(r_fail)


def test_format_kv_multi_step_shows_pass_or_fail():
    assert "YES" in nv.format_kv_multi_step(
        {"T": 5, "max_diff": 1e-7, "passed": True})
    assert "NO" in nv.format_kv_multi_step(
        {"T": 5, "max_diff": 1.0, "passed": False})


def test_draw_length_sweep_returns_fig_and_summary(tiny_model, tiny_config):
    """Tiny generations on CPU — just verifying the render path, not the speedup."""
    fig, summary = nv.draw_length_sweep(
        tiny_model, tiny_config, prompt_lens=[2, 4], gen_len=2,
    )
    assert isinstance(fig, plt.Figure)
    assert "speedup" in summary
    # One row per prompt length, plus a header + separator
    assert len(summary.splitlines()) == 2 + 2
    plt.close(fig)


def test_draw_length_sweep_clamps_lengths(tiny_model, tiny_config):
    """Prompts longer than max_seq_len - gen_len should be clamped."""
    too_long = tiny_config.max_seq_len + 50
    fig, summary = nv.draw_length_sweep(
        tiny_model, tiny_config, prompt_lens=[too_long], gen_len=2,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
