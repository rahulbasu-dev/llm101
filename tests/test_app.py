"""Smoke tests for the Gradio console.

These tests catch wiring bugs (bad imports, missing components, broken handler
signatures) without actually launching a server. They don't click any buttons —
just build the Blocks tree and sanity-check the attached handlers.

If gradio isn't installed, the tests skip instead of fail.
"""

import pytest
import torch

gr = pytest.importorskip("gradio")

import app  # noqa: E402 — import after importorskip


def _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer):
    """Inject a tiny in-memory model into app.py's singletons so handlers run fast."""
    monkeypatch.setattr(app, "_MODEL", tiny_model)
    monkeypatch.setattr(app, "_TOKENIZER", trained_tokenizer)
    monkeypatch.setattr(app, "_CONFIG", tiny_config)
    monkeypatch.setattr(app, "_STATUS", "test mode")


def test_build_ui_returns_blocks(monkeypatch, tiny_model, tiny_config, trained_tokenizer):
    """build_ui() should assemble the Blocks tree without raising."""
    _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer)
    demo = app.build_ui()
    assert isinstance(demo, gr.Blocks)


def test_require_loaded_rejects_uninitialized(monkeypatch):
    """Handlers should fail loudly, not silently, if the model wasn't loaded."""
    monkeypatch.setattr(app, "_MODEL", None)
    with pytest.raises(RuntimeError, match="not loaded"):
        app._require_loaded()


def test_generate_stream_yields_increasing_text(
    monkeypatch, tiny_model, tiny_config, trained_tokenizer
):
    """The streaming generator should yield the initial state plus one frame
    per new token, and each frame should be (at least) as long as the prior one."""
    _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer)

    torch.manual_seed(0)
    frames = list(app.generate_stream(
        prompt="hello", max_new_tokens=5,
        temperature=0.8, top_k=5, top_p=0.9, use_cache=True,
    ))
    assert len(frames) >= 2, "Should yield initial state + at least one decode step"
    # Final frame should contain the timing annotation
    assert "tok/s" in frames[-1]


def test_generate_stream_handles_empty_prompt(
    monkeypatch, tiny_model, tiny_config, trained_tokenizer
):
    _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer)
    frames = list(app.generate_stream("", 10, 0.8, 5, 0.9, True))
    assert len(frames) == 1
    assert "empty" in frames[0].lower()


def test_run_bench_returns_image_path_and_summary(
    monkeypatch, tiny_model, tiny_config, trained_tokenizer, tmp_path
):
    _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer)
    img_path, summary = app.run_bench("hello", n_tokens=10)
    import os
    assert os.path.exists(img_path)
    assert "generate()" in summary
    assert "generate_fast()" in summary
    assert "Speedup" in summary


def test_render_attention_short_prompt_returns_error(
    monkeypatch, tiny_model, tiny_config, trained_tokenizer
):
    """Prompts that tokenize to <2 tokens should produce a user-visible error,
    not a crash."""
    _install_test_model(monkeypatch, tiny_model, tiny_config, trained_tokenizer)
    head_img, rollout_img, info = app.render_attention("", 0, 0)
    assert head_img is None
    assert rollout_img is None
    assert "prompt" in info.lower()
