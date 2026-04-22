"""Tests for NanoLLMConfig — property derivations and validation."""

import os
import pytest
import torch

from config import NanoLLMConfig, require_cuda


def test_d_head_divides_evenly():
    cfg = NanoLLMConfig(d_model=384, n_heads=6)
    assert cfg.d_head == 64


def test_d_head_asserts_on_mismatch():
    cfg = NanoLLMConfig(d_model=100, n_heads=6)  # 100 % 6 != 0
    with pytest.raises(AssertionError, match="must be divisible"):
        _ = cfg.d_head


def test_device_is_torch_device():
    cfg = NanoLLMConfig()
    assert isinstance(cfg.device, torch.device)
    assert cfg.device.type in ("cuda", "cpu")


def test_amp_dtype_cpu_fallback():
    """On CPU, amp_dtype falls back to fp16 per the current implementation.
    (On bf16-capable CUDA it returns bfloat16.)"""
    cfg = NanoLLMConfig()
    dtype = cfg.amp_dtype
    assert dtype in (torch.bfloat16, torch.float16)


def test_defaults_stable():
    """Guard against accidental breakage of the documented defaults."""
    cfg = NanoLLMConfig()
    assert cfg.d_model == 384
    assert cfg.n_layers == 6
    assert cfg.n_heads == 6
    assert cfg.max_seq_len == 256
    assert cfg.target_vocab_size == 4096


def test_require_cuda_honors_allow_cpu_env(monkeypatch):
    """NANOLLM_ALLOW_CPU=1 should let CPU runs through (for CI / this test)."""
    monkeypatch.setenv("NANOLLM_ALLOW_CPU", "1")
    device = require_cuda()
    assert isinstance(device, torch.device)
    # On a CPU-only test environment, we expect CPU; on a GPU CI, we expect CUDA
    # — either is fine, the point is it doesn't sys.exit.
    assert device.type in ("cuda", "cpu")


def test_require_cuda_exits_without_cuda(monkeypatch):
    """Without the escape hatch, require_cuda must sys.exit on CPU-only envs."""
    monkeypatch.delenv("NANOLLM_ALLOW_CPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exc_info:
        require_cuda()
    assert exc_info.value.code == 2
