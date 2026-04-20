"""Tests for NanoLLMConfig — property derivations and validation."""

import pytest
import torch

from config import NanoLLMConfig


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
