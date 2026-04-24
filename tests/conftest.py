"""Shared fixtures for NanoLLM tests.

Design goals:
  - Zero external data (no TinyStories download required).
  - CPU-only (tests run anywhere).
  - Tiny everything: d_model=32, n_layers=2, seq_len=16, vocab~300.
    Full test suite completes in seconds.
"""

from __future__ import annotations
import os
import sys
import random

import pytest
import torch
import numpy as np

# Make project root importable so `from model import ...` works from tests/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import NanoLLMConfig
from tokenizer import BPETokenizer, NUM_BASE
from dataset import TextDataset
from model import NanoLLM


@pytest.fixture(autouse=True)
def _seed_everything(monkeypatch):
    """Autouse: every test starts from the same RNG state so results are
    reproducible. Tests that need a different seed can call torch.manual_seed
    again inside the test body.

    Also sets NANOLLM_ALLOW_CPU=1 so any test that accidentally triggers
    require_cuda() doesn't sys.exit — tests stay CPU-portable. Individual
    tests can monkeypatch this env away to exercise the strict gate.
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    monkeypatch.setenv("NANOLLM_ALLOW_CPU", "1")
    yield


@pytest.fixture
def mock_corpus() -> str:
    """Deterministic mock corpus — enough variety to train a small BPE and
    provide ~2000 tokens after encoding, but fully self-contained."""
    # Two reasonably varied sentences repeated — gives BPE real merges to find.
    a = "the quick brown fox jumps over the lazy dog. "
    b = "to be or not to be, that is the question. "
    c = "all animals are equal, but some are more equal than others. "
    corpus = (a + b + c) * 40  # ~5 KB
    return corpus


@pytest.fixture
def trained_tokenizer(mock_corpus) -> BPETokenizer:
    """BPE tokenizer trained on the mock corpus with a small target vocab."""
    tok = BPETokenizer(target_vocab_size=320)  # 260 base + 60 merges
    tok.train(mock_corpus)
    return tok


@pytest.fixture
def mock_tokens(trained_tokenizer, mock_corpus) -> list[int]:
    """Full corpus encoded as token IDs (no BOS/EOS)."""
    return trained_tokenizer.encode(mock_corpus, add_special=False)


@pytest.fixture
def tiny_config(trained_tokenizer) -> NanoLLMConfig:
    """Tiny config sized for fast tests."""
    return NanoLLMConfig(
        vocab_size=trained_tokenizer.vocab_size,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,  # deterministic tests
    )


@pytest.fixture
def tiny_model(tiny_config) -> NanoLLM:
    """A NanoLLM instance with tiny_config, in eval mode (no dropout)."""
    model = NanoLLM(tiny_config)
    model.eval()
    return model
