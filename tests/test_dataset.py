"""Tests for TextDataset — sliding-window causal-LM sample builder."""

import torch
import pytest

from dataset import TextDataset


def test_sample_shapes(mock_tokens):
    seq_len = 16
    ds = TextDataset(mock_tokens, seq_len=seq_len)
    inp, tgt = ds[0]
    assert inp.shape == (seq_len,)
    assert tgt.shape == (seq_len,)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long


def test_target_is_input_shifted_by_one(mock_tokens):
    """The causal-LM invariant: target[t] = input[t+1] (within a window)."""
    ds = TextDataset(mock_tokens, seq_len=16)
    inp, tgt = ds[3]
    # target[0..-2] should equal input[1..-1]
    assert torch.equal(tgt[:-1], inp[1:])


def test_sample_count_matches_stride(mock_tokens):
    """With stride=seq_len, the number of samples should be
    floor((n - seq_len - 1) / stride) + 1.
    """
    seq_len = 16
    stride = 16
    ds = TextDataset(mock_tokens, seq_len=seq_len, stride=stride)
    expected = (len(mock_tokens) - seq_len - 1) // stride + 1
    assert len(ds) == expected


def test_default_stride_is_half_seq_len(mock_tokens):
    """Default stride = seq_len // 2 → roughly 2x more samples than stride=seq_len."""
    seq_len = 16
    ds_default = TextDataset(mock_tokens, seq_len=seq_len)
    ds_nonoverlap = TextDataset(mock_tokens, seq_len=seq_len, stride=seq_len)
    assert len(ds_default) > len(ds_nonoverlap)


def test_too_few_tokens_returns_empty():
    """A corpus that's too short to fit even one window should produce 0 samples."""
    ds = TextDataset([1, 2, 3], seq_len=16)
    assert len(ds) == 0
