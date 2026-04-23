"""Tests for visualize_anim -- tensor collection for the animated visualization."""

import torch
import pytest

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from model import NanoLLM
from visualize_anim import collect_viz_data


def test_collect_viz_data_structure(tiny_model, trained_tokenizer, tiny_config):
    """collect_viz_data returns the expected JSON-serializable structure."""
    data = collect_viz_data(tiny_model, trained_tokenizer, "The cat sat", tiny_config)

    assert isinstance(data, dict)
    assert data["n_layers"] == tiny_config.n_layers  # 2
    assert data["n_heads"] == tiny_config.n_heads    # 2
    assert data["d_model"] == tiny_config.d_model    # 32
    assert data["d_head"] == tiny_config.d_head      # 16
    assert isinstance(data["tokens"], list)
    assert len(data["tokens"]) > 0
    assert isinstance(data["layers"], list)
    assert len(data["layers"]) == tiny_config.n_layers


def test_collect_viz_data_layer_contents(tiny_model, trained_tokenizer, tiny_config):
    """Each layer entry has attention weights, hidden norm, and shapes."""
    data = collect_viz_data(tiny_model, trained_tokenizer, "The cat sat", tiny_config)
    T = len(data["tokens"])

    for i, layer in enumerate(data["layers"]):
        assert layer["layer_idx"] == i

        # attn_weights: [n_heads][T][T], values in [0, 1]
        aw = layer["attn_weights"]
        assert len(aw) == tiny_config.n_heads
        assert len(aw[0]) == T
        assert len(aw[0][0]) == T
        # Check post-softmax range
        assert all(0.0 <= v <= 1.0 for row in aw[0] for v in row)

        # hidden_norm: positive scalar
        assert isinstance(layer["hidden_norm"], float)
        assert layer["hidden_norm"] > 0

        # shapes: dict with expected keys
        shapes = layer["shapes"]
        assert "input" in shapes
        assert "qkv" in shapes
        assert "q" in shapes
        assert "k" in shapes
        assert "v" in shapes
        assert "attn_out" in shapes
        assert "ffn_out" in shapes
        assert "output" in shapes


def test_collect_viz_data_is_json_serializable(tiny_model, trained_tokenizer, tiny_config):
    """The returned dict must be JSON-serializable (no tensors, no numpy)."""
    import json
    data = collect_viz_data(tiny_model, trained_tokenizer, "hello", tiny_config)
    # This will raise TypeError if any value is a tensor/ndarray
    json_str = json.dumps(data)
    assert len(json_str) > 100
