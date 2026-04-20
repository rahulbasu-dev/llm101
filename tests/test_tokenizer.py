"""Tests for the byte-level BPE tokenizer."""

import pytest
from tokenizer import BPETokenizer, NUM_BASE, PAD_ID, BOS_ID, EOS_ID, UNK_ID


def test_base_vocab_size_is_260():
    """4 specials + 256 bytes = 260 base tokens before any merges."""
    tok = BPETokenizer(target_vocab_size=500)
    assert len(tok.vocab) == NUM_BASE == 260


def test_special_token_ids():
    assert PAD_ID == 0
    assert BOS_ID == 1
    assert EOS_ID == 2
    assert UNK_ID == 3


def test_train_reaches_target_size(mock_corpus):
    tok = BPETokenizer(target_vocab_size=320)
    actual = tok.train(mock_corpus)
    assert actual == 320
    assert len(tok.merges) == 60  # 320 - 260 base


def test_roundtrip_ascii(trained_tokenizer):
    """Encoding then decoding should recover the original ASCII text."""
    text = "the quick brown fox"
    ids = trained_tokenizer.encode(text, add_special=False)
    recovered = trained_tokenizer.decode(ids)
    assert recovered == text


def test_roundtrip_utf8(trained_tokenizer):
    """Byte-level BPE must handle any UTF-8 text."""
    text = "caf\u00e9 r\u00e9sum\u00e9"  # café résumé
    ids = trained_tokenizer.encode(text, add_special=False)
    recovered = trained_tokenizer.decode(ids)
    assert recovered == text


def test_add_special_wraps_with_bos_eos(trained_tokenizer):
    ids = trained_tokenizer.encode("hi", add_special=True)
    assert ids[0] == BOS_ID
    assert ids[-1] == EOS_ID


def test_decode_skips_special_tokens(trained_tokenizer):
    """Specials should not appear in decoded text."""
    ids = trained_tokenizer.encode("hello", add_special=True)
    # Confirm BOS/EOS are in the list we're decoding
    assert BOS_ID in ids and EOS_ID in ids
    decoded = trained_tokenizer.decode(ids)
    assert decoded == "hello"


def test_merges_compress(trained_tokenizer, mock_corpus):
    """After training, encoding should produce FEWER tokens than raw bytes."""
    n_bytes = len(mock_corpus.encode("utf-8"))
    n_tokens = len(trained_tokenizer.encode(mock_corpus, add_special=False))
    assert n_tokens < n_bytes, "BPE should compress vs raw bytes"


def test_save_load_roundtrip(tmp_path, trained_tokenizer):
    """Saving and loading should yield identical encoding."""
    path = tmp_path / "tok.json"
    trained_tokenizer.save(str(path))

    reloaded = BPETokenizer(target_vocab_size=trained_tokenizer.target_vocab_size)
    reloaded.load(str(path))

    # Same vocab size
    assert reloaded.vocab_size == trained_tokenizer.vocab_size
    # Same merges (tuples compare by value)
    assert reloaded.merges == trained_tokenizer.merges
    # Same encoding for an arbitrary test string
    text = "sample roundtrip text for identity check"
    assert reloaded.encode(text) == trained_tokenizer.encode(text)


def test_decode_token_renders_specials():
    tok = BPETokenizer(target_vocab_size=300)
    assert tok.decode_token(PAD_ID) == "<PAD>"
    assert tok.decode_token(BOS_ID) == "<BOS>"
    assert tok.decode_token(EOS_ID) == "<EOS>"
    assert tok.decode_token(UNK_ID) == "<UNK>"


def test_unknown_id_renders_safely():
    """decode_token should not crash on IDs outside the vocab."""
    tok = BPETokenizer(target_vocab_size=300)
    assert tok.decode_token(999_999).startswith("<")


def test_empty_string(trained_tokenizer):
    ids = trained_tokenizer.encode("", add_special=False)
    assert ids == []
    assert trained_tokenizer.decode([]) == ""
