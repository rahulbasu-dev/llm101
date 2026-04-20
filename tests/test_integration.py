"""End-to-end integration smoke tests.

Runs the full pipeline on mock data:
  corpus -> BPE -> dataset -> model -> N training steps -> generate -> checkpoint roundtrip

These are slower (~10s total) but catch wiring bugs that unit tests miss.
"""

import torch
import pytest
from torch.utils.data import DataLoader

from config import NanoLLMConfig
from tokenizer import BPETokenizer
from dataset import TextDataset
from model import NanoLLM


def test_training_step_reduces_loss_on_tiny_overfit(tiny_model, tiny_config, mock_tokens):
    """Overfit a single batch for ~50 steps. Loss should drop meaningfully.

    If it doesn't, the optimizer / autograd / loss hookup is broken.
    """
    # Use train mode (even though fixture returns eval) for this test
    tiny_model.train()

    # One fixed batch
    dataset = TextDataset(mock_tokens, seq_len=tiny_config.max_seq_len)
    assert len(dataset) > 0
    inp, tgt = dataset[0]
    inp = inp.unsqueeze(0)
    tgt = tgt.unsqueeze(0)

    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=3e-3)

    # Initial loss
    with torch.no_grad():
        _, loss0 = tiny_model(inp, tgt)
    loss0 = loss0.item()

    # 50 training steps on the same batch
    for _ in range(50):
        _, loss = tiny_model(inp, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_final = loss.item()
    assert loss_final < loss0 * 0.5, \
        f"Overfit test failed: loss went {loss0:.3f} -> {loss_final:.3f} (expected much lower)"


def test_full_pipeline_mock_corpus(tmp_path, mock_corpus):
    """Build tokenizer + dataset + model + dataloader + one training batch.

    Touches every component through its public interface. Uses the shared
    mock_corpus fixture (varied enough that BPE can't collapse it into a
    single token).
    """
    # ── 1. Tokenizer ──
    tok = BPETokenizer(target_vocab_size=320)
    tok.train(mock_corpus)
    assert tok.vocab_size == 320

    # ── 2. Tokenize ──
    tokens = tok.encode(mock_corpus, add_special=False)
    assert len(tokens) > 100

    # ── 4. Dataset + DataLoader ──
    ds = TextDataset(tokens, seq_len=16)
    assert len(ds) > 0
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(dl))
    assert batch[0].shape[0] == 4

    # ── 5. Model ──
    cfg = NanoLLMConfig(
        vocab_size=tok.vocab_size,
        d_model=32, n_layers=2, n_heads=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
    )
    model = NanoLLM(cfg)
    model.train()

    # ── 6. One training step ──
    inp, tgt = batch
    _, loss = model(inp, tgt)
    assert torch.isfinite(loss)
    loss.backward()

    # ── 7. Generation on the trained (1-step) model ──
    model.eval()
    prompt = inp[:1, :4]
    out = model.generate_fast(prompt, max_new_tokens=5)
    assert out.size(1) == 9  # 4 + 5
    assert (out >= 0).all() and (out < cfg.vocab_size).all()


def test_checkpoint_roundtrip(tmp_path, tiny_model, tiny_config):
    """Save a model to disk, load it, and verify weights match."""
    ckpt_path = tmp_path / "test.pt"

    # Save
    torch.save({
        "model_state_dict": tiny_model.state_dict(),
        "config": tiny_config,
    }, ckpt_path)

    # Load into a fresh model
    ckpt = torch.load(ckpt_path, weights_only=False)
    fresh = NanoLLM(ckpt["config"])
    fresh.load_state_dict(ckpt["model_state_dict"])

    # Weight tying must survive the roundtrip
    assert fresh.lm_head.weight is fresh.token_emb.weight

    # Every param should match exactly
    for (n1, p1), (n2, p2) in zip(tiny_model.named_parameters(),
                                   fresh.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"Mismatch in {n1}"

    # And produce identical outputs
    tiny_model.eval()
    fresh.eval()
    idx = torch.randint(0, tiny_config.vocab_size, (1, 8))
    with torch.no_grad():
        out1, _ = tiny_model(idx)
        out2, _ = fresh(idx)
    assert torch.equal(out1, out2)


def test_tokenizer_save_then_train_from_loaded(tmp_path, mock_corpus):
    """A loaded tokenizer should encode identically to the original."""
    tok1 = BPETokenizer(target_vocab_size=300)
    tok1.train(mock_corpus)
    path = tmp_path / "tok.json"
    tok1.save(str(path))

    tok2 = BPETokenizer(target_vocab_size=300)
    tok2.load(str(path))

    text = "test encoding consistency after save/load"
    assert tok1.encode(text) == tok2.encode(text)
    assert tok1.decode(tok1.encode(text)) == tok2.decode(tok2.encode(text))
