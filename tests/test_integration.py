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


def test_train_iter_auto_scales_warmup_for_short_runs(tmp_path, mock_corpus):
    """When warmup_steps would eat > 25% of total_steps (common with short
    demo runs), train_iter should auto-reduce it and emit a visible log
    message. Conversely, if _warmup_explicit is set, respect the user's value.
    """
    from train import train_iter

    (tmp_path / "data").mkdir()
    corpus_path = tmp_path / "data" / "corpus.txt"
    corpus_path.write_text(mock_corpus * 3)
    ckpt_dir = tmp_path / "checkpoints"

    # Tiny run: ~3 batches/epoch × 1 epoch = ~3 total_steps.
    # warmup_steps=200 is absurd → should auto-scale down.
    cfg = NanoLLMConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
        max_epochs=1, log_interval=1, save_interval=1,
        target_vocab_size=300, batch_size=64,
        warmup_steps=200,  # ← the mistuned value
        data_path=str(corpus_path),
        tokenizer_path=str(tmp_path / "tok.json"),
        checkpoint_dir=str(ckpt_dir),
    )

    events = list(train_iter(cfg))
    # Auto-scale should have fired and logged a notice
    notices = [e for e in events if e["type"] == "log"
               and "auto-reducing" in e["msg"].lower()]
    assert notices, "train_iter should auto-reduce over-long warmup and log it"
    # Final warmup_steps should be smaller than the original 200
    assert cfg.warmup_steps < 200
    assert cfg.warmup_steps >= 20  # minimum floor respected

    # ── Same run but with _warmup_explicit → auto-scale MUST NOT fire ──
    cfg2 = NanoLLMConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
        max_epochs=1, log_interval=1, save_interval=1,
        target_vocab_size=300, batch_size=64,
        warmup_steps=200,
        data_path=str(corpus_path),
        tokenizer_path=str(tmp_path / "tok2.json"),
        checkpoint_dir=str(tmp_path / "ckpt2"),
    )
    setattr(cfg2, "_warmup_explicit", True)
    events2 = list(train_iter(cfg2))
    notices2 = [e for e in events2 if e["type"] == "log"
                and "auto-reducing" in e["msg"].lower()]
    assert not notices2, "auto-scale must be skipped when _warmup_explicit is set"
    assert cfg2.warmup_steps == 200  # value preserved


def test_train_iter_yields_expected_event_types(tmp_path, mock_corpus, monkeypatch):
    """A tiny end-to-end training run through train_iter() must yield at least
    one of every event type the UI and CLI depend on. This pins the generator
    contract so refactors don't silently break Train-tab streaming.
    """
    from train import train_iter

    # Stand up a miniature corpus + env on tmp_path
    (tmp_path / "data").mkdir()
    corpus_path = tmp_path / "data" / "corpus.txt"
    corpus_path.write_text(mock_corpus * 3)

    tok_path = tmp_path / "tokenizer.json"
    ckpt_dir = tmp_path / "checkpoints"

    cfg = NanoLLMConfig(
        d_model=32, n_layers=2, n_heads=2, d_ff=64,
        max_seq_len=16, dropout=0.0,
        max_epochs=1, warmup_steps=2, log_interval=1, save_interval=1,
        target_vocab_size=300, batch_size=4,
        data_path=str(corpus_path),
        tokenizer_path=str(tok_path),
        checkpoint_dir=str(ckpt_dir),
    )

    # Collect all events
    events = list(train_iter(cfg))
    types = {e["type"] for e in events}

    # Every event type the UI depends on must appear at least once
    required = {"log", "step", "epoch", "done"}
    missing = required - types
    assert not missing, f"train_iter did not yield: {missing}. Got: {types}"

    # Event shape checks (Train tab will break if these drift)
    step_evts = [e for e in events if e["type"] == "step"]
    assert step_evts, "need at least one step event"
    s = step_evts[0]
    assert {"global_step", "epoch", "batch_idx", "total_batches",
            "loss", "lr", "grad_norm", "tps"} <= s.keys()

    epoch_evts = [e for e in events if e["type"] == "epoch"]
    assert len(epoch_evts) == 1, "should have exactly one epoch event for max_epochs=1"
    e = epoch_evts[0]
    assert {"epoch", "max_epochs", "train_loss", "val_loss",
            "train_ppl", "val_ppl", "elapsed", "samples"} <= e.keys()

    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1 and "best_val_loss" in done[0]


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
