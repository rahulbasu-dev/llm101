"""NanoLLM Configuration — all hyperparameters in one place."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class NanoLLMConfig:
    # ── Model Architecture ──────────────────────────────────
    vocab_size: int = 0          # Set after tokenizer training
    d_model: int = 384           # Embedding / hidden dimension
    n_layers: int = 6            # Number of transformer blocks
    n_heads: int = 6             # Number of attention heads
    d_ff: int = 1536             # FFN intermediate dim (4 × d_model)
    max_seq_len: int = 256       # Context window
    dropout: float = 0.1         # Dropout rate

    # ── Tokenizer ───────────────────────────────────────────
    target_vocab_size: int = 4096  # BPE merges target (small corpus → smaller vocab)

    # ── Training ────────────────────────────────────────────
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_epochs: int = 15
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_interval: int = 25       # Print every N steps
    eval_interval: int = 1       # Evaluate every N epochs
    save_interval: int = 5       # Checkpoint every N epochs

    # ── Generation ──────────────────────────────────────────
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9

    # ── Paths ───────────────────────────────────────────────
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "tokenizer.json"
    checkpoint_dir: str = "checkpoints"

    # ── Derived ─────────────────────────────────────────────
    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, \
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        return self.d_model // self.n_heads

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
