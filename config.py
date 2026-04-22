"""NanoLLM Configuration — all hyperparameters in one place."""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch


# ═══════════════════════════════════════════════════════════════
# CUDA gate — the application entry points must run on CUDA.
# Tests intentionally do not call this (they live on CPU for portability).
# Set NANOLLM_ALLOW_CPU=1 to force-enable CPU mode (e.g. CI runners).
# ═══════════════════════════════════════════════════════════════

def require_cuda() -> torch.device:
    """Ensure a CUDA device is available; return it. Otherwise exit loudly.

    Called by every user-facing script (`train.py`, `generate.py`, `teach.py`,
    `visualise.py`, `app.py`). NOT called by the test suite.

    Bypass for CI or CPU-only demos: set `NANOLLM_ALLOW_CPU=1`.
    """
    if os.environ.get("NANOLLM_ALLOW_CPU") == "1":
        print(" NANOLLM_ALLOW_CPU=1 set — running on CPU. Training will be SLOW.",
              file=sys.stderr)
        return torch.device("cpu")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        return dev

    # No CUDA. Print a big, helpful error and exit.
    print("", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    print(" NanoLLM requires CUDA. No CUDA device was detected.", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    print(f" Installed torch:  {torch.__version__}", file=sys.stderr)
    build = "cpu-only" if "+cpu" in torch.__version__ else "unknown"
    print(f" Torch build:      {build}", file=sys.stderr)
    print("", file=sys.stderr)
    print(" To install the CUDA build of PyTorch (cu121 — works with any", file=sys.stderr)
    print(" RTX 3xxx/4xxx/5xxx GPU):", file=sys.stderr)
    print("", file=sys.stderr)
    print("   pip install --upgrade torch torchvision torchaudio \\", file=sys.stderr)
    print("       --index-url https://download.pytorch.org/whl/cu121", file=sys.stderr)
    print("", file=sys.stderr)
    print(" Or re-run the project setup:", file=sys.stderr)
    print("", file=sys.stderr)
    print("   bash run.sh setup", file=sys.stderr)
    print("", file=sys.stderr)
    print(" If you REALLY want to run on CPU (training will take hours):", file=sys.stderr)
    print("", file=sys.stderr)
    print("   NANOLLM_ALLOW_CPU=1 bash run.sh ui", file=sys.stderr)
    print("", file=sys.stderr)
    print("=" * 68, file=sys.stderr)
    sys.exit(2)


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
