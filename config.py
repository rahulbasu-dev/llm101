"""NanoLLM Configuration — all hyperparameters in one place."""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch


# ═══════════════════════════════════════════════════════════════
# Matplotlib font fix — WSL2 often has no system fonts, which
# crashes the mathtext parser when measuring text for layout.
# We nuke the font cache, force DejaVu Sans, and provide a
# safe_savefig() that falls back when bbox_inches="tight" fails.
# ═══════════════════════════════════════════════════════════════

def _fix_matplotlib_fonts():
    """One-time matplotlib font configuration. Safe to call multiple times."""
    try:
        import matplotlib
        # Delete stale font cache so matplotlib discovers its bundled fonts
        import pathlib
        cache_dir = pathlib.Path(matplotlib.get_cachedir())
        for f in cache_dir.glob("fontlist-*.json"):
            f.unlink(missing_ok=True)
        # Force bundled DejaVu Sans
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["mathtext.fontset"] = "dejavusans"
        # Rebuild from scratch
        matplotlib.font_manager._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

_fix_matplotlib_fonts()


def safe_savefig(path, *, dpi=140, **kwargs):
    """Save the current matplotlib figure, falling back if bbox_inches fails.

    bbox_inches='tight' triggers text measurement which crashes on WSL2
    when fonts are missing. This tries tight first, then falls back to
    a plain save.
    """
    import matplotlib.pyplot as plt
    try:
        plt.savefig(path, dpi=dpi, bbox_inches="tight", **kwargs)
    except (ValueError, Exception):
        try:
            plt.savefig(path, dpi=dpi, **kwargs)
        except Exception:
            pass


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
        # Honour NANOLLM_ALLOW_CPU=1 so tests (and CPU-only demos) stay on CPU
        # even when running on a CUDA-capable machine. Without this, fixtures
        # built on CPU collide with handler tensors created on cuda:0.
        if os.environ.get("NANOLLM_ALLOW_CPU") == "1":
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp_dtype(self) -> torch.dtype:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
