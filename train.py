"""LLM101 Training Loop.

Features:
  • bf16 mixed-precision (RTX 4080 supports bf16 natively)
  • AdamW with decoupled weight decay
  • Linear warmup + cosine decay LR schedule
  • Gradient clipping
  • Periodic generation samples (watch the model learn in real time)
  • Checkpoint saving

Architecture:
  train_iter(cfg) is a generator that yields structured events
  (log / step / epoch / best / done / error). Consumers — the CLI
  `train()` wrapper below, and app.py's Train tab — share one training
  implementation.  No duplicated loops; one source of truth.
"""

from __future__ import annotations
import os
import sys
import time
import math
from typing import Iterator

import torch
from torch.amp import autocast, GradScaler  # unified API (torch 2.x) — accepts device_type

from config import NanoLLMConfig, require_cuda
from tokenizer import BPETokenizer
from model import NanoLLM
from dataset import TextDataset, create_dataloader


# ═══════════════════════════════════════════════════════════════
# Helpers (unchanged from prior version)
# ═══════════════════════════════════════════════════════════════

def get_lr(step: int, config: NanoLLMConfig, total_steps: int) -> float:
    """Learning rate schedule: linear warmup → cosine decay."""
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(decay_steps, 1)
    return config.learning_rate * 0.1 + 0.5 * (config.learning_rate * 0.9) * (
        1 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def evaluate(model, dataloader, device, config, use_amp):
    """Average cross-entropy loss on a held-out dataloader (eval mode)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for input_ids, targets in dataloader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(device_type=device.type, dtype=config.amp_dtype, enabled=use_amp):
            _, loss = model(input_ids, targets)
        n = input_ids.numel()
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def save_loss_curve(train_history, epoch_history, path):
    """Render the train/val loss curve to PNG.
    train_history: list of (global_step, train_loss)
    epoch_history: list of (epoch, train_avg, val_avg)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    if not train_history or not epoch_history:
        return False

    steps_per_epoch = train_history[-1][0] / max(len(epoch_history), 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    steps = [s for s, _ in train_history]
    losses = [l for _, l in train_history]
    ax.plot(steps, losses, color="#9ecae1", linewidth=0.7,
            label="Train (per step)", alpha=0.6)

    ep_steps = [e * steps_per_epoch for e, _, _ in epoch_history]
    train_avgs = [t for _, t, _ in epoch_history]
    val_avgs = [v for _, _, v in epoch_history]
    ax.plot(ep_steps, train_avgs, "o-", color="#08519c", linewidth=2,
            markersize=7, label="Train (epoch avg)")
    ax.plot(ep_steps, val_avgs, "s-", color="#cb181d", linewidth=2,
            markersize=7, label="Validation")

    ax.set_xlabel("Global step", fontsize=11)
    ax.set_ylabel("Cross-entropy loss (log scale)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title("LLM101 training curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


# ═══════════════════════════════════════════════════════════════
# Core: training as a generator of events
# ═══════════════════════════════════════════════════════════════

def train_iter(config: NanoLLMConfig | None = None) -> Iterator[dict]:
    """Training loop as an event generator.

    Yields dicts with a "type" field:
      {"type": "log",   "msg": str}                                  — freeform text
      {"type": "step",  "global_step", "epoch", "loss", "lr",
                        "grad_norm", "tps"}                          — per-batch
      {"type": "epoch", "epoch", "train_loss", "val_loss",
                        "train_ppl", "val_ppl", "elapsed",
                        "samples": list[str]}                        — per-epoch
      {"type": "best",  "epoch", "val_loss", "path"}                 — new best val
      {"type": "done",  "best_val_loss", "curve_path"}               — training over
      {"type": "error", "msg"}                                       — setup failed

    The CLI wrapper `train()` below consumes these with print();
    the Gradio Train tab consumes the same events into log/plot widgets.
    """
    if config is None:
        config = NanoLLMConfig()

    device = require_cuda()

    yield {"type": "log", "msg": "=" * 60}
    yield {"type": "log", "msg": "LLM101 Training"}
    yield {"type": "log", "msg": "=" * 60}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        yield {"type": "log", "msg": f"GPU:   {props.name}"}
        yield {"type": "log", "msg": f"VRAM:  {props.total_memory / 1e9:.1f} GB"}
        yield {"type": "log", "msg": f"AMP:   {config.amp_dtype}"}
    else:
        yield {"type": "log", "msg": "WARNING: Training on CPU — will be slow."}
    yield {"type": "log", "msg": ""}

    # ── Corpus ──
    if not os.path.exists(config.data_path):
        yield {"type": "error",
               "msg": f"Training data not found at {config.data_path}. "
                      f"Run: bash run.sh setup  (downloads TinyShakespeare)"}
        return

    yield {"type": "log", "msg": f"Reading corpus from {config.data_path}..."}
    with open(config.data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    yield {"type": "log", "msg": f"Corpus: {len(raw_text):,} characters"}

    # ── Tokenizer ──
    tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
    if os.path.exists(config.tokenizer_path):
        tokenizer.load(config.tokenizer_path)
        yield {"type": "log", "msg": f"Loaded tokenizer from {config.tokenizer_path}"}
    else:
        yield {"type": "log", "msg": "Training BPE tokenizer (this is the slow step)..."}
        tokenizer.train(raw_text)
        tokenizer.save(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size
    yield {"type": "log", "msg": f"Vocab size: {config.vocab_size}"}

    # ── Tokenise & split ──
    yield {"type": "log", "msg": "Tokenising corpus..."}
    tokens = tokenizer.encode(raw_text, add_special=False)
    yield {"type": "log",
           "msg": f"Tokens: {len(tokens):,} "
                  f"(compression ratio: {len(raw_text)/len(tokens):.2f}x)"}

    split = int(0.9 * len(tokens))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    if len(val_tokens) < config.max_seq_len + 1:
        yield {"type": "error",
               "msg": f"Val split too small: {len(val_tokens)} tokens. "
                      f"Use larger corpus or smaller max_seq_len."}
        return
    yield {"type": "log",
           "msg": f"Train / Val split: {len(train_tokens):,} / {len(val_tokens):,} tokens"}

    # ── Datasets ──
    train_dataset = TextDataset(train_tokens, config.max_seq_len)
    val_dataset = TextDataset(val_tokens, config.max_seq_len)
    dataloader = create_dataloader(train_dataset, config.batch_size)
    val_dataloader = create_dataloader(val_dataset, config.batch_size, shuffle=False)
    total_steps = len(dataloader) * config.max_epochs
    yield {"type": "log",
           "msg": f"Batches per epoch: {len(dataloader)} (train) | "
                  f"{len(val_dataloader)} (val)  ·  Total steps: {total_steps:,}"}

    # ── Model & optimizer ──
    model = NanoLLM(config).to(device)
    decay_params, no_decay_params = [], []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay_params if param.dim() >= 2 else no_decay_params).append(param)

    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": config.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8,
    )
    yield {"type": "log",
           "msg": f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})  "
                  f"· decay={sum(p.numel() for p in decay_params):,}  "
                  f"no-decay={sum(p.numel() for p in no_decay_params):,}"}

    use_amp = device.type == "cuda"
    # GradScaler only needed for fp16 (bf16 has enough dynamic range to skip scaling).
    scaler = GradScaler(device.type, enabled=use_amp and config.amp_dtype == torch.float16)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    gen_prompts = ["The ", "To be or not to be", "Once upon a time", "What is"]

    yield {"type": "log", "msg": ""}
    yield {"type": "log", "msg": "=" * 60}
    yield {"type": "log", "msg": "Starting training..."}
    yield {"type": "log", "msg": "=" * 60}

    # ── Training loop ──
    global_step = 0
    best_val_loss = float("inf")
    train_loss_history = []      # (global_step, loss)
    epoch_history = []           # (epoch, train_avg, val_avg)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            lr = get_lr(global_step, config, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with autocast(device_type=device.type, dtype=config.amp_dtype, enabled=use_amp):
                _, loss = model(input_ids, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            batch_loss = loss.item()
            batch_tokens = input_ids.numel()
            epoch_loss += batch_loss * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1
            train_loss_history.append((global_step, batch_loss))

            elapsed = time.time() - t_epoch
            tps = epoch_tokens / max(elapsed, 1e-6)

            yield {"type": "step",
                   "global_step": global_step, "epoch": epoch,
                   "batch_idx": batch_idx, "total_batches": len(dataloader),
                   "loss": batch_loss, "lr": lr,
                   "grad_norm": float(grad_norm),
                   "tps": tps}

        # ── Epoch summary ──
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        avg_ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t_epoch
        val_loss = evaluate(model, val_dataloader, device, config, use_amp)
        val_ppl = math.exp(min(val_loss, 20))
        epoch_history.append((epoch, avg_loss, val_loss))

        # ── Generation samples ──
        samples = []
        if epoch % config.eval_interval == 0:
            model.eval()
            for prompt_text in gen_prompts:
                prompt_tokens = tokenizer.encode(prompt_text, add_special=False)
                prompt_tensor = torch.tensor([prompt_tokens], device=device)
                with torch.no_grad():
                    output = model.generate(
                        prompt_tensor, max_new_tokens=60,
                        temperature=config.temperature,
                        top_k=config.top_k, top_p=config.top_p,
                    )
                generated = tokenizer.decode(output[0].tolist())[:200]
                samples.append(generated)
            model.train()

        yield {"type": "epoch",
               "epoch": epoch, "max_epochs": config.max_epochs,
               "train_loss": avg_loss, "val_loss": val_loss,
               "train_ppl": avg_ppl, "val_ppl": val_ppl,
               "elapsed": elapsed,
               "samples": samples}

        # ── Checkpoint ──
        if epoch % config.save_interval == 0 or val_loss < best_val_loss:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch, "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss, "val_loss": val_loss, "config": config,
            }, ckpt_path)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, "best.pt")
                torch.save({
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss, "val_loss": val_loss, "config": config,
                }, best_path)
                yield {"type": "best", "epoch": epoch,
                       "val_loss": val_loss, "path": best_path}

    # ── Loss curve & done ──
    curve_path = os.path.join(config.checkpoint_dir, "loss_curve.png")
    ok = save_loss_curve(train_loss_history, epoch_history, curve_path)
    yield {"type": "done",
           "best_val_loss": best_val_loss,
           "curve_path": curve_path if ok else None}


# ═══════════════════════════════════════════════════════════════
# CLI wrapper — consumes train_iter, formats to stdout
# ═══════════════════════════════════════════════════════════════

def train():
    """Command-line training entrypoint. Formats train_iter events to stdout."""
    cfg = NanoLLMConfig()
    log_interval = cfg.log_interval

    for evt in train_iter(cfg):
        t = evt["type"]

        if t == "log":
            print(evt["msg"])

        elif t == "step":
            if evt["batch_idx"] % log_interval == 0:
                ppl = math.exp(min(evt["loss"], 20))
                print(
                    f"  Epoch {evt['epoch']:>2}/{cfg.max_epochs} | "
                    f"Step {evt['batch_idx']:>4}/{evt['total_batches']} | "
                    f"Loss {evt['loss']:.4f} | PPL {ppl:>8.1f} | "
                    f"LR {evt['lr']:.2e} | "
                    f"Grad {evt['grad_norm']:.2f} | "
                    f"{evt['tps']:,.0f} tok/s"
                )

        elif t == "epoch":
            print()
            print(f"  +- Epoch {evt['epoch']} Summary -----------------")
            print(f"  | Train Loss: {evt['train_loss']:.4f}  |  PPL: {evt['train_ppl']:.1f}")
            print(f"  | Val   Loss: {evt['val_loss']:.4f}  |  PPL: {evt['val_ppl']:.1f}")
            print(f"  | Time: {evt['elapsed']:.1f}s")
            if evt["samples"]:
                print(f"  | Generation samples:")
                for s in evt["samples"]:
                    disp = s.replace("\n", " / ")
                    print(f"  |   \"{disp}\"")
            print(f"  +-------------------------------------------------")
            print()

        elif t == "best":
            print(f"  * New best model saved (val_loss={evt['val_loss']:.4f}) -> {evt['path']}")

        elif t == "done":
            print("=" * 60)
            print("Training complete!")
            print(f"Best val loss: {evt['best_val_loss']:.4f}")
            if evt["curve_path"]:
                print(f"Loss curve:    {evt['curve_path']}")
            print("=" * 60)

        elif t == "error":
            print(f"ERROR: {evt['msg']}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    train()
