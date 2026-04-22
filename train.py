"""NanoLLM Training Loop.

Features:
  • bf16 mixed-precision (RTX 4080 supports bf16 natively)
  • AdamW with decoupled weight decay
  • Linear warmup + cosine decay LR schedule
  • Gradient clipping
  • Periodic generation samples (watch the model learn in real time)
  • Checkpoint saving
"""

import os
import sys
import time
import math
import torch
from torch.cuda.amp import autocast, GradScaler

from config import NanoLLMConfig, require_cuda
from tokenizer import BPETokenizer
from model import NanoLLM
from dataset import TextDataset, create_dataloader


def get_lr(step: int, config: NanoLLMConfig, total_steps: int) -> float:
    """Learning rate schedule: linear warmup → cosine decay."""
    # Warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    # Cosine decay phase
    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(decay_steps, 1)
    # Decay to 10% of peak LR
    return config.learning_rate * 0.1 + 0.5 * (config.learning_rate * 0.9) * (
        1 + math.cos(math.pi * progress)
    )


@torch.no_grad()
def evaluate(model, dataloader, device, config, use_amp):
    """Compute average cross-entropy loss on a held-out dataloader.

    Runs in eval mode (dropout off) with no_grad. Uses the same mixed-precision
    dtype as training for a comparable number.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
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
        print("  (matplotlib not installed; skipping loss curve)")
        return

    if not train_history or not epoch_history:
        return

    steps_per_epoch = train_history[-1][0] / max(len(epoch_history), 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Per-step train loss (faint)
    steps = [s for s, _ in train_history]
    losses = [l for _, l in train_history]
    ax.plot(steps, losses, color="#9ecae1", linewidth=0.7,
            label="Train (per step)", alpha=0.6)

    # Per-epoch averages
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
    ax.set_title("NanoLLM training curve", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve saved: {path}")


def train():
    config = NanoLLMConfig()

    # ── Device setup ────────────────────────────────────────
    # Hard-require CUDA: training the full config on CPU takes hours.
    device = require_cuda()
    print("=" * 60)
    print("NanoLLM Training")
    print("=" * 60)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:   {props.name}")
        print(f"VRAM:  {props.total_mem / 1e9:.1f} GB")
        print(f"AMP:   {config.amp_dtype}")
    else:
        print("WARNING: No CUDA device found — training on CPU (will be slow)")
    print()

    # ── Load / prepare data ─────────────────────────────────
    if not os.path.exists(config.data_path):
        print(f"ERROR: Training data not found at {config.data_path}")
        print()
        print("To get started, download a small corpus:")
        print()
        print("  mkdir -p data")
        print("  # Option A: TinyShakespeare (~1MB)")
        print("  wget -q https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O data/corpus.txt")
        print()
        print("  # Option B: Paste your own text into data/corpus.txt")
        sys.exit(1)

    print(f"Reading corpus from {config.data_path}...")
    with open(config.data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"Corpus: {len(raw_text):,} characters")

    # ── Tokenizer ───────────────────────────────────────────
    tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)

    if os.path.exists(config.tokenizer_path):
        tokenizer.load(config.tokenizer_path)
    else:
        print("\nTraining BPE tokenizer...")
        tokenizer.train(raw_text)
        tokenizer.save(config.tokenizer_path)

    # Update config with actual vocab size
    config.vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {config.vocab_size}")

    # ── Tokenise corpus ─────────────────────────────────────
    print("\nTokenising corpus...")
    tokens = tokenizer.encode(raw_text, add_special=False)
    print(f"Tokens: {len(tokens):,} (compression ratio: {len(raw_text)/len(tokens):.2f}x)")

    # ── Train / Val split (sequential 90/10) ────────────────
    # Windows overlap, so a random split would leak. Sequential split keeps
    # the validation tail genuinely held out.
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    if len(val_tokens) < config.max_seq_len + 1:
        raise ValueError(
            f"Val split too small: {len(val_tokens)} tokens < max_seq_len+1 "
            f"({config.max_seq_len+1}). Use a larger corpus or smaller max_seq_len."
        )

    print(f"Train / Val split: {len(train_tokens):,} / {len(val_tokens):,} tokens")

    # ── Datasets & DataLoaders ──────────────────────────────
    train_dataset = TextDataset(train_tokens, config.max_seq_len)
    val_dataset = TextDataset(val_tokens, config.max_seq_len)
    dataloader = create_dataloader(train_dataset, config.batch_size)
    val_dataloader = create_dataloader(val_dataset, config.batch_size, shuffle=False)
    total_steps = len(dataloader) * config.max_epochs

    print(f"Batches per epoch: {len(dataloader)} (train) | {len(val_dataloader)} (val)")
    print(f"Total steps: {total_steps:,}")
    print()

    # ── Model ───────────────────────────────────────────────
    model = NanoLLM(config).to(device)
    print()

    # ── Optimizer: separate weight-decay groups ─────────────
    # Apply weight decay to 2D+ params (weight matrices), NOT to
    # biases, norms, or embeddings — this is standard practice.
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(0.9, 0.95),  # β2=0.95 is standard for LLM training
        eps=1e-8,
    )

    print(f"Optimizer: AdamW (lr={config.learning_rate}, wd={config.weight_decay})")
    print(f"  Decay params:    {sum(p.numel() for p in decay_params):,}")
    print(f"  No-decay params: {sum(p.numel() for p in no_decay_params):,}")

    # ── Mixed precision ─────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp and config.amp_dtype == torch.float16)

    # ── Checkpointing setup ─────────────────────────────────
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # ── Training prompts for periodic generation ────────────
    gen_prompts = [
        "The ",
        "To be or not to be",
        "Once upon a time",
        "What is",
    ]

    # ═══════════════════════════════════════════════════════════
    # Training Loop
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 60)
    print("Starting training...")
    print("=" * 60)

    global_step = 0
    best_val_loss = float("inf")

    # Loss history for the end-of-training curve plot
    train_loss_history = []              # list of (global_step, loss)
    epoch_history = []                   # list of (epoch, train_avg, val_avg)

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t_epoch = time.time()

        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Set learning rate for this step
            lr = get_lr(global_step, config, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward pass (mixed precision)
            with autocast(device_type=device.type, dtype=config.amp_dtype, enabled=use_amp):
                logits, loss = model(input_ids, targets)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm computation)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            # Accumulate stats
            batch_loss = loss.item()
            batch_tokens = input_ids.numel()
            epoch_loss += batch_loss * batch_tokens
            epoch_tokens += batch_tokens
            global_step += 1
            train_loss_history.append((global_step, batch_loss))

            # Logging
            if batch_idx % config.log_interval == 0:
                elapsed = time.time() - t_epoch
                tok_per_sec = epoch_tokens / max(elapsed, 1e-6)
                ppl = math.exp(min(batch_loss, 20))  # Cap to avoid overflow
                print(
                    f"  Epoch {epoch:>2}/{config.max_epochs} │ "
                    f"Step {batch_idx:>4}/{len(dataloader)} │ "
                    f"Loss {batch_loss:.4f} │ PPL {ppl:>8.1f} │ "
                    f"LR {lr:.2e} │ "
                    f"Grad {grad_norm:.2f} │ "
                    f"{tok_per_sec:,.0f} tok/s"
                )

        # ── Epoch summary ───────────────────────────────────
        avg_loss = epoch_loss / max(epoch_tokens, 1)
        avg_ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t_epoch
        tok_per_sec = epoch_tokens / max(elapsed, 1e-6)

        # ── Validation loss ─────────────────────────────────
        val_loss = evaluate(model, val_dataloader, device, config, use_amp)
        val_ppl = math.exp(min(val_loss, 20))
        epoch_history.append((epoch, avg_loss, val_loss))

        print()
        print(f"  ┌─ Epoch {epoch} Summary ───────────────────────────")
        print(f"  │ Train Loss: {avg_loss:.4f}  │  PPL: {avg_ppl:.1f}")
        print(f"  │ Val   Loss: {val_loss:.4f}  │  PPL: {val_ppl:.1f}")
        print(f"  │ Time: {elapsed:.1f}s  │  Throughput: {tok_per_sec:,.0f} tok/s")

        # ── Generate samples ────────────────────────────────
        if epoch % config.eval_interval == 0:
            model.eval()
            print(f"  │")
            print(f"  │ Generation samples (T={config.temperature}, top_k={config.top_k}):")
            for prompt_text in gen_prompts:
                prompt_tokens = tokenizer.encode(prompt_text, add_special=False)
                prompt_tensor = torch.tensor([prompt_tokens], device=device)
                with torch.no_grad():
                    output = model.generate(
                        prompt_tensor,
                        max_new_tokens=60,
                        temperature=config.temperature,
                        top_k=config.top_k,
                        top_p=config.top_p,
                    )
                generated = tokenizer.decode(output[0].tolist())
                # Truncate for display
                display = generated[:200].replace("\n", "↵")
                print(f"  │   \"{display}\"")
            model.train()

        print(f"  └────────────────────────────────────────────────")
        print()

        # ── Checkpoint ──────────────────────────────────────
        if epoch % config.save_interval == 0 or val_loss < best_val_loss:
            ckpt_path = os.path.join(config.checkpoint_dir, f"epoch_{epoch:03d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "val_loss": val_loss,
                    "config": config,
                },
                ckpt_path,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.checkpoint_dir, "best.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "loss": avg_loss,
                        "val_loss": val_loss,
                        "config": config,
                    },
                    best_path,
                )
                print(f"  * New best model saved (val_loss={val_loss:.4f})")

    # ── Final: loss curve PNG ───────────────────────────────
    save_loss_curve(
        train_loss_history,
        epoch_history,
        os.path.join(config.checkpoint_dir, "loss_curve.png"),
    )

    print("=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints:   {config.checkpoint_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    train()
