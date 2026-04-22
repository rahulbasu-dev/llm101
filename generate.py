"""Interactive text generation with a trained LLM101 checkpoint.

Usage:
    python generate.py                          # Interactive, no KV cache
    python generate.py --fast                   # Interactive, with KV cache
    python generate.py --benchmark              # Compare generate() vs generate_fast()
    python generate.py --checkpoint checkpoints/epoch_010.pt
    python generate.py --temperature 1.0 --top_k 50
"""

import argparse
import time
import torch

from config import NanoLLMConfig, require_cuda
from tokenizer import BPETokenizer
from model import NanoLLM


def load_model(checkpoint_path: str, device: torch.device):
    """Load model + tokenizer from a checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config: NanoLLMConfig = ckpt["config"]
    tokenizer = BPETokenizer(target_vocab_size=config.target_vocab_size)
    tokenizer.load(config.tokenizer_path)
    config.vocab_size = tokenizer.vocab_size

    model = NanoLLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    loss = ckpt.get("loss", "?")
    print(f"Loaded epoch {epoch}, loss={loss}")
    return model, tokenizer, config


def generate_text(
    model: NanoLLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    device: torch.device = None,
    use_cache: bool = False,
) -> str:
    """Generate text from a prompt.

    Args:
        use_cache: If True, use `model.generate_fast()` (with KV cache).
                   If False, use `model.generate()` (recomputes Q,K,V each step).
    """
    tokens = tokenizer.encode(prompt, add_special=False)
    idx = torch.tensor([tokens], device=device)

    gen_fn = model.generate_fast if use_cache else model.generate
    with torch.no_grad():
        output = gen_fn(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    return tokenizer.decode(output[0].tolist())


def run_benchmark(
    model: NanoLLM,
    tokenizer: BPETokenizer,
    device: torch.device,
    prompt: str = "The ",
    n_tokens: int = 100,
):
    """Side-by-side speed comparison: generate() vs generate_fast()."""
    tokens = tokenizer.encode(prompt, add_special=False)
    idx = torch.tensor([tokens], device=device)

    print()
    print("=" * 60)
    print(f"Benchmark: {n_tokens} tokens from prompt '{prompt}'")
    print("=" * 60)

    # Warm-up to pay the CUDA kernel/compile cost once
    with torch.no_grad():
        model.generate(idx, max_new_tokens=8, temperature=0.8, top_k=40, top_p=0.9)
        model.generate_fast(idx, max_new_tokens=8, temperature=0.8, top_k=40, top_p=0.9)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # No-cache path
    t0 = time.time()
    with torch.no_grad():
        model.generate(idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40, top_p=0.9)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_slow = time.time() - t0

    # KV-cache path
    t0 = time.time()
    with torch.no_grad():
        model.generate_fast(idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40, top_p=0.9)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fast = time.time() - t0

    slow_tps = n_tokens / t_slow
    fast_tps = n_tokens / t_fast
    speedup = t_slow / t_fast

    print(f"  generate()      : {t_slow*1000:8.1f} ms  |  {slow_tps:6.1f} tok/s")
    print(f"  generate_fast() : {t_fast*1000:8.1f} ms  |  {fast_tps:6.1f} tok/s")
    print(f"  Speedup         : {speedup:.2f}x")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LLM101 Text Generation")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--max_tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k filtering (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling threshold")
    parser.add_argument("--fast", action="store_true",
                        help="Use KV-cache generation (generate_fast)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Compare generate() vs generate_fast() speed")
    parser.add_argument("--benchmark_tokens", type=int, default=100,
                        help="Tokens to generate in --benchmark mode")
    args = parser.parse_args()

    device = require_cuda()
    model, tokenizer, config = load_model(args.checkpoint, device)

    if args.benchmark:
        run_benchmark(model, tokenizer, device, n_tokens=args.benchmark_tokens)
        return

    mode = "KV-cached (fast)" if args.fast else "no cache (reference)"
    print()
    print("=" * 50)
    print(f"LLM101 Interactive Generation [{mode}]")
    print(f"Temperature: {args.temperature}  Top-k: {args.top_k}  Top-p: {args.top_p}")
    print("Type a prompt and press Enter. Type 'quit' to exit.")
    print("=" * 50)

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        text = generate_text(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
            use_cache=args.fast,
        )
        print(f"\n{text}")


if __name__ == "__main__":
    main()
