#!/bin/bash
# NanoLLM — Setup & Run Script
# Run: bash run.sh [setup|train|generate|visualise]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colors ──────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║              NanoLLM — Build Your LLM            ║"
    echo "║     From Scratch on RTX 4080 (12GB VRAM)        ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ── Setup ───────────────────────────────────────────────────
do_setup() {
    echo -e "${GREEN}[1/4] Creating Python venv...${NC}"
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    source .venv/bin/activate

    echo -e "${GREEN}[2/4] Installing PyTorch + dependencies...${NC}"
    pip install --upgrade pip -q
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q 2>/dev/null || \
    pip install torch torchvision torchaudio -q
    pip install matplotlib numpy tqdm -q

    echo -e "${GREEN}[3/4] Verifying GPU...${NC}"
    python3 -c "
import torch
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  ✓ GPU: {props.name}')
    print(f'  ✓ VRAM: {props.total_mem / 1e9:.1f} GB')
    print(f'  ✓ bf16: {torch.cuda.is_bf16_supported()}')
    print(f'  ✓ CUDA: {torch.version.cuda}')
else:
    print('  ⚠ No CUDA GPU found — will train on CPU')
"

    echo -e "${GREEN}[4/4] Downloading training data...${NC}"
    mkdir -p data
    if [ ! -f "data/corpus.txt" ]; then
        echo "  Downloading TinyShakespeare (~1.1MB)..."
        if command -v wget &> /dev/null; then
            wget -q "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -O data/corpus.txt
        elif command -v curl &> /dev/null; then
            curl -sL "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -o data/corpus.txt
        else
            echo "  ERROR: Neither wget nor curl found. Please download manually:"
            echo "  URL: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            echo "  Save to: data/corpus.txt"
            exit 1
        fi
        echo "  ✓ data/corpus.txt ($(wc -c < data/corpus.txt) bytes)"
    else
        echo "  ✓ data/corpus.txt already exists ($(wc -c < data/corpus.txt) bytes)"
    fi

    echo
    echo -e "${GREEN}Setup complete!${NC} Run: ${YELLOW}bash run.sh train${NC}"
}

# ── Train ───────────────────────────────────────────────────
do_train() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Starting NanoLLM training...${NC}"
    python3 train.py
}

# ── Generate ────────────────────────────────────────────────
do_generate() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Interactive generation mode${NC}"
    python3 generate.py "$@"
}

# ── Visualise ───────────────────────────────────────────────
do_visualise() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Generating attention visualisations...${NC}"
    python3 visualise.py "$@"
}

# ── Teach (step-by-step forward-pass slides) ────────────────
do_teach() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Generating teaching slides (16-step forward-pass walkthrough)...${NC}"
    if [ -n "$1" ]; then
        python3 teach.py --text "$@"
    else
        python3 teach.py
    fi
}

# ── Benchmark (cache vs no-cache generation) ────────────────
do_benchmark() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Benchmarking generate() vs generate_fast()...${NC}"
    python3 generate.py --benchmark "$@"
}

# ── Verify (quick shape test without training) ──────────────
do_verify() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Running model shape verification...${NC}"
    python3 model.py
    echo
    echo -e "${GREEN}Running tokenizer test...${NC}"
    python3 tokenizer.py
}

# ── Main ────────────────────────────────────────────────────
banner

case "${1:-help}" in
    setup)      do_setup ;;
    train)      do_train ;;
    generate)   shift; do_generate "$@" ;;
    visualise)  shift; do_visualise "$@" ;;
    teach)      shift; do_teach "$@" ;;
    benchmark)  shift; do_benchmark "$@" ;;
    verify)     do_verify ;;
    *)
        echo "Usage: bash run.sh <command>"
        echo
        echo "Commands:"
        echo "  setup      Install dependencies + download training data"
        echo "  verify     Quick test — verify model shapes and tokenizer"
        echo "  train      Train the NanoLLM (takes ~5-15 min on RTX 4080)"
        echo "  generate   Interactive text generation from trained model"
        echo "  visualise  Generate attention heatmaps (for webinar slides)"
        echo "  teach      Generate 16-slide step-by-step forward-pass walkthrough"
        echo "  benchmark  Compare generate() vs generate_fast() (KV-cache speedup)"
        echo
        echo "Quick start:"
        echo "  bash run.sh setup"
        echo "  bash run.sh verify"
        echo "  bash run.sh train"
        echo "  bash run.sh generate --fast"
        echo "  bash run.sh teach"
        echo "  bash run.sh benchmark"
        ;;
esac
