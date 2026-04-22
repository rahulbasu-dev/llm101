#!/bin/bash
# LLM101 — Setup & Run Script
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
    echo "║              LLM101 — Build Your LLM             ║"
    echo "║     From Scratch on RTX 4080 (12GB VRAM)        ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ── Setup ───────────────────────────────────────────────────
do_setup() {
    # ── Sanity check: warn if running as root ──
    if [ "${EUID:-$(id -u)}" -eq 0 ]; then
        echo -e "${YELLOW}  ⚠ You are running run.sh as root (sudo).${NC}"
        echo -e "${YELLOW}    This will install pip packages to root's site-packages,${NC}"
        echo -e "${YELLOW}    which your non-root user won't find when running Python.${NC}"
        echo -e "${YELLOW}    Recommended:  exit sudo, then:  bash run.sh setup${NC}"
        echo -e "${YELLOW}    Use sudo only for 'apt install' (package management).${NC}"
        echo
    fi

    echo -e "${GREEN}[1/4] Python environment...${NC}"

    # Remove any partial / broken venv from a prior failed run (no activate
    # script means ensurepip failed and the directory is useless).
    if [ -d ".venv" ] && [ ! -f ".venv/bin/activate" ]; then
        echo -e "${YELLOW}  Removing broken .venv from a prior failed setup.${NC}"
        rm -rf .venv 2>/dev/null || \
            { echo -e "${YELLOW}  (Could not rm -rf .venv — probably owned by root. Try:${NC}"
              echo -e "${YELLOW}     sudo rm -rf .venv   then re-run without sudo)${NC}"
              return 1; }
    fi

    # Try to create a venv. Use mktemp for error capture (avoids /tmp/ name
    # collisions and sudo ownership weirdness). Real error goes to stderr
    # regardless, so the user sees it.
    if [ ! -d ".venv" ]; then
        VENV_ERR=$(mktemp -t llm101_venv_err.XXXXXX 2>/dev/null || echo "/dev/null")
        if python3 -m venv .venv 2>"$VENV_ERR"; then
            echo -e "${GREEN}  Created .venv${NC}"
        else
            echo -e "${YELLOW}  Could not create .venv. Error:${NC}"
            [ -s "$VENV_ERR" ] && sed 's/^/    /' "$VENV_ERR" || echo "    (no stderr captured)"
            rm -rf .venv 2>/dev/null
            echo
            echo -e "${YELLOW}  Fix (Debian/Ubuntu WSL — run ONCE):${NC}"
            echo -e "${YELLOW}    sudo apt install -y python3-venv python3-pip${NC}"
            echo -e "${YELLOW}  Then re-run (without sudo):${NC}"
            echo -e "${YELLOW}    bash run.sh setup${NC}"
            echo
        fi
        rm -f "$VENV_ERR" 2>/dev/null
    fi

    # Activate the venv if it's usable; otherwise use system Python.
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        PIP="python3 -m pip"
    else
        # System Python fallback. Detect PEP 668 — Ubuntu 24.04 / Python 3.12+
        # refuse system-wide pip install without --break-system-packages.
        # Rather than break the system, we fail with clear instructions.
        if python3 -c "import pathlib, sys, sysconfig; p = pathlib.Path(sysconfig.get_paths()['stdlib']).parent / 'EXTERNALLY-MANAGED'; sys.exit(0 if p.exists() else 1)"; then
            echo -e "${YELLOW}  ✗ System Python is externally-managed (PEP 668).${NC}"
            echo -e "${YELLOW}    Can't pip install without a venv. Fix:${NC}"
            echo -e "${YELLOW}      sudo apt install -y python3-venv python3-pip${NC}"
            echo -e "${YELLOW}      bash run.sh setup    (without sudo)${NC}"
            return 1
        fi
        PIP="python3 -m pip"
    fi

    echo -e "${GREEN}[2/4] Installing CUDA-enabled PyTorch (cu121) + deps...${NC}"
    echo -e "${YELLOW}    (torch cu121 is ~2 GB — 3-10 min depending on connection)${NC}"
    $PIP install --upgrade pip -q
    # CUDA-enabled build (cu121 — works with any RTX 3xxx/4xxx/5xxx GPU).
    # LLM101 requires CUDA; see config.require_cuda().
    # NOT using -q here: users need to see progress bars on the multi-GB
    # torch download, otherwise the terminal looks frozen for minutes.
    $PIP install --upgrade torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu121
    $PIP install matplotlib numpy tqdm pytest -q

    echo -e "${GREEN}[3/4] Verifying GPU...${NC}"
    python3 -c "
import sys, torch
if '+cpu' in torch.__version__:
    print('  ✗ ERROR: CPU-only torch installed (got ' + torch.__version__ + ')')
    print('  ✗ LLM101 requires CUDA. Run: bash run.sh setup again.')
    sys.exit(1)
if not torch.cuda.is_available():
    print('  ✗ ERROR: CUDA torch installed but no GPU detected.')
    print('    - Check NVIDIA driver: run nvidia-smi')
    print('    - Are you in WSL2 with GPU passthrough enabled?')
    sys.exit(1)
props = torch.cuda.get_device_properties(0)
print(f'  ✓ GPU:  {props.name}')
print(f'  ✓ VRAM: {props.total_mem / 1e9:.1f} GB')
print(f'  ✓ bf16: {torch.cuda.is_bf16_supported()}')
print(f'  ✓ CUDA: {torch.version.cuda} · torch {torch.__version__}')
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
    echo -e "${GREEN}Starting LLM101 training...${NC}"
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

# ── Test (pytest suite on mock data, CPU-only) ──────────────
do_test() {
    source .venv/bin/activate 2>/dev/null || true
    echo -e "${GREEN}Running test suite...${NC}"
    PYTHONIOENCODING=utf-8 python3 -m pytest tests/ "$@"
}

# ── UI (Gradio webinar console) ─────────────────────────────
do_ui() {
    source .venv/bin/activate 2>/dev/null || true
    # Gradio 5.x — 4.44 has a schema-introspection bug that crashes on launch.
    if ! python3 -c "import gradio; v=int(gradio.__version__.split('.')[0]); exit(0 if v>=5 else 1)" 2>/dev/null; then
        echo -e "${YELLOW}Installing/upgrading gradio to 5.x...${NC}"
        pip install "gradio>=5,<6" -q
    fi
    echo -e "${GREEN}Launching LLM101 Webinar Console on http://127.0.0.1:7860 ...${NC}"
    echo -e "${YELLOW}Tip: add --share for a public URL (webinar mode)${NC}"
    PYTHONIOENCODING=utf-8 python3 app.py "$@"
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
    test)       shift; do_test "$@" ;;
    ui)         shift; do_ui "$@" ;;
    verify)     do_verify ;;
    *)
        echo "Usage: bash run.sh <command>"
        echo
        echo "Commands:"
        echo "  setup      Install dependencies + download training data"
        echo "  verify     Quick test — verify model shapes and tokenizer"
        echo "  train      Train the LLM101 (takes ~5-15 min on RTX 4080)"
        echo "  generate   Interactive text generation from trained model"
        echo "  visualise  Generate attention heatmaps (for webinar slides)"
        echo "  teach      Generate 16-slide step-by-step forward-pass walkthrough"
        echo "  benchmark  Compare generate() vs generate_fast() (KV-cache speedup)"
        echo "  test       Run pytest suite on mock data (CPU, no training data needed)"
        echo "  ui         Launch Gradio webinar console (Generate/Teach/Attention/Benchmark tabs)"
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
