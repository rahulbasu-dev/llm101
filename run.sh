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

# ── Hardware Detection ─────────────────────────────────────
# Probes for ML accelerators and sets:
#   HW_ACCEL       — "nvidia" | "amd" | "cpu"
#   HW_LABEL       — human-readable string for the banner
#   HW_TORCH_INDEX — pip --index-url for the correct PyTorch build
#   HW_NOTE        — optional advisory (e.g. NPU detected)
detect_hw() {
    HW_ACCEL="cpu"
    HW_LABEL=""
    HW_TORCH_INDEX=""
    HW_NOTE=""

    # 1. NVIDIA GPU — nvidia-smi works in WSL2 via Windows driver passthrough
    if command -v nvidia-smi &>/dev/null; then
        local gpu_csv
        gpu_csv=$(nvidia-smi --query-gpu=name,memory.total \
                  --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -n "$gpu_csv" ]; then
            local gpu_name gpu_mem_mib gpu_mem_gb
            gpu_name=$(echo "$gpu_csv" | cut -d',' -f1 | xargs)
            gpu_mem_mib=$(echo "$gpu_csv" | cut -d',' -f2 | xargs)
            gpu_mem_gb=$(( gpu_mem_mib / 1024 ))
            HW_ACCEL="nvidia"
            HW_LABEL="${gpu_name} (${gpu_mem_gb} GB VRAM)"
            HW_TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            return
        fi
    fi

    # 2. AMD GPU with ROCm
    if command -v rocminfo &>/dev/null; then
        local amd_name
        amd_name=$(rocminfo 2>/dev/null | grep -m1 "Marketing Name" \
                   | sed 's/.*:\s*//' | xargs)
        if [ -n "$amd_name" ]; then
            HW_ACCEL="amd"
            HW_LABEL="${amd_name} (ROCm)"
            HW_TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
            return
        fi
    fi

    # 3. CPU fallback — identify the processor and check for Intel NPU
    local cpu_name
    cpu_name=$(lscpu 2>/dev/null | grep -i "Model name" \
               | sed 's/.*:\s*//' | xargs)
    cpu_name="${cpu_name:-unknown CPU}"

    # Intel NPU (Core Ultra / Meteor Lake / Arrow Lake / Lunar Lake).
    # WSL2 doesn't expose NPU devices, but we infer from the CPU model.
    if [ -d "/sys/class/accel" ] && \
       ls /sys/class/accel/accel* &>/dev/null 2>&1; then
        HW_NOTE="Intel NPU detected (not yet supported by PyTorch — using CPU)"
    elif echo "$cpu_name" | grep -qiE "Core.*Ultra|Meteor.Lake|Arrow.Lake|Lunar.Lake"; then
        HW_NOTE="Intel NPU likely present (not exposed in WSL2 — using CPU)"
    fi

    HW_LABEL="CPU — ${cpu_name}"
    HW_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
}

banner() {
    local sub="$HW_LABEL"
    local len=${#sub}
    [ "$len" -gt 48 ] && sub="${sub:0:48}" && len=48
    local total=$(( 50 - len ))
    local l=$(( total / 2 ))
    local r=$(( total - l ))
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║              LLM101 — Build Your LLM             ║"
    printf '║%*s%s%*s║\n' "$l" '' "$sub" "$r" ''
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"
    if [ -n "$HW_NOTE" ]; then
        echo -e "${YELLOW}  ℹ ${HW_NOTE}${NC}"
        echo
    fi
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

    # matplotlib needs fonts for plot text rendering; WSL2 often ships without them.
    if ! fc-list 2>/dev/null | grep -qi dejavu; then
        echo -e "${YELLOW}  Installing fonts for matplotlib (DejaVu)...${NC}"
        sudo apt install -y fonts-dejavu-core >/dev/null 2>&1 || true
    fi

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

    echo -e "${GREEN}[2/4] Installing PyTorch (${HW_ACCEL})...${NC}"
    $PIP install --upgrade pip -q
    case "$HW_ACCEL" in
        nvidia)
            echo -e "${YELLOW}    CUDA-enabled build (cu121) — ~2 GB download${NC}" ;;
        amd)
            echo -e "${YELLOW}    ROCm-enabled build — ~2 GB download${NC}" ;;
        *)
            echo -e "${YELLOW}    CPU-only build — ~200 MB download${NC}"
            echo -e "${YELLOW}    Training will work but is ~10× slower than GPU.${NC}" ;;
    esac
    # Show progress bars — the download can be multi-GB for GPU builds.
    $PIP install --upgrade torch torchvision torchaudio \
        --index-url "$HW_TORCH_INDEX"
    $PIP install matplotlib numpy tqdm pytest -q

    echo -e "${GREEN}[3/4] Verifying accelerator...${NC}"
    python3 -c "
import torch, platform, multiprocessing
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  ✓ GPU:  {props.name}')
    print(f'  ✓ VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'  ✓ bf16: {torch.cuda.is_bf16_supported()}')
    print(f'  ✓ CUDA: {torch.version.cuda} · torch {torch.__version__}')
else:
    cores = multiprocessing.cpu_count()
    cpu = platform.processor() or 'unknown'
    print(f'  ✓ torch {torch.__version__} (CPU mode)')
    print(f'  ✓ CPU:  {cpu}  ·  Cores: {cores}')
    print(f'  ℹ No GPU detected — training runs on CPU (slower but works)')
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
    # All args pass through to train.py — supports:
    #   bash run.sh train --max-epochs 10 --warmup-steps 50 --batch-size 32
    python3 train.py "$@"
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

# ── Verify (GPU + shape test + tokenizer test) ──────────────
do_verify() {
    source .venv/bin/activate 2>/dev/null || true

    echo -e "${GREEN}[1/4] Accelerator check...${NC}"
    python3 -c "
import torch, platform, os, multiprocessing
# ── Device ──
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'  ✓ GPU:  {props.name}')
    print(f'  ✓ VRAM: {props.total_memory / 1e9:.1f} GB')
    print(f'  ✓ bf16: {torch.cuda.is_bf16_supported()}')
    print(f'  ✓ CUDA: {torch.version.cuda} · torch {torch.__version__}')
else:
    cores = multiprocessing.cpu_count()
    cpu = platform.processor() or 'unknown'
    print(f'  ✓ torch {torch.__version__} (CPU mode)')
    print(f'  ✓ CPU:  {cpu}')
    print(f'  ✓ Cores: {cores}')
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        print(f'  ✓ RAM:  {ram_gb:.1f} GB')
    except ImportError:
        pass
    print(f'  ℹ No GPU — training runs on CPU')
"

    echo
    echo -e "${GREEN}[2/4] Model shape + KV-cache equivalence...${NC}"
    python3 model.py
    echo
    echo -e "${GREEN}[3/4] Tokenizer roundtrip...${NC}"
    python3 tokenizer.py
    echo

    echo -e "${GREEN}[4/4] Training time estimate...${NC}"
    python3 -c "
import time, sys, torch, os
from config import NanoLLMConfig
from model import NanoLLM

config = NanoLLMConfig()
config.vocab_size = config.target_vocab_size  # approximate for estimate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NanoLLM(config).to(device)
model.train()

# ── torch.compile (mirrors what train.py does) ──
compiled = False
if hasattr(torch, 'compile'):
    print('  Compiling model (one-time cost)...', end='', flush=True)
    t_comp = time.perf_counter()
    try:
        model = torch.compile(model)
        _w = torch.randint(0, config.vocab_size, (2, config.max_seq_len), device=device)
        with torch.no_grad():
            model(_w)
        del _w
        comp_s = time.perf_counter() - t_comp
        compiled = True
        print(f' done in {comp_s:.0f}s')
    except Exception as e:
        print(f' skipped ({e})')

B, T = config.batch_size, config.max_seq_len
dummy = torch.randint(0, config.vocab_size, (B, T), device=device)

# Warm-up pass
logits, loss = model(dummy, targets=dummy)
loss.backward()

# Timed passes
n_runs = 5
torch.cuda.synchronize() if device.type == 'cuda' else None
t0 = time.perf_counter()
for _ in range(n_runs):
    logits, loss = model(dummy, targets=dummy)
    loss.backward()
    torch.cuda.synchronize() if device.type == 'cuda' else None
elapsed = (time.perf_counter() - t0) / n_runs

# Estimate total training time from the corpus size
corpus_path = config.data_path
if os.path.exists(corpus_path):
    corpus_chars = os.path.getsize(corpus_path)
    est_tokens = int(corpus_chars / 4.0)  # rough char-to-token ratio
else:
    est_tokens = 280_000  # TinyShakespeare default

train_tokens = int(est_tokens * 0.9)
stride = config.max_seq_len // 2
n_windows = max(1, (train_tokens - config.max_seq_len) // stride + 1)
batches_per_epoch = max(1, n_windows // config.batch_size)
total_steps = batches_per_epoch * config.max_epochs
total_secs = total_steps * elapsed
# Add compile overhead to the total (one-time cost during training)
if compiled:
    total_secs += comp_s

label = 'compiled' if compiled else 'uncompiled'
print(f'  Step time:   {elapsed:.3f}s  ({label}, batch={B}, seq_len={T})')
print(f'  Steps/epoch: ~{batches_per_epoch}  ·  Epochs: {config.max_epochs}  ·  Total: ~{total_steps:,} steps')

if total_secs < 120:
    print(f'  ⏱ Estimated training: ~{total_secs:.0f} seconds')
elif total_secs < 3600:
    print(f'  ⏱ Estimated training: ~{total_secs/60:.0f} minutes')
else:
    h = int(total_secs // 3600)
    m = int((total_secs % 3600) // 60)
    print(f'  ⏱ Estimated training: ~{h}h {m}m')

if device.type != 'cuda':
    print()
    print(f'  Tip: reduce epochs for a quick test run:')
    print(f'    bash run.sh train --max-epochs 3')
"
}

# ── Main ────────────────────────────────────────────────────
detect_hw

# Auto-enable CPU mode when no CUDA GPU is detected, so all Python
# entry points (which call config.require_cuda()) work without manual
# env vars.  Users running Python directly still get the CUDA gate.
if [ "$HW_ACCEL" != "nvidia" ]; then
    export NANOLLM_ALLOW_CPU=1
fi

banner

case "${1:-help}" in
    setup)      do_setup ;;
    train)      shift; do_train "$@" ;;
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
        echo "  train      Train LLM101 (~5 min GPU, ~30 min CPU)"
        echo "  generate   Interactive text generation from trained model"
        echo "  visualise  Generate attention heatmaps (for webinar slides)"
        echo "  teach      Generate 16-slide step-by-step forward-pass walkthrough"
        echo "  benchmark  Compare generate() vs generate_fast() (KV-cache speedup)"
        echo "  test       Run pytest suite on mock data (CPU, no training data needed)"
        echo "  ui         Launch Gradio webinar console (7-tab interface)"
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
