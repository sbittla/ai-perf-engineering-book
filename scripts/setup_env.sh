#!/usr/bin/env bash
# =============================================================================
# setup_env.sh  —  Environment Setup for AI Systems Performance Engineering
# =============================================================================
# PURPOSE:
#   Install every dependency needed across all 4 capstone projects.
#   Verifies CUDA is accessible, creates a virtual environment, installs
#   Python packages, and checks Linux profiling tools are available.
#
# RUN ONCE before starting any project:
#   chmod +x setup_env.sh && ./setup_env.sh
# =============================================================================

set -euo pipefail   # Exit on error, unset vars, pipe failures

# ── Colour helpers for readable output ───────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
fail() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info() { echo -e "\n─── $* ───"; }

# =============================================================================
# STEP 1 — Verify CUDA is present
# =============================================================================
info "Checking CUDA"

# nvidia-smi talks directly to the GPU driver (not CUDA toolkit)
# If this fails, your GPU driver is not installed
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. Install NVIDIA drivers first."
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
ok "GPU detected"

# nvcc is the CUDA compiler — confirms CUDA Toolkit is installed
if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found. Install CUDA Toolkit 12.x from developer.nvidia.com"
    warn "Required for: Nsight tools, TensorRT"
else
    nvcc --version | grep "release"
    ok "CUDA Toolkit found"
fi

# =============================================================================
# STEP 2 — Create Python virtual environment
# =============================================================================
info "Creating Python virtual environment"

# Use Python 3.10+ for best PyTorch 2.x compatibility
PYTHON_BIN=$(which python3)
$PYTHON_BIN --version

# Create isolated venv so packages don't conflict with system Python
VENV_DIR="$HOME/capstone_venv"
if [ -d "$VENV_DIR" ]; then
    warn "venv already exists at $VENV_DIR — skipping creation"
else
    $PYTHON_BIN -m venv "$VENV_DIR"
    ok "Created venv at $VENV_DIR"
fi

# Activate the venv for the rest of this script
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet

# =============================================================================
# STEP 3 — Install PyTorch (CUDA build)
# =============================================================================
info "Installing PyTorch 2.x (CUDA 12.1 build)"

# The cu121 index URL fetches wheels built against CUDA 12.1
# Change cu121 to cu118 if your CUDA Toolkit is 11.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet

# Verify PyTorch can see the GPU
python3 -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available in PyTorch!'
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
ok "PyTorch CUDA verified"

# =============================================================================
# STEP 4 — Install AI/ML packages
# =============================================================================
info "Installing AI/ML packages"

# transformers  — HuggingFace model hub, tokenizers, generate()
# accelerate    — required by transformers for device_map and multi-GPU
# vllm          — high-throughput LLM serving with PagedAttention
# nvitop        — rich GPU process monitor (better nvidia-smi)
# matplotlib    — plotting memory logs in Project 3
# pandas        — data manipulation for results tables
# tqdm          — progress bars for benchmarks
pip install \
    transformers \
    accelerate \
    datasets \
    vllm \
    nvitop \
    matplotlib \
    pandas \
    tqdm \
    httpx \
    aiohttp \
    --quiet

ok "AI/ML packages installed"

# =============================================================================
# STEP 5 — Install profiling Python tools
# =============================================================================
info "Installing Python profiling tools"

# py-spy  — sampling profiler for Python; generates flamegraph SVGs
#           does NOT require modifying source code
# torch-tb-profiler — TensorBoard plugin for torch.profiler traces
pip install \
    py-spy \
    torch-tb-profiler \
    --quiet

ok "Python profiling tools installed"

# =============================================================================
# STEP 6 — Install Linux system profiling tools (apt)
# =============================================================================
info "Installing Linux system tools"

# perf         — Linux perf_events: hardware counters, CPU profiling
# bpfcc-tools  — BCC eBPF tools: opensnoop, biolatency, runqlat, etc.
# bpftrace     — high-level eBPF scripting language
# sysstat      — provides: iostat, sar, pidstat, mpstat
# linux-tools  — provides: perf
# numactl      — NUMA topology inspection and process binding
# hwloc        — lstopo: visualise CPU/cache/NUMA/GPU topology
# git          — needed for cloning FlameGraph repo
sudo apt-get update -qq
sudo apt-get install -y \
    linux-tools-common \
    linux-tools-generic \
    linux-tools-$(uname -r) \
    bpfcc-tools \
    bpftrace \
    sysstat \
    numactl \
    hwloc \
    strace \
    ltrace \
    htop \
    git \
    --quiet 2>/dev/null || warn "Some apt packages failed — may need sudo or different package names"

ok "Linux tools installed"

# =============================================================================
# STEP 7 — Clone Brendan Gregg's FlameGraph tools
# =============================================================================
info "Installing FlameGraph tools"

FLAMEGRAPH_DIR="$HOME/FlameGraph"
if [ -d "$FLAMEGRAPH_DIR" ]; then
    warn "FlameGraph already cloned at $FLAMEGRAPH_DIR"
else
    # Brendan Gregg's scripts: stackcollapse-perf.pl and flamegraph.pl
    # These convert perf/py-spy output into SVG flame graphs
    git clone --quiet https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
    ok "FlameGraph cloned to $FLAMEGRAPH_DIR"
fi

# Add to PATH so flamegraph.pl is usable directly
echo "export PATH=\$PATH:$FLAMEGRAPH_DIR" >> "$HOME/.bashrc"

# =============================================================================
# STEP 8 — Verify Nsight Systems is installed
# =============================================================================
info "Checking Nsight Systems"

# nsys ships with CUDA Toolkit >= 11 under /usr/local/cuda/bin/nsys
# or can be separately downloaded from developer.nvidia.com/nsight-systems
if command -v nsys &>/dev/null; then
    nsys --version | head -1
    ok "Nsight Systems (nsys) found"
else
    warn "nsys not found. Download Nsight Systems from:"
    warn "  https://developer.nvidia.com/nsight-systems"
    warn "  Or: sudo apt install nsight-systems"
fi

if command -v ncu &>/dev/null; then
    ncu --version | head -1
    ok "Nsight Compute (ncu) found"
else
    warn "ncu not found. Download Nsight Compute from:"
    warn "  https://developer.nvidia.com/nsight-compute"
fi

# =============================================================================
# STEP 9 — Print activation instructions
# =============================================================================
info "Setup Complete"

echo ""
echo "Before running any project script, activate the venv:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Add to your ~/.bashrc to auto-activate:"
echo "  echo 'source $VENV_DIR/bin/activate' >> ~/.bashrc"
echo ""
ok "All done. Start with Project 1: cd project1 && python baseline_inference.py"
