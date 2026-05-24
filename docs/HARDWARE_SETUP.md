# Hardware Setup Guide — RTX 4060 + Ubuntu

Step-by-step environment setup for this course on an RTX 4060 (8GB).

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 4060 (8GB) or RTX 4070 |
| CPU | 6 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 500 GB NVMe |
| OS | Ubuntu 20.04 | Ubuntu 22.04 LTS |

---

## Step 1: Install NVIDIA Drivers

```bash
# Check current driver
nvidia-smi

# Install latest recommended driver
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt install nvidia-driver-535
sudo reboot

# Verify
nvidia-smi
# Should show: NVIDIA RTX 4060, driver 535.x, CUDA 12.x
```

---

## Step 2: Install CUDA Toolkit 12.x

```bash
# Method 1: CUDA Toolkit installer (recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run --toolkit --silent

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Method 2: apt (easier but may not have latest)
sudo apt install cuda-toolkit-12-3

# Verify
nvcc --version
# Should show: release 12.3
```

---

## Step 3: Install Nsight Systems & Nsight Compute

```bash
# Nsight Systems (nsys)
# Usually ships with CUDA Toolkit at: /usr/local/cuda/bin/nsys
which nsys || echo "nsys not found"

# If not found:
sudo apt install nsight-systems

# Download from NVIDIA directly:
# https://developer.nvidia.com/nsight-systems

# Nsight Compute (ncu)
which ncu || echo "ncu not found"

# Install:
sudo apt install nsight-compute
# Or: https://developer.nvidia.com/nsight-compute

# Verify
nsys --version
ncu --version
```

---

## Step 4: Python Environment

```bash
# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip

# Create virtual environment
python3.10 -m venv ~/ai_perf_env
source ~/ai_perf_env/bin/activate

# Auto-activate on login (optional)
echo 'source ~/ai_perf_env/bin/activate' >> ~/.bashrc
```

---

## Step 5: Install PyTorch (CUDA build)

```bash
source ~/ai_perf_env/bin/activate

# PyTorch 2.x with CUDA 12.1
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is visible to PyTorch
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
print(f'Compute cap: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}')
"
```

---

## Step 6: Install Course Dependencies

```bash
source ~/ai_perf_env/bin/activate

# Run the setup script
chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh

# Or install manually:
pip install \
    transformers accelerate datasets \
    vllm \
    nvitop \
    matplotlib pandas tqdm \
    httpx aiohttp \
    py-spy \
    torch-tb-profiler \
    torchvision \
    bitsandbytes
```

---

## Step 7: Install Linux Performance Tools

```bash
# perf
sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)

# bpfcc-tools (opensnoop, biolatency, runqlat, etc.)
sudo apt install bpfcc-tools

# bpftrace
sudo apt install bpftrace

# sysstat (iostat, sar, pidstat)
sudo apt install sysstat

# numactl + lstopo
sudo apt install numactl hwloc

# strace + ltrace
sudo apt install strace ltrace

# htop + btop
sudo apt install htop btop

# git, curl, wget
sudo apt install git curl wget
```

---

## Step 8: Install FlameGraph

```bash
git clone https://github.com/brendangregg/FlameGraph.git ~/FlameGraph
echo 'export PATH=$PATH:~/FlameGraph' >> ~/.bashrc
source ~/.bashrc

# Test
echo "test 10" | flamegraph.pl > /dev/null && echo "FlameGraph works"
```

---

## Step 9: NVIDIA Container Toolkit (for Docker)

```bash
# Install Docker
sudo apt install docker.io
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

---

## Step 10: Verify Everything Works

```bash
source ~/ai_perf_env/bin/activate
cd ai-perf-engineering

# Run the verification script
python3 -c "
import torch, subprocess, sys

checks = {
    'Python 3.10+': sys.version_info >= (3, 10),
    'PyTorch 2.x':  torch.__version__.startswith('2.'),
    'CUDA available': torch.cuda.is_available(),
    'GPU detected': torch.cuda.device_count() > 0,
}

for name, ok in checks.items():
    print(f'  [{'✓' if ok else '✗'}] {name}')

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'\n  GPU: {p.name}')
    print(f'  VRAM: {p.total_memory/1e9:.1f}GB')
    print(f'  Tensor Cores: {\"Yes\" if p.major >= 7 else \"No\"}')
    print(f'  BF16 support: {torch.cuda.is_bf16_supported()}')
"

# Quick tool checks
echo "nsys:      $(nsys --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "ncu:       $(ncu --version  2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "perf:      $(perf --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "py-spy:    $(py-spy --version 2>/dev/null || echo 'NOT FOUND')"
echo "nvitop:    $(python3 -c 'import nvitop; print(nvitop.__version__)' 2>/dev/null || echo 'NOT FOUND')"
echo "flamegraph:$(which flamegraph.pl 2>/dev/null || echo 'NOT FOUND')"
```

---

## RTX 4060 Specific Notes

```
GPU:               Ada Lovelace architecture
CUDA cores:        3072
Tensor Cores:      Yes (4th gen, supports FP16/BF16/INT8/FP8)
VRAM:              8GB GDDR6
Memory bandwidth:  ~272 GB/s
Peak FP16:         ~136 TFLOP/s (tensor cores)
Peak FP32:         ~17 TFLOP/s
PCIe:              4.0 x8 (not x16)
NVLink:            No (consumer GPU)
```

**Memory limits for LLMs on 8GB VRAM:**
- GPT-2 (124M): FP32 ✓, FP16 ✓, easy
- Llama-2-7B: needs INT8 or INT4 quantization (7B × 2 bytes = 14GB FP16)
- INT4 (GPTQ/AWQ): 7B × 0.5 bytes = 3.5GB ✓ with room for KV cache

**Recommended settings:**
```python
# Always use FP16 for inference
model = model.half()

# Use torch.compile for 1.3-2x speedup
model = torch.compile(model, mode="default")

# vLLM with constrained memory
llm = LLM(model="model_name", gpu_memory_utilization=0.85)
```

---

## Useful Aliases

Add to `~/.bashrc`:

```bash
# GPU monitoring
alias gpu='watch -n 0.5 nvidia-smi'
alias gpumem='nvidia-smi --query-gpu=memory.used,memory.free --format=csv'

# Profiling shortcuts
alias nsys-quick='nsys profile --stats=true'
alias flame='py-spy record -o /tmp/flame.svg --'

# Python environment
alias ai='source ~/ai_perf_env/bin/activate'
alias jlab='jupyter lab --no-browser'

# System monitoring
alias sysmon='watch -n 1 "vmstat 1 1 && echo && iostat -xz 1 1"'
```

---

## Troubleshooting

**`CUDA out of memory`**
```python
torch.cuda.empty_cache()
# Check what's using memory:
print(torch.cuda.memory_summary())
# Reduce batch size or use INT8 quantization
```

**`nsys: Permission denied`**
```bash
# Allow perf events for non-root users
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
echo 0 | sudo tee /proc/sys/kernel/kptr_restrict
```

**`ncu: Unable to profile`**
```bash
sudo nvidia-smi -i 0 -pm 1      # enable persistent mode
# Run ncu with sudo or set capabilities:
sudo ncu --set basic python script.py
```

**`bpftrace: permission denied`**
```bash
sudo bpftrace -e '...'
# Or: sudo sysctl -w kernel.unprivileged_bpf_disabled=0
```

**Slow first run (JIT compilation)**
Always warm up before benchmarking — first call triggers cuBLAS autotuning and Triton compilation.
