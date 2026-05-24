# AI Systems Performance Engineering — Complete Reference

**Hands-on GPU profiling, LLM inference optimisation, and Linux systems performance.**  
Full-stack environment built on NVIDIA NGC PyTorch 25.01 (Ubuntu 24.04 · CUDA 12.8 · PyTorch 2.6).

---

## Who This Is For

Engineers transitioning from **Enterprise / Cloud Performance Engineering** into **AI Systems & Infrastructure Performance Engineering**. You already know distributed systems, observability, and performance methodology. This repository adds GPU systems depth.

---

## What You Will Be Able to Do After Completion

- Profile GPU workloads with `nsys`, `ncu`, and `torch.profiler`
- Diagnose CPU↔GPU bottlenecks using `perf`, `eBPF`, and flamegraphs
- Optimise LLM inference: batch size, KV cache, quantization, `torch.compile`
- Benchmark AI workloads end-to-end with repeatable methodology
- Understand and navigate distributed inference systems (NCCL, FSDP, tensor parallelism)
- Discuss hardware/software tradeoffs like a systems performance engineer

---

## Repository Structure

```
ai-perf-engineering/
│
├── 00_pytorch_basics/          ← Start here if new to PyTorch
│   └── exercises/              ← Fill in the TODOs (solutions in the book)
│
├── 01_phase1_foundations/      ← CPU architecture & Linux internals
│   ├── cpu_arch/
│   └── linux_perf/
│
├── 02_phase2_gpu/              ← GPU programming & CUDA profiling (most critical)
│   ├── module3_cuda_fundamentals/
│   ├── module4_profiling/
│   └── module5_pytorch_perf/
│
├── 03_phase3_systems/          ← Linux low-level profiling
│   ├── module6_linux_profiling/
│   └── module7_memory_numa/
│
├── 04_phase4_inference/        ← LLM inference systems
│   ├── module8_llm_systems/
│   └── module9_distributed/
│
├── 05_phase5_workload/         ← Workload porting & benchmarking
│   ├── module10_porting/
│   └── module11_benchmarking/
│
├── 06_capstone_projects/       ← End-to-end portfolio projects
│   ├── project1_llm_opt/       ← LLM Inference Optimisation Lab
│   ├── project2_dataloader/    ← DataLoader I/O Bottleneck Hunt
│   ├── project3_kv_cache/      ← KV Cache Memory Pressure Experiment
│   └── project4_flamegraph/    ← CPU-to-GPU Pipeline Flamegraph Challenge
│
├── shared/
│   ├── models/model.py         ← TinyTransformer used across all scripts
│   └── utils/results_table.py  ← Before/after comparison table printer
│
├── scripts/
│   └── setup_env.sh            ← Install all dependencies
│
└── docs/
    ├── ROADMAP.md              ← 6-phase learning plan with timelines
    ├── COMMANDS.md             ← Every profiling command in one place
    ├── INTERVIEW_PREP.md       ← Questions to answer before interviewing
    └── HARDWARE_SETUP.md       ← RTX 4060 environment setup guide
```

---

## Hardware Requirements

- **GPU:** NVIDIA RTX 4060 (8GB) or equivalent CUDA-capable GPU
- **OS:** Ubuntu 22.04 / 24.04 (recommended) or WSL2 on Windows
- **RAM:** 16GB+ recommended
- **Storage:** 50GB+ for models and datasets
- **CUDA:** 12.x / 12.8

See [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) for detailed setup.

---

## Tool Reference

| Tool | Category | Purpose | Install |
|------|----------|---------|---------|
| `nsys` | GPU profiling | System-wide GPU timeline | CUDA Toolkit |
| `ncu` | GPU profiling | Per-kernel hardware counters | CUDA Toolkit |
| `torch.profiler` | GPU profiling | PyTorch op-level timing | `pip install torch` |
| `nvidia-smi` | GPU profiling | GPU utilisation & memory | CUDA Toolkit |
| `nvitop` | GPU profiling | Rich GPU process monitor | `pip install nvitop` |
| `vLLM` | LLM serving | High-throughput LLM serving | `pip install vllm` |
| `benchmark_throughput.py` | LLM serving | Offline throughput benchmarks | bundled with vLLM |
| `benchmark_serving.py` | LLM serving | Online serving benchmarks | bundled with vLLM |
| `py-spy` | Python profiling | Python CPU flamegraphs | `pip install py-spy` |
| `torch.utils.bottleneck` | Python profiling | Quick bottleneck scan | bundled with torch |
| `perf` | Linux perf | Linux hardware counters | `apt install linux-tools-generic` |
| `bpftrace` | Linux perf / eBPF | eBPF kernel tracing | `apt install bpftrace` |
| `flamegraph.pl` | Linux perf | SVG flamegraph renderer | bundled with perf scripts |
| `opensnoop-bpfcc` | eBPF / I/O | File open tracing | `apt install bpfcc-tools` |
| `biolatency-bpfcc` | eBPF / I/O | Block I/O latency histograms | `apt install bpfcc-tools` |
| `iostat` / `vmstat` | eBPF / I/O | I/O and VM statistics | `apt install sysstat` |
| `numactl` | NUMA | NUMA memory binding | `apt install numactl` |

> **Ubuntu 24.04 note:** BCC tool names carry a `-bpfcc` suffix. Use `opensnoop-bpfcc` instead of `opensnoop`, `biolatency-bpfcc` instead of `biolatency`, etc.

See [`docs/COMMANDS.md`](docs/COMMANDS.md) for every profiling command in one place.

---

## Docker Environment

### Prerequisites

- NVIDIA GPU with driver ≥ 545 on the host
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker 24+ or Docker Desktop with WSL2 backend
- Free NVIDIA NGC account to pull the base image

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <your NGC API key>
```

### Build

```bash
docker build -t gpu-llm-monitoring-nvcr .
```

Expected build time: **20–35 minutes** (the `vllm` pip step compiles CUDA kernels).  
To skip vLLM during initial testing, comment out the pip line in the Dockerfile, verify the image, then restore and rebuild.

### Run

**Standard — with Learning folder mounted:**

```bash
docker run --gpus all -it --rm \
  --privileged \
  --pid=host \
  --ipc=host \
  -v /lib/modules:/lib/modules:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /mnt/d/Learning:/workspace/Learning \
  --name gpu-llm-monitor \
  gpu-llm-monitoring-nvcr
```

**Read-only mount (protect host files):**

```bash
docker run --gpus all -it --rm \
  --privileged \
  --pid=host \
  --ipc=host \
  -v /lib/modules:/lib/modules:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /mnt/d/Learning:/workspace/Learning:ro \
  --name gpu-llm-monitor \
  gpu-llm-monitoring-nvcr
```

**Docker Desktop for Windows (native Windows paths):**

```bash
docker run --gpus all -it --rm \
  --privileged \
  --pid=host \
  --ipc=host \
  -v /lib/modules:/lib/modules:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v D:\Learning:/workspace/Learning \
  --name gpu-llm-monitor \
  gpu-llm-monitoring-nvcr
```

**Re-attach to a running container:**

```bash
docker exec -it gpu-llm-monitor bash
```

### Docker Flag Reference

| Flag | Why it's required |
|------|-------------------|
| `--gpus all` | Expose all NVIDIA GPUs to the container |
| `--privileged` | Required for `perf` and all eBPF / BCC tools |
| `--pid=host` | Required for `py-spy` to attach to host processes |
| `--ipc=host` | Required for vLLM shared memory (tensor parallel) |
| `-v /lib/modules:ro` | `perf` needs host kernel module symbols |
| `-v /sys/kernel/debug` | `bpftrace` needs tracefs access |
| `-v /mnt/d/Learning` | Your roadmap scripts and learning materials |

---

## Getting Started

### 1. Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/ai-perf-engineering-book.git
cd ai-perf-engineering-book

chmod +x scripts/setup_env.sh
bash scripts/setup_env.sh

source ~/capstone_venv/bin/activate
```

### 2. PyTorch Basics (2–3 days if new to PyTorch)

```bash
cd 00_pytorch_basics/exercises

python exercise_01_tensors.py            # tensors, shapes, devices
python exercise_02_autograd.py           # gradients, backward, no_grad
python exercise_03_nn_modules.py         # building models
python exercise_04_training_loop.py      # complete training loop + AMP
python exercise_05_performance_basics.py # CUDA timing, profiler, memory
```

Each exercise has `TODO` blocks with hints at the bottom.

### 3. Phase 2 — GPU Profiling (most important phase)

```bash
cd 02_phase2_gpu/module3_cuda_fundamentals

python vector_add.py
python occupancy_experiment.py
python matmul_bench.py              # builds roofline intuition

cd ../module4_profiling
nsys profile --stats=true python train.py --task lm --steps 50
python profile_pytorch_infer.py     # all profiling tools in one script

cd ../module5_pytorch_perf
python fp16_bf16_bench.py
python llama_infer_optimize.py      # 5-step optimisation ladder
```

### 4. Capstone Projects (portfolio artifacts)

```bash
# Project 1: LLM Inference Optimisation
cd 06_capstone_projects/project1_llm_opt
python baseline_inference.py --model gpt2
bash profile_nsys.sh gpt2
python torch_compile_bench.py --model gpt2

# Project 2: DataLoader I/O Bottleneck
cd ../project2_dataloader
python slow_dataloader.py &
bash diagnose_io.sh
python fast_dataloader.py
```

---

## Quick-Start Commands Inside the Container

```bash
# Verify GPU is visible
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Real-time GPU dashboard
nvitop

# First GPU profile
nsys profile --stats=true python /workspace/Learning/train.py --steps 20

# Kernel deep dive
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    --kernel-name ".*sgemm.*" python /workspace/Learning/train.py

# vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 --dtype float16 --gpu-memory-utilization 0.85 --port 8000

# eBPF disk tracer (Ubuntu 24.04 suffix)
opensnoop-bpfcc
biolatency-bpfcc

# Flamegraph
perf record -g -F 99 python /workspace/Learning/train.py
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
```

---

## Learning Path

| Phase | Duration | Key Outcome |
|-------|----------|-------------|
| **00 PyTorch Basics** | 2–3 days | Write & time any PyTorch code |
| **Phase 1** Foundation | 3 weeks | Hardware + Linux mental model |
| **Phase 2** GPU Profiling | 5 weeks | Profile any GPU workload |
| **Phase 3** Systems | 4 weeks | Flamegraph, perf, eBPF fluency |
| **Phase 4** Inference | 4 weeks | LLM serving + distributed systems |
| **Phase 5** Benchmarking | 4 weeks | Characterise and compare workloads |
| **Capstone** Projects | 4 weeks | Portfolio-ready artifacts |

**Total: ~5–6 months part-time**

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the detailed plan.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `E: Unable to locate package bcc-tools` | Use `bpfcc-tools` — package renamed in Ubuntu 24.04 |
| `perf: Permission denied` | Add `--privileged` to the `docker run` command |
| `bpftrace: Cannot open tracefs` | Mount `-v /sys/kernel/debug:/sys/kernel/debug` |
| `py-spy: Permission denied (os error 1)` | Add `--pid=host` to the `docker run` command |
| `torch.cuda.is_available() == False` | Install NVIDIA Container Toolkit on the host |
| vLLM OOM on RTX 4060 8 GB | Use `--gpu-memory-utilization 0.85` and INT4 models |

---

## Contributing

This is a personal learning repository. PRs welcome for bugs or improvements.

## License

MIT
