# AI Systems Performance Engineering — Companion Repository

**Hands-on exercises for GPU profiling, LLM inference optimization, and Linux systems performance.**

Companion repository for the book *AI Performance Engineering: From GPU Kernels to LLM Inference* by Srinivasa Rao Bittla.

---

## Who This Is For

Engineers transitioning from **Enterprise / Cloud Performance Engineering** into **AI Systems & Infrastructure**. You already understand distributed systems, observability, and performance methodology. This repository adds GPU systems depth.

---

## Repository Structure

```
ai-perf-engineering-book/
│
├── I.Foundations/                     ← Part I: Chapters 0–3
│   ├── 0.GPU_Evolution/               ← Ch 0: GPU history (read-and-run)
│   ├── 1.What_Is_AI_Performance_Engineering/  ← Ch 1: Roofline model
│   ├── 2.PyTorch_Fundamentals/        ← Ch 2: Tensors → training loop → timing
│   └── 3.The_AI_Hardware_Stack/       ← Ch 3: CPU, PCIe, GPU, HBM
│
├── II.GPU_Programming_and_Profiling/  ← Part II: Chapters 4–7
│   ├── 4.The_CUDA_Execution_Model/    ← Ch 4: Warps, coalescing, Tensor Cores
│   ├── 5.GPU_Profiling/               ← Ch 5: nsys, ncu, torch.profiler
│   ├── 6.PyTorch_Optimization/        ← Ch 6: AMP, quantization, torch.compile
│   └── 7.DataLoader_Optimization/     ← Ch 7: Pipeline tuning, I/O bottlenecks
│
├── III.Linux_Systems_Profiling/       ← Part III: Chapters 8–9
│   ├── 8.Perf_eBPF_and_Flamegraphs/  ← Ch 8: perf, flamegraphs, bpftrace, locks
│   └── 9.Memory_Hierarchy_and_NUMA/  ← Ch 9: Cache, NUMA, memory bandwidth
│
├── IV.LLM_Inference_Systems/          ← Part IV: Chapters 10–13
│   ├── 10.LLM_Inference_Fundamentals/ ← Ch 10: Prefill/decode, KV cache, metrics
│   ├── 11.Batching_Strategies/        ← Ch 11: Static, continuous, PagedAttention
│   ├── 12.Speculative_Decoding/       ← Ch 12: Draft/target, acceptance rate
│   └── 13.Distributed_Inference/     ← Ch 13: Tensor parallel, NCCL, FSDP
│
├── V.Workload_Benchmarking/           ← Part V: Chapters 14–15
│   ├── 14.Benchmarking_Methodology/   ← Ch 14: Five mistakes, T-L curve
│   └── 15.Porting_a_Workload/         ← Ch 15: Checklist, bottleneck shift
│
├── VI.Capstone_Projects/              ← Part VI: Chapters 16–19
│   ├── 16.LLM_Inference_Optimization/ ← Ch 16: Full optimization lab
│   ├── 17.DataLoader_Bottleneck_Hunt/ ← Ch 17: Diagnose + fix I/O starvation
│   ├── 18.KV_Cache_Memory_Pressure/   ← Ch 18: Scaling + budget planning
│   └── 19.Flamegraph_Challenge/       ← Ch 19: CPU-to-GPU pipeline diagnosis
│
├── VII.Hardware_Landscape/            ← Part VII: Chapters 20–23
├── VIII.Advanced_Performance/         ← Part VIII: Chapters 24–27
├── IX.Production_Capstones/           ← Part IX: Chapters 28–30
│   ├── 20.NVIDIA_Blackwell/           ← Ch 20: Blackwell architecture + profiling
│   ├── 21.AMD_MI300X_ROCm/            ← Ch 21: MI300X, ROCm, vendor comparison
│   └── 22.Custom_Silicon/             ← Ch 22: Gaudi 3, Trainium, AMX, CXL
│
├── Appendices/                        ← Reference material
│   ├── A.Environment_Setup/           ← App A: System setup + verification
│   ├── B.Command_Reference/           ← App B: Every profiling command
│   └── C.Interview_Prep/              ← App C: 50 Q&A for interviews
│
├── shared/
│   ├── models/model.py                ← TinyTransformer used across all exercises
│   └── utils/results_table.py         ← Before/after comparison printer
│
├── scripts/
│   └── setup_env.sh                   ← Install all dependencies
│
└── docs/
    ├── ROADMAP.md                     ← Learning plan with timelines
    ├── COMMANDS.md                    ← Profiling command reference
    ├── INTERVIEW_PREP.md              ← Interview Q&A reference
    └── HARDWARE_SETUP.md              ← RTX 4060 / cloud setup guide
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 (12 GB) | RTX 4060 (8 GB) / RTX 4070 |
| CPU | 6 cores | 8+ cores |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB SSD | 500 GB NVMe |
| OS | Ubuntu 20.04 | Ubuntu 22.04 LTS |
| CUDA | 11.8 | 12.x |
| Python | 3.10 | 3.11 |
| PyTorch | 2.0 | 2.2+ |

**CPU-only:** All exercises include CPU fallbacks. GPU-specific measurements are skipped gracefully.

---

## Getting Started

### 1. Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/ai-perf-engineering-book.git
cd ai-perf-engineering-book

# Verify your environment before starting any chapter
python Appendices/A.Environment_Setup/A.1_environment_check.py
```

### 2. Part I — Foundations (start here)

```bash
# Chapter 0: GPU history (read-and-run, no TODOs)
python I.Foundations/0.GPU_Evolution/0.1_gpu_evolution.py

# Chapter 1: Roofline model
python I.Foundations/1.What_Is_AI_Performance_Engineering/1.1_roofline_model.py

# Chapter 2: PyTorch fundamentals (6 exercises with TODOs)
python I.Foundations/2.PyTorch_Fundamentals/2.1_tensors.py
python I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py
python I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py
python I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py
python I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py
python I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py
```

### 3. Part II — GPU Profiling (most important part)

```bash
# Chapter 4: CUDA execution model
python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.1_cuda_foundations.py
python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.2_execution_model.py
python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.3_memory_coalescing.py
python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py

# Chapter 5: Profiling tools
python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py
python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py
python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.3_torch_profiler.py

# Chapter 6: Optimization
python II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.1_precision_and_amp.py
python II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.2_quantization.py
python II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.3_torch_compile.py
```

### 4. Part VI — Capstone Projects (portfolio artifacts)

```bash
# Chapter 16: LLM inference optimization lab
python VI.Capstone_Projects/16.LLM_Inference_Optimization/16.1_production_readiness.py
python VI.Capstone_Projects/16.LLM_Inference_Optimization/16.2_baseline_inference.py
python VI.Capstone_Projects/16.LLM_Inference_Optimization/16.3_precision_and_compile.py
python VI.Capstone_Projects/16.LLM_Inference_Optimization/16.4_profiling_audit.py

# Chapter 17: DataLoader bottleneck hunt
python VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.1_slow_dataloader.py
python VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.2_fast_dataloader.py
```

### 5. Appendices

```bash
# Full profiling command reference + live demo
python Appendices/B.Command_Reference/B.1_profiling_cheatsheet.py

# Interview prep: 50 Q&A across 5 domains
python Appendices/C.Interview_Prep/C.1_interview_questions.py
python Appendices/C.Interview_Prep/C.1_interview_questions.py --quiz
python Appendices/C.Interview_Prep/C.1_interview_questions.py --cat 2
```

---

## Exercise Format

Every exercise file:

- Has a **`Run:` line** in the docstring — the exact command from the repo root.
- Has **`TODO` blocks** — fill in the calculation or code; assertions verify your answer.
- Has a **`Next:` footer** pointing to the next exercise in reading order.
- Runs on **CPU-only** machines (GPU sections degrade gracefully).
- Background/intro files (numbered `N.0_` or the `.1_` file in chapters with an intro) are **read-and-run** with no TODOs.

---

## Tool Reference

| Tool | Category | Purpose | Install |
|---|---|---|---|
| `nsys` | GPU profiling | System-wide GPU timeline | CUDA Toolkit |
| `ncu` | GPU profiling | Per-kernel hardware counters | CUDA Toolkit |
| `torch.profiler` | GPU profiling | PyTorch op-level timing | `pip install torch` |
| `nvidia-smi` | GPU monitoring | Utilisation & memory | CUDA Toolkit |
| `nvitop` | GPU monitoring | Rich process monitor | `pip install nvitop` |
| `py-spy` | Python profiling | CPU flamegraphs | `pip install py-spy` |
| `perf` | Linux perf | Hardware event counters | `apt install linux-tools-generic` |
| `bpftrace` | eBPF | Kernel tracing one-liners | `apt install bpftrace` |
| `opensnoop-bpfcc` | eBPF / I/O | File open tracing | `apt install bpfcc-tools` |
| `biolatency-bpfcc` | eBPF / I/O | Block I/O latency histograms | `apt install bpfcc-tools` |
| `iostat` / `vmstat` | I/O monitoring | I/O and VM statistics | `apt install sysstat` |
| `numactl` | NUMA | Memory node binding | `apt install numactl` |

> **Ubuntu 24.04:** BCC tools use a `-bpfcc` suffix.

---

## Learning Path

| Part | Chapters | Duration | Key Outcome |
|---|---|---|---|
| I — Foundations | 0–3 | 2 weeks | Roofline model, PyTorch, hardware mental model |
| II — GPU Profiling | 4–7 | 5 weeks | Profile any GPU workload end-to-end |
| III — Linux Systems | 8–9 | 3 weeks | Flamegraphs, perf, eBPF, NUMA |
| IV — LLM Inference | 10–13 | 4 weeks | KV cache, batching, distributed inference |
| V — Benchmarking | 14–15 | 2 weeks | Repeatable methodology, T-L curves |
| VI — Capstone | 16–19 | 4 weeks | Portfolio-ready optimization projects |
| VII — Hardware | 20–23 | 1 week | Blackwell, MI300X, custom silicon, accelerator spectrum |
| VIII — Advanced | 24–27 | 1 week | Triton, FlashAttention, distributed training, observability |
| IX — Capstones | 28–30 | 1 week | vLLM serving, multi-GPU scaling, cloud cost |

**Total: ~5 months part-time**

---

## Docker Environment

```bash
docker build -t ai-perf-eng .

docker run --gpus all -it --rm \
  --privileged --pid=host --ipc=host \
  -v /lib/modules:/lib/modules:ro \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v $(pwd):/workspace \
  ai-perf-eng
```

See [`docs/HARDWARE_SETUP.md`](docs/HARDWARE_SETUP.md) for full setup including Windows/WSL2 and cloud instances.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `bpfcc-tools` not found | Package renamed in Ubuntu 24.04 — use `-bpfcc` suffix |
| `perf: Permission denied` | Add `--privileged` to `docker run` |
| `bpftrace: Cannot open tracefs` | Mount `-v /sys/kernel/debug:/sys/kernel/debug` |
| `torch.cuda.is_available() == False` | Install NVIDIA Container Toolkit on host |
| CUDA OOM on 8 GB GPU | Switch to FP8/INT4, reduce batch size, enable `torch.compile` |

---

## License

MIT
