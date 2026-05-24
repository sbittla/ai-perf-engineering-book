# Learning Roadmap — AI Systems Performance Engineering

**Goal:** Transition from Enterprise Performance Engineering to AI Systems & Infrastructure Performance Engineering in ~5–6 months part-time.

---

## Phase 0 — PyTorch Basics (2–3 days)

**Do this before anything else if you are new to PyTorch.**

| Exercise | File | Concepts |
|----------|------|---------|
| 01 | `00_pytorch_basics/exercises/exercise_01_tensors.py` | Tensors, shapes, dtypes, CPU↔GPU |
| 02 | `00_pytorch_basics/exercises/exercise_02_autograd.py` | Gradients, backward, no_grad, detach |
| 03 | `00_pytorch_basics/exercises/exercise_03_nn_modules.py` | nn.Module, Sequential, train/eval |
| 04 | `00_pytorch_basics/exercises/exercise_04_training_loop.py` | DataLoader, loss, optimizer, AMP |
| 05 | `00_pytorch_basics/exercises/exercise_05_performance_basics.py` | CUDA events, profiler, memory |

**Outcome:** You can read and write any PyTorch model code.

---

## Phase 1 — Computer Architecture & Linux Systems (3 weeks)

**Goal:** Build hardware-near systems understanding before touching GPU code.

### Module 1 — Modern CPU/GPU Architecture

**Learn:**
- CPU pipelines, IPC, cache hierarchy, branch prediction, NUMA, SIMD
- GPU architecture: CUDA cores, Tensor Cores, warps/wavefronts
- Memory bandwidth: PCIe, NVLink, HBM, GDDR6

**Resources:**
- NVIDIA CUDA Architecture Docs: https://docs.nvidia.com/cuda/
- CMU Computer Architecture Course: https://www.cs.cmu.edu/~18447/
- NVIDIA Developer Training: https://developer.nvidia.com/cuda-training

**Labs:**
```bash
# No scripts needed yet — read and build mental models
# Key concepts to understand:
#   - Why does warp size = 32?
#   - What is the cache hierarchy on your RTX 4060?
#   - What is the difference between CUDA cores and Tensor Cores?
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

### Module 2 — Linux Performance Internals

**Learn:**
- Processes/threads, context switching, virtual memory, page faults
- CPU scheduling, I/O stack, interrupts, cgroups

**Labs:**
```bash
# CPU starvation analysis
vmstat 1 20                           # watch us/sy/wa/id columns
perf stat -e context-switches python train.py

# Memory pressure analysis
free -h && vmstat -s
cat /proc/meminfo | grep -E "MemTotal|Cached|SwapUsed"

# I/O bottleneck identification
iostat -xz 1 10
```

---

## Phase 2 — GPU Programming & CUDA Profiling (5 weeks) ← MOST CRITICAL

**Goal:** Establish GPU systems credibility. Everything else builds on this.

### Module 3 — CUDA Fundamentals

**Learn:**
- CUDA execution model: kernels, grids, blocks, threads
- Occupancy, synchronization, streams, pinned memory, unified memory

**Scripts:**
```bash
cd 02_phase2_gpu/module3_cuda_fundamentals/

python vector_add.py                  # threads, blocks, grids, streams
python vector_add.py --exp block_size # how block size affects throughput
python occupancy_experiment.py        # occupancy limiters
python occupancy_experiment.py --exp wave_count   # wave quantisation
python matmul_bench.py                # roofline model
python matmul_bench.py --size 1024    # specific size deep dive
python cuda_kernels.py                # 6 experiments: coalescing, divergence, fusion...
```

**Resources:**
- CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA CUDA Samples: https://github.com/NVIDIA/cuda-samples

### Module 4 — GPU Profiling & Optimisation

**Learn:**
- SM utilisation, warp divergence, memory coalescing, tensor core utilisation
- Memory-bound vs compute-bound kernels, roofline model

**Scripts:**
```bash
cd 02_phase2_gpu/module4_profiling/

# Profile the training script
nsys profile --stats=true python train.py --task lm --steps 50
nsys profile --trace=cuda,nvtx python train.py --nvtx
ncu --set basic python train.py --task lm --steps 5

# Profile HuggingFace model
python profile_pytorch_infer.py --model gpt2

# Find optimal batch size
python optimize_batch_size.py --task lm
python optimize_batch_size.py --task image
```

### Module 5 — PyTorch Performance Engineering

**Learn:**
- Autograd internals, CUDA graphs, torch.compile, mixed precision
- Quantization, activation checkpointing, kernel fusion

**Scripts:**
```bash
cd 02_phase2_gpu/module5_pytorch_perf/

python fp16_bf16_bench.py             # FP32 vs FP16 vs BF16 comparison
python quantization_bench.py          # INT8/INT4 memory + speed tradeoffs
python llama_infer_optimize.py        # 5-step optimisation ladder
```

**Resources:**
- PyTorch Performance Tuning Guide: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

---

## Phase 3 — Systems Profiling & Kernel-Level Analysis (4 weeks)

### Module 6 — Linux Low-Level Profiling

**Learn:**
- CPU flamegraphs, syscall tracing, hardware counters, cache miss analysis

**Scripts:**
```bash
cd 03_phase3_systems/module6_linux_profiling/

# Lock contention — diagnose GIL and mutex issues
python lock_contention.py
strace -c python lock_contention.py

# Cache miss patterns
python cache_miss_analysis.py
sudo perf stat -e cache-misses,cache-references python cache_miss_analysis.py

# Flamegraph of training script
py-spy record -o flamegraph.svg -- python ../../02_phase2_gpu/module4_profiling/train.py
# Open flamegraph.svg in browser
```

**Resources:**
- Brendan Gregg Performance Resources: https://www.brendangregg.com/

### Module 7 — Memory & NUMA Optimisation

**Scripts:**
```bash
cd 03_phase3_systems/module7_memory_numa/

python memory_bench.py                # CPU cache → DRAM → PCIe → GPU BW
python numa_workload.py               # default binding
numactl --cpunodebind=0 --membind=0 python numa_workload.py --label node0
grep "mem_bandwidth" numa_result_*.txt  # compare results
```

---

## Phase 4 — AI Inference Systems & Distributed Runtime (4 weeks)

### Module 8 — LLM Inference Systems

**Learn:**
- KV cache, token generation pipeline, speculative decoding
- Batching strategies, continuous batching, paged attention

**Scripts:**
```bash
cd 04_phase4_inference/module8_llm_systems/

# Understand prefill vs decode
python infer.py --bench --tokens 100
nsys profile --trace=cuda,nvtx python infer.py --nvtx

# KV cache sizing
python kv_cache_sim.py --estimate
python kv_cache_sim.py --simulate --vram-gb 8

# Advanced inference techniques
python speculative_decode.py --K 4
python continuous_batching.py --exp compare

# Long-running server for live profiling
python serve.py &
py-spy record --pid $(pgrep -f serve.py) --duration 10 -o serve_flame.svg
```

### Module 9 — Distributed AI Systems

**Learn:**
- Tensor parallelism, pipeline parallelism, FSDP, NCCL, all-reduce

**Scripts:**
```bash
cd 04_phase4_inference/module9_distributed/

# Single GPU baseline
python infer_distributed.py

# Multi-GPU (requires 2 GPUs)
torchrun --nproc_per_node=2 infer_distributed.py

# NCCL bandwidth
torchrun --nproc_per_node=2 nccl_bench.py

# FSDP training
python fsdp_train.py                  # single GPU simulation
torchrun --nproc_per_node=2 fsdp_train.py --distributed  # real FSDP
```

---

## Phase 5 — Workload Porting & Hardware Characterisation (4 weeks)

### Module 10 — Workload Porting

**Scripts:**
```bash
cd 05_phase5_workload/module10_porting/

# The core porting skill: same workload, different hardware
python workload_port.py --mode all    # CPU → GPU FP32 → FP16 → INT8 → compiled
python workload_port.py --mode compare

# DataLoader CPU affinity
python dataloader_worker.py --label default
taskset -c 0-7 python dataloader_worker.py --label taskset
numactl --cpunodebind=0 --membind=0 python dataloader_worker.py --label numa
grep "Batches/sec" dataloader_result_*.txt
```

### Module 11 — Benchmarking & Workload Characterisation

**Scripts:**
```bash
cd 05_phase5_workload/module11_benchmarking/

# Throughput-latency Pareto curve
python throughput_latency_curve.py --plot

# Full characterisation report (JSON output)
python workload_characterize.py --output report.json

# Profile the characterisation itself
nsys profile python workload_characterize.py --quick
```

---

## Phase 6 — Capstone Projects (4 weeks)

Four end-to-end projects that produce portfolio artifacts.

### Project 1 — LLM Inference Optimisation Lab

```bash
cd 06_capstone_projects/project1_llm_opt/

# Step 1: Baseline
python baseline_inference.py --model gpt2 --runs 5

# Step 2: Profile
bash profile_nsys.sh gpt2
bash profile_ncu.sh gpt2
python torch_profiler_trace.py --model gpt2

# Step 3: Optimise
python torch_compile_bench.py --model gpt2
bash vllm_server_start.sh gpt2 baseline
bash vllm_benchmark.sh
```

**Deliverable:** Before/after tok/s table + Nsight screenshots

### Project 2 — DataLoader I/O Bottleneck Hunt

```bash
cd 06_capstone_projects/project2_dataloader/

# Terminal 1: run slow script
python slow_dataloader.py --sleep-ms 20

# Terminal 2: diagnose
bash diagnose_io.sh

# Fix and verify
python fast_dataloader.py
```

**Deliverable:** iostat/vmstat screenshots + GPU utilisation before/after

### Project 3 — KV Cache Memory Pressure

```bash
cd 06_capstone_projects/project3_kv_cache/

# Terminal 1: constrained server
bash kv_pressure_server.sh gpt2 constrained

# Terminal 2: memory monitor
bash memory_monitor.sh

# Terminal 3: load
python concurrent_requests.py --concurrency 20 --total 100

# Analyse
python plot_memory.py
```

**Deliverable:** Memory plot PNG + TTFT P99 before/after table

### Project 4 — CPU-to-GPU Pipeline Flamegraph

```bash
cd 06_capstone_projects/project4_flamegraph/

# Profile slow version
bash profile_flamegraph.sh slow

# Profile fast version
bash profile_flamegraph.sh fast

# Generate differential flamegraph
bash diff_flamegraph.sh
```

**Deliverable:** `diff.svg` differential flamegraph (the portfolio artifact)

---

## Interview Preparation

See [`docs/INTERVIEW_PREP.md`](INTERVIEW_PREP.md) for full question list.

**Priority questions to answer fluently:**

**GPU Architecture:**
- What causes warp divergence and how do you detect it?
- Difference between memory-bound and compute-bound kernels?
- What is the roofline model and how do you use it?

**AI Runtime:**
- How does KV cache work in autoregressive generation?
- Why does continuous batching improve throughput?
- What is PagedAttention and what problem does it solve?

**Systems:**
- Why does NUMA topology matter for GPU workloads?
- What causes cache misses and how do you measure them?
- How do flamegraphs work?

**Distributed:**
- Why does NCCL matter?
- What are the tradeoffs of tensor vs pipeline parallelism?
- How does FSDP differ from DDP?

---

## Priority Order (if time is limited)

1. **GPU profiling** (`nsys`, `ncu`, `torch.profiler`) — highest ROI
2. **PyTorch optimisation** (`torch.compile`, FP16, AMP) — critical
3. **vLLM + TensorRT** — very high for inference roles
4. **perf + eBPF + flamegraphs** — high for systems roles
5. **Distributed inference** — NCCL, tensor parallelism
6. **Workload characterisation** — benchmarking methodology
7. **CUDA programming** — kernels, occupancy
8. **Compiler/runtime internals** — nice to have
