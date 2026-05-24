# AI Systems Performance Engineering
## Course Outline — Kindle Direct Publishing Edition

**Subtitle:** GPU Profiling, LLM Optimisation, and Production Benchmarking for Engineers

**Target Reader:** Enterprise and cloud performance engineers transitioning into AI infrastructure and GPU systems roles. You already understand distributed systems, observability, and performance methodology. This book adds GPU systems depth.

**Companion Repository:** `ai-perf-engineering-book` — exercise files, reference scripts, and capstone projects referenced throughout.

**Estimated Length:** 380–430 pages  
**Format:** Technical non-fiction, code-heavy, hands-on

---

## Book Structure at a Glance

| Part | Title | Chapters | Est. Pages |
|------|-------|----------|------------|
| I    | Foundations | 1–3 | 55 |
| II   | GPU Programming & Profiling | 4–7 | 110 |
| III  | Linux Systems Profiling | 8–9 | 60 |
| IV   | LLM Inference Systems | 10–13 | 95 |
| V    | Workload Benchmarking | 14–15 | 45 |
| VI   | Capstone Projects | 16–19 | 70 |
| —    | Appendices | A–C | 30 |
| **Total** | | **19 chapters + 3 appendices** | **~465 pages** |

---

## PART I — Foundations

### Chapter 1: What Is AI Performance Engineering?

**Learning Objectives**
- Distinguish AI performance engineering from classical performance engineering
- Understand the hardware stack: CPU → PCIe → GPU → HBM
- Recognise the three root causes of all AI performance problems: compute-bound, memory-bandwidth-bound, and I/O-bound

**Topics**
1. The AI Infrastructure Stack
   - CPU + DRAM → PCIe bus → GPU + HBM
   - Why your CPU profiling intuition doesn't transfer directly
2. The Roofline Model — one diagram that explains everything
   - Arithmetic intensity: FLOPs per byte
   - Where transformers sit on the roofline
3. What Performance Engineers Do Differently in AI Teams
   - Profiling hierarchy: system → GPU → kernel
   - The four tools you will use on every workload: `nsys`, `ncu`, `torch.profiler`, `perf`
4. How to Use This Book and the Companion Repository

**Code / Exercises**
- `shared/models/model.py` — reference model used throughout the book
- Run: `python -c "import torch; print(torch.cuda.get_device_name(0))"` — verify environment

**Est. Pages:** 18

---

### Chapter 2: PyTorch Fundamentals

**Learning Objectives**
- Create and manipulate tensors on CPU and GPU
- Understand autograd and when NOT to use it
- Build a complete training loop from scratch
- Time GPU operations correctly with CUDA events

**Topics**
1. Tensors — the unit of work
   - Creation, shapes, dtypes, devices
   - Why `float16` uses half the memory of `float32`
   - CPU↔GPU movement and its cost
2. Autograd — how PyTorch computes gradients
   - The computation graph and `grad_fn`
   - `torch.no_grad()` — your inference best friend
   - The gradient accumulation trap (and how to avoid it)
3. `nn.Module` — building models
   - `train()` vs `eval()` — why forgetting this breaks inference
   - Saving and loading state dicts
4. The Complete Training Loop
   - Dataset and DataLoader
   - Forward → Loss → Backward → Step
   - Automatic Mixed Precision (AMP) with `GradScaler`
5. GPU Timing and Profiling Basics
   - Why `time.time()` lies about GPU performance
   - CUDA events — the correct GPU timer
   - Warmup runs — why the first iteration is always slow
   - `torch.profiler` — find slow ops in one command
   - GPU memory: allocated vs reserved vs peak

**Code / Exercises**
- `exercise_01_tensors.py` — 22 TODO blocks
- `exercise_02_autograd.py` — 16 TODO blocks
- `exercise_03_nn_modules.py` — 6 TODO blocks
- `exercise_04_training_loop.py` — 16 TODO blocks
- `exercise_05_performance_basics.py` — 8 TODO blocks

**Est. Pages:** 28

---

### Chapter 3: The AI Hardware Stack

**Learning Objectives**
- Explain how CPU architecture affects AI workloads
- Measure cache effects on workload performance
- Understand memory bandwidth as a first-class bottleneck

**Topics**
1. CPU Architecture for AI Engineers
   - Cache hierarchy: L1 / L2 / L3 / DRAM
   - NUMA — why CPU socket topology matters for dataloading
   - CPU instruction sets: AVX-512, AMX
2. PCIe — The Bridge Between CPU and GPU
   - PCIe bandwidth limits and when they matter
   - NVLink and GPU-to-GPU communication
3. GPU Hardware Fundamentals
   - Streaming Multiprocessors (SMs) and CUDA cores
   - HBM vs GDDR — bandwidth vs capacity tradeoff
   - Tensor Cores — the hardware behind `torch.float16`

**Code / Exercises**
- `01_phase1_foundations/exercise_01_tensors.py`
- `03_phase3_systems/module7_memory_numa/memory_bench.py` — measure cache effects

**Est. Pages:** 18

---

## PART II — GPU Programming and Profiling

### Chapter 4: The CUDA Execution Model

**Learning Objectives**
- Explain threads, warps, blocks, and grids
- Predict the performance impact of warp divergence
- Calculate and interpret SM occupancy
- Identify memory coalescing opportunities

**Topics**
1. The GPU Execution Model from First Principles
   - Thread → Warp (32 threads) → Block → Grid
   - SIMT: Single Instruction, Multiple Threads
2. Warp Divergence
   - What happens when threads branch differently
   - How to detect divergence with `ncu`
   - Practical patterns that cause and avoid divergence
3. SM Occupancy
   - Ratio of active warps to maximum warps
   - Registers and shared memory as occupancy limiters
   - Measuring occupancy: `ncu --metrics sm__occupancy`
4. Memory Coalescing
   - How 32 threads load memory in one transaction (stride-1)
   - The cost of non-contiguous access (stride > 1)
   - Coalescing in practice: row-major vs column-major layouts
5. GPU Memory Hierarchy
   - L1 cache (shared per SM) → L2 → HBM (global memory)
   - Shared memory: programmer-managed L1
   - Latency: registers (1 cycle) → L1 (20 cycles) → HBM (600+ cycles)
6. Tensor Cores
   - FP16 matrix multiply: 8× throughput vs FP32 CUDA cores
   - Requirements: shapes must be multiples of 8 (or 16)
   - How `torch.float16` and `torch.bfloat16` activate Tensor Cores

**Code / Exercises**
- `02_phase2_gpu/module3_cuda_fundamentals/vector_add.py` — observe the execution model
- `02_phase2_gpu/module3_cuda_fundamentals/cuda_kernels.py` — 6 experiments: occupancy, coalescing, divergence, Tensor Cores, memory hierarchy, kernel fusion
- `02_phase2_gpu/module3_cuda_fundamentals/occupancy_experiment.py`
- `02_phase2_gpu/module3_cuda_fundamentals/matmul_bench.py` — roofline analysis

**Est. Pages:** 32

---

### Chapter 5: GPU Profiling with nsys and ncu

**Learning Objectives**
- Capture a system-wide GPU timeline with `nsys`
- Drill into kernel-level hardware counters with `ncu`
- Identify memory-bound vs compute-bound kernels
- Read a Chrome trace / TensorBoard trace

**Topics**
1. The Profiling Hierarchy
   - Layer 1: `nsys` — system timeline (ms granularity)
   - Layer 2: `torch.profiler` — PyTorch op granularity
   - Layer 3: `ncu` — hardware counter granularity
   - When to use each tool
2. Nsight Systems (`nsys`)
   - `nsys profile --stats=true python train.py`
   - Reading the timeline: CPU threads, CUDA kernels, memory copies
   - Identifying idle GPU time and PCIe transfers
   - NVTX markers: annotate your own code sections
3. Nsight Compute (`ncu`)
   - `ncu --set basic --kernel-name gemm python train.py`
   - Key metrics: `dram__throughput`, `sm__throughput`, `l1tex__throughput`
   - The Roofline chart in `ncu` — is this kernel compute or memory bound?
   - Identifying uncoalesced access and low occupancy
4. `torch.profiler` — Python-level profiling
   - Recording CPU ops and CUDA kernels together
   - Exporting Chrome trace JSON and TensorBoard traces
   - The profiler schedule: wait/warmup/active
   - Reading the op table: `Self CUDA %` is your target

**Code / Exercises**
- `02_phase2_gpu/module4_profiling/train.py` — the profiling target
- `02_phase2_gpu/module4_profiling/profile_pytorch_infer.py` — all tools in one script
- `06_capstone_projects/project1_llm_opt/torch_profiler_trace.py`
- Shell scripts: `project1_llm_opt/profile_nsys.sh`, `profile_ncu.sh`

**Est. Pages:** 28

---

### Chapter 6: PyTorch Performance Optimisation

**Learning Objectives**
- Choose the right numeric precision for training vs inference
- Apply quantization and measure the accuracy / speed tradeoff
- Use `torch.compile` and understand what it does internally

**Topics**
1. Numeric Precision: FP32 vs FP16 vs BF16
   - Memory cost, throughput, and overflow risk
   - When to use each: `float32` (stable training), `float16` (inference), `bfloat16` (modern training)
   - AMP (`torch.autocast`) — let PyTorch choose automatically
2. Quantization
   - INT8 dynamic quantization — zero extra training, modest speedup
   - INT8 static quantization — requires calibration dataset
   - INT4 weight-only — aggressive compression for LLM deployment
   - Measuring accuracy degradation with perplexity
3. `torch.compile`
   - What it does: graph capture → TorchInductor → Triton kernel generation
   - Compilation modes: `default`, `reduce-overhead`, `max-autotune`
   - Warmup behaviour — compiled models need multiple warmup runs
   - When it helps (transformers) and when it doesn't (dynamic shapes)
4. Batch Size Optimisation
   - Why larger batches improve GPU utilisation
   - Memory vs throughput tradeoff
   - Finding the optimal batch size with a sweep

**Code / Exercises**
- `02_phase2_gpu/module5_pytorch_perf/fp16_bf16_bench.py`
- `02_phase2_gpu/module5_pytorch_perf/quantization_bench.py`
- `02_phase2_gpu/module5_pytorch_perf/llama_infer_optimize.py` — 5-step optimisation ladder
- `02_phase2_gpu/module4_profiling/optimize_batch_size.py`
- `06_capstone_projects/project1_llm_opt/torch_compile_bench.py`

**Est. Pages:** 28

---

### Chapter 7: DataLoader Optimisation

**Learning Objectives**
- Diagnose I/O bottlenecks that starve the GPU
- Configure `DataLoader` workers, prefetch, and pinned memory correctly
- Use `eBPF` and `iostat` to measure disk read patterns

**Topics**
1. The DataLoader Pipeline
   - How `num_workers`, `prefetch_factor`, and `pin_memory` interact
   - CPU augmentation vs GPU augmentation
   - The `.item()` anti-pattern — don't call it in the training inner loop
2. Diagnosing I/O Starvation
   - `nvidia-smi` idle GPU — the first symptom
   - `iostat` and `biolatency-bpfcc` — measure disk read latency
   - `opensnoop-bpfcc` — which files is the dataloader opening?
3. Pinned Memory and Non-Blocking Transfers
   - `pin_memory=True` — allocate host memory that GPU can DMA directly
   - `.to(device, non_blocking=True)` — overlap copy with compute
4. GPU Augmentation
   - `torchvision.transforms.v2` with GPU tensors
   - When to augment on CPU vs GPU

**Code / Exercises**
- `06_capstone_projects/project2_dataloader/slow_dataloader.py` — intentional bottleneck
- `06_capstone_projects/project2_dataloader/fast_dataloader.py` — optimised version
- `06_capstone_projects/project2_dataloader/diagnose_io.sh`

**Est. Pages:** 22

---

## PART III — Linux Systems Profiling

### Chapter 8: Perf, eBPF, and Flamegraphs

**Learning Objectives**
- Profile CPU time and hardware events with `perf`
- Write basic `bpftrace` one-liners to trace kernel events
- Generate and read CPU flamegraphs
- Identify lock contention in multi-threaded workloads

**Topics**
1. `perf` — The Swiss Army Knife of Linux Profiling
   - `perf stat` — count hardware events (cache misses, branch mispredictions)
   - `perf record` + `perf report` — statistical CPU profiling
   - Key hardware counters for AI: L1/L2/L3 miss rate, IPC
2. CPU Flamegraphs
   - What the x-axis and y-axis mean (width = time, height = call depth)
   - Generating: `perf record -g` → `stackcollapse-perf.pl` → `flamegraph.pl`
   - Reading: wide flat tops = hotspots; tall stacks = deep call chains
   - Differential flamegraphs: before vs after an optimisation
3. eBPF and `bpftrace`
   - What eBPF is and why it's safe for production
   - Key one-liners: file opens, syscall latency, block I/O
   - `opensnoop-bpfcc`, `biolatency-bpfcc`, `execsnoop-bpfcc`
4. Lock Contention
   - How Python's GIL affects multi-threaded DataLoaders
   - Detecting lock contention with `perf lock` and `bpftrace`

**Code / Exercises**
- `03_phase3_systems/module6_linux_profiling/cache_miss_analysis.py`
- `03_phase3_systems/module6_linux_profiling/lock_contention.py`
- `06_capstone_projects/project4_flamegraph/profile_flamegraph.sh`
- `06_capstone_projects/project4_flamegraph/diff_flamegraph.sh`

**Est. Pages:** 30

---

### Chapter 9: Memory Hierarchy and NUMA

**Learning Objectives**
- Measure the latency difference between L1, L2, L3, and DRAM
- Bind processes to NUMA nodes with `numactl`
- Diagnose and fix remote NUMA memory access

**Topics**
1. The Memory Hierarchy in Numbers
   - L1: ~4 cycles / L2: ~12 cycles / L3: ~40 cycles / DRAM: ~200 cycles
   - Bandwidth: L1 > L2 > L3 >> DRAM
   - Cache-friendly access patterns (stride-1 vs stride-N)
2. NUMA — Non-Uniform Memory Architecture
   - Why multi-socket servers have remote memory
   - Measuring NUMA effects: `numastat`, `perf stat -e node-stores`
   - Binding: `numactl --cpunodebind=0 --membind=0 python train.py`
3. Memory Bandwidth as a Bottleneck
   - How to measure peak bandwidth: `stream` benchmark
   - When your AI workload is memory-bandwidth bound
   - Prefetching and write-combining

**Code / Exercises**
- `03_phase3_systems/module7_memory_numa/memory_bench.py`
- `03_phase3_systems/module7_memory_numa/numa_workload.py`

**Est. Pages:** 22

---

## PART IV — LLM Inference Systems

### Chapter 10: LLM Inference Fundamentals

**Learning Objectives**
- Explain prefill vs decode phases and their different bottlenecks
- Implement and measure KV cache behaviour
- Calculate TTFT and TPS for a serving system

**Topics**
1. The Two Phases of LLM Inference
   - Prefill: process all input tokens in parallel — compute-bound
   - Decode: generate one token per step — memory-bandwidth-bound
   - Why they need different optimisation strategies
2. Key Metrics
   - TTFT (Time To First Token) — dominated by prefill latency
   - TPS (Tokens Per Second) — dominated by decode throughput
   - P50 / P99 latency — tail latency matters for user experience
3. The KV Cache
   - What it stores: Key and Value tensors for all past tokens
   - Why it grows linearly with sequence length and batch size
   - Memory pressure: what happens when the KV cache fills
   - Cache eviction strategies: full eviction, sliding window, PagedAttention

**Code / Exercises**
- `04_phase4_inference/module8_llm_systems/infer.py` — prefill + decode phases, timed separately
- `04_phase4_inference/module8_llm_systems/kv_cache_sim.py` — KV cache memory behaviour
- `06_capstone_projects/project1_llm_opt/baseline_inference.py` — establish your baseline numbers

**Est. Pages:** 25

---

### Chapter 11: Batching Strategies

**Learning Objectives**
- Implement continuous batching and explain why it beats static batching
- Measure throughput vs latency under different batch strategies
- Configure `vLLM` for a production serving scenario

**Topics**
1. Static Batching — the Naive Approach
   - Padding to the longest sequence wastes compute
   - GPU sits idle waiting for the slowest request in the batch
2. Continuous Batching
   - Insert new requests as soon as a slot frees — no idle waiting
   - The scheduler: max tokens in flight, preemption
   - Why vLLM's continuous batching changes throughput by 10–20×
3. PagedAttention
   - KV cache as virtual memory pages
   - Eliminates fragmentation: more requests fit in the same GPU memory
   - How vLLM implements it
4. Benchmarking a Serving System
   - `vllm benchmark_throughput.py` — offline (no latency target)
   - `vllm benchmark_serving.py` — online (QPS ramp test)

**Code / Exercises**
- `04_phase4_inference/module8_llm_systems/continuous_batching.py`
- `04_phase4_inference/module8_llm_systems/serve.py`
- `06_capstone_projects/project1_llm_opt/vllm_benchmark.sh`

**Est. Pages:** 22

---

### Chapter 12: Speculative Decoding

**Learning Objectives**
- Explain how speculative decoding reduces decode latency
- Measure the acceptance rate and its effect on throughput
- Identify the right draft model / target model pairing

**Topics**
1. The Decode Bottleneck Revisited
   - Each decode step loads all model weights from HBM — bandwidth-bound
   - Batching helps but latency still scales with steps
2. Speculative Decoding — the Core Idea
   - A small draft model proposes K tokens cheaply
   - The large target model verifies all K tokens in one forward pass
   - Accepted tokens: latency is K× better; rejected: no regression
3. Acceptance Rate and When It Works
   - Acceptance rate depends on draft/target alignment
   - Best case: same tokenizer, similar training distribution
   - Practical acceptance rates: 70–85% on code, 55–70% on open-ended text
4. Implementation Considerations
   - Draft model must be much smaller (3–10× fewer parameters)
   - Memory: both models must fit in GPU memory simultaneously

**Code / Exercises**
- `04_phase4_inference/module8_llm_systems/speculative_decode.py`

**Est. Pages:** 18

---

### Chapter 13: Distributed Inference

**Learning Objectives**
- Explain tensor parallelism, pipeline parallelism, and FSDP
- Measure NCCL collective performance
- Configure a two-GPU tensor-parallel inference setup

**Topics**
1. Why We Need Distributed Inference
   - Models larger than single GPU VRAM (70B needs ~140GB in FP16)
   - Latency reduction by splitting the model across GPUs
2. Tensor Parallelism
   - Split weight matrices across GPUs — each GPU holds a column shard
   - All-reduce after each layer to recombine activations
   - Communication overhead: only worthwhile with NVLink
3. Pipeline Parallelism
   - Split the model into stages — each GPU owns a set of layers
   - Micro-batching to hide pipeline bubbles
4. FSDP — Fully Sharded Data Parallel
   - Shard parameters, gradients, and optimizer state across GPUs
   - Full reconstruction only during forward/backward
   - When to use FSDP vs DDP vs tensor parallel
5. NCCL — The Communication Backbone
   - AllReduce, AllGather, ReduceScatter
   - Measuring NCCL bandwidth: `nccl-tests`
   - Bottleneck: PCIe (CPU-to-GPU) vs NVLink (GPU-to-GPU)

**Code / Exercises**
- `04_phase4_inference/module9_distributed/nccl_bench.py`
- `04_phase4_inference/module9_distributed/fsdp_train.py`
- `04_phase4_inference/module9_distributed/infer_distributed.py`

**Est. Pages:** 30

---

## PART V — Workload Benchmarking

### Chapter 14: Benchmarking Methodology

**Learning Objectives**
- Design a repeatable, statistically valid benchmark
- Build throughput–latency curves that reveal system limits
- Avoid the five most common benchmarking mistakes

**Topics**
1. The Five Benchmarking Mistakes
   - No warmup — first run is always slow
   - Using `time.time()` for GPU ops — measures CPU, not GPU
   - Single sample — no variance, no P99
   - Changing multiple variables at once — can't isolate cause
   - Reporting peak, not steady state
2. Benchmark Design
   - Define the workload: fixed input shape, fixed batch, representative prompt
   - Warmup: minimum 5 runs, more for compiled models
   - Measurement: CUDA events, 20+ iterations, report mean ± std
3. Throughput–Latency Curves
   - Sweep request rate from 0.1× to 2× system capacity
   - Plot RPS vs P50, P99 latency
   - Identify the "knee" — where latency starts degrading
4. Characterising a Workload
   - Arithmetic intensity: FLOPs / bytes of memory traffic
   - Compute utilisation: `sm__throughput.avg.pct_of_peak`
   - Memory bandwidth utilisation: `dram__throughput.avg.pct_of_peak`
   - Using the roofline model to classify your workload

**Code / Exercises**
- `05_phase5_workload/module11_benchmarking/throughput_latency_curve.py`
- `05_phase5_workload/module11_benchmarking/workload_characterize.py`

**Est. Pages:** 22

---

### Chapter 15: Porting a Workload

**Learning Objectives**
- Migrate an existing CPU workload to GPU with minimal code changes
- Identify the porting bottlenecks: precision, ops, memory layout
- Measure the CPU↔GPU bandwidth cost during porting

**Topics**
1. The Porting Checklist
   - Replace NumPy arrays with CUDA tensors
   - Replace CPU ops with `torch.*` equivalents
   - Check dtype compatibility: `float64` → `float32`
   - Profile CPU → GPU data transfers
2. Common Porting Pitfalls
   - `.numpy()` calls inside loops force CPU→GPU→CPU round-trips
   - `torch.quantization` deprecated APIs — migrate to `torchao`
   - Batch dimension matters: GPU prefers large batch × small op vs small batch × large op
3. DataLoader Parallelism at Scale
   - Tuning `num_workers` for your I/O subsystem
   - CPU affinity and NUMA binding for workers
   - Measuring worker utilisation

**Code / Exercises**
- `05_phase5_workload/module10_porting/workload_port.py`
- `05_phase5_workload/module10_porting/dataloader_worker.py`

**Est. Pages:** 20

---

## PART VI — Capstone Projects

*Each capstone is a self-contained portfolio project: establish a baseline, profile, optimise, and document the before/after comparison.*

---

### Chapter 16: Capstone 1 — LLM Inference Optimisation Lab

**Goal:** Take a GPT-2 baseline from ~80 tok/s to 150+ tok/s using the full optimisation stack.

**Workflow**
1. Establish baseline: `baseline_inference.py` — record TTFT, TPS, peak VRAM
2. Profile with `nsys` — identify idle time and copy overhead
3. Profile with `torch.profiler` — find the top 3 ops by CUDA time
4. Apply optimisations in order:
   - FP16 precision → FP16 + `torch.compile` default → `torch.compile max-autotune`
5. Benchmark each mode: `torch_compile_bench.py`
6. Document speedup table: eager vs compiled, P50/P99 latency, compile time

**Code / Exercises**
- `06_capstone_projects/project1_llm_opt/`
- `baseline_inference.py`, `torch_compile_bench.py`, `torch_profiler_trace.py`
- `profile_nsys.sh`, `profile_ncu.sh`, `vllm_benchmark.sh`

**Est. Pages:** 18

---

### Chapter 17: Capstone 2 — DataLoader I/O Bottleneck Hunt

**Goal:** Diagnose why `slow_dataloader.py` runs at 30% GPU utilisation, then fix it to reach 90%+.

**Workflow**
1. Run `slow_dataloader.py` — observe GPU utilisation with `nvidia-smi`
2. Diagnose with `diagnose_io.sh` — `iostat`, `biolatency`, `opensnoop`
3. Identify the bottlenecks: single worker, no pin_memory, CPU augmentation blocking
4. Apply fixes in `fast_dataloader.py`: 8 workers, pin_memory, GPU augmentation
5. Compare throughput: batches/sec before and after

**Code / Exercises**
- `06_capstone_projects/project2_dataloader/`
- `slow_dataloader.py`, `fast_dataloader.py`, `diagnose_io.sh`

**Est. Pages:** 16

---

### Chapter 18: Capstone 3 — KV Cache Memory Pressure Experiment

**Goal:** Measure GPU memory behaviour as concurrent requests fill the KV cache; identify the OOM threshold.

**Workflow**
1. Start the vLLM server: `kv_pressure_server.sh`
2. Monitor GPU memory: `memory_monitor.sh` → `reports/gpu_memory_log.csv`
3. Generate concurrent load: `concurrent_requests.py` (ramp from 1 to 32 concurrent)
4. Visualise: `plot_memory.py` → annotated memory vs time graph
5. Interpret: at what concurrency does memory hit 90%? When does latency spike?

**Code / Exercises**
- `06_capstone_projects/project3_kv_cache/`
- `concurrent_requests.py`, `memory_monitor.sh`, `plot_memory.py`

**Est. Pages:** 18

---

### Chapter 19: Capstone 4 — CPU-to-GPU Pipeline Flamegraph Challenge

**Goal:** Use flamegraphs to find and fix three hidden bottlenecks in `slow_training.py`.

**Workflow**
1. Profile `slow_training.py` with `perf` + flamegraph
2. Read the flamegraph: identify the top 3 wide stacks
3. Reproduce each bottleneck and understand the root cause
4. Apply fixes in `fast_training.py`: data pinning, non-blocking transfer, gradient accumulation
5. Generate a differential flamegraph: `diff_flamegraph.sh` — before vs after side-by-side

**Code / Exercises**
- `06_capstone_projects/project4_flamegraph/`
- `slow_training.py`, `fast_training.py`, `profile_flamegraph.sh`, `diff_flamegraph.sh`

**Est. Pages:** 18

---

## APPENDICES

### Appendix A: Environment Setup

- Docker-based environment (NVIDIA NGC PyTorch 25.01)
- Installing `nsys`, `ncu`, `nvitop`, `bpfcc-tools`
- WSL2 and Windows considerations
- RTX 4060 specific notes

*File: `docs/HARDWARE_SETUP.md`, `scripts/setup_env.sh`, `Dockerfile`*

**Est. Pages:** 10

---

### Appendix B: Command Reference

*Every profiling command used in the book, grouped by tool.*

| Tool | Command | Purpose |
|------|---------|---------|
| `nsys` | `nsys profile --stats=true python train.py` | Full GPU timeline |
| `ncu` | `ncu --set basic --kernel-name gemm python train.py` | Kernel counters |
| `nvidia-smi` | `nvidia-smi dmon -s u` | Real-time GPU utilisation |
| `nvitop` | `nvitop` | Rich GPU dashboard |
| `perf` | `perf stat -e cache-misses python workload.py` | CPU hardware events |
| `bpftrace` | `opensnoop-bpfcc` | File open tracing |
| `numactl` | `numactl --cpunodebind=0 --membind=0 python train.py` | NUMA binding |

*File: `docs/COMMANDS.md`*

**Est. Pages:** 12

---

### Appendix C: Interview Preparation

*50 questions categorised by topic, with model answers.*

**Categories**
1. GPU Architecture (10 questions) — warps, occupancy, memory hierarchy
2. LLM Inference (10 questions) — KV cache, continuous batching, speculative decoding
3. Profiling Tools (10 questions) — nsys vs ncu vs torch.profiler
4. Distributed Systems (10 questions) — NCCL, FSDP, tensor vs pipeline parallelism
5. Benchmarking (10 questions) — roofline model, arithmetic intensity, P99 vs P50

*File: `docs/INTERVIEW_PREP.md`*

**Est. Pages:** 18

---

## Front Matter

| Section | Content |
|---------|---------|
| **Cover** | Title, subtitle, author name |
| **Copyright Page** | Copyright, ISBN, disclaimer |
| **Dedication** | Optional |
| **Preface** | Why this book exists; who it is for; what you need before starting |
| **How to Use This Book** | How the companion repo connects to each chapter; suggested reading paths |
| **Table of Contents** | All chapters and appendices |

---

## Suggested Reading Paths

**Path A — Fast Track (GPU Profiling focus, ~4 weeks)**
Chapters 1 → 2 → 4 → 5 → 6 → 16 → Appendix B

**Path B — LLM Inference focus (~6 weeks)**
Chapters 1 → 2 → 4 → 5 → 10 → 11 → 12 → 13 → 16 → 18

**Path C — Full Course (~5 months part-time)**
All chapters in order, completing all exercise files and capstone projects

---

## KDP Publishing Notes

- **Category suggestions:** Computer Science > AI & Machine Learning; Engineering > Computer Hardware
- **Keywords:** GPU profiling, LLM inference, PyTorch performance, CUDA programming, AI infrastructure
- **Price range:** $34.99–$44.99 (technical non-fiction, code-heavy)
- **Format:** Kindle + Paperback; consider KDP Print for the command reference appendix
- **Page count estimate:** 430–465 pages (6×9 inch trim, 11pt body)
