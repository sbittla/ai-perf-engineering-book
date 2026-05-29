#!/usr/bin/env python3
"""
Appendices/C.Interview_Prep/C.1_interview_questions.py  —  Appendix C: Interview Preparation

50 interview questions with model answers, across five categories:
  1. GPU Architecture         (Q1–10)
  2. LLM Inference Systems    (Q11–20)
  3. Profiling Tools          (Q21–30)
  4. Distributed Systems      (Q31–40)
  5. Benchmarking             (Q41–50)

Usage:
  python Appendices/C.Interview_Prep/C.1_interview_questions.py           # print all Q&A
  python Appendices/C.Interview_Prep/C.1_interview_questions.py --quiz    # self-quiz mode (no answers)
  python Appendices/C.Interview_Prep/C.1_interview_questions.py --cat 2   # category 2 only
"""

import sys
import argparse
import textwrap

# ──────────────────────────────────────────────────────────────────────────────
# Question bank
# ──────────────────────────────────────────────────────────────────────────────

QA = [
    # ── Category 1: GPU Architecture ─────────────────────────────────────────
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is warp divergence and how do you detect it?",
        "a": """\
When threads in a 32-thread warp take different code paths (if/else), the warp
serialises both paths with inactive threads masked off — halving throughput per
divergent path.

Detect with:
  ncu --metrics sm__sass_average_branch_targets_threads_uniform.pct script.py
Low uniform branch % → high divergence.

Fix: restructure branches so entire warps take the same path, or rearrange data
to group similar inputs."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is the difference between compute-bound and memory-bound kernels?",
        "a": """\
Compute-bound: FLOPs are the bottleneck — GPU ALUs are always busy, throughput
scales with peak FP16 tensor core speed.
Memory-bound: data loading from HBM/GDDR6 is the bottleneck — ALUs idle waiting
for reads.

Measure:
  ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,
               sm__throughput.avg.pct_of_peak_sustained_elapsed script.py
  dram≈100%, sm<50% → memory-bound
  sm≈100%, dram<50% → compute-bound

LLM decode (batch=1) is almost always memory-bound. Prefill at large batch is
compute-bound."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "Explain the roofline model.",
        "a": """\
Two ceilings on achievable throughput (FLOP/s):
  1. Memory BW ceiling:  throughput ≤ bandwidth_GBs × arithmetic_intensity
  2. Compute ceiling:    throughput ≤ peak_FLOPS

Ridge point = peak_FLOPS / bandwidth_GBs — the minimum arithmetic intensity
(FLOPs/byte) to be compute-bound.

Left of the ridge → memory-bound; fix by reusing data (tiling, fusing ops).
Right of the ridge → compute-bound; need a faster GPU or better kernel.

On an RTX 4060: peak FP16 ≈ 136 TFLOP/s, bandwidth ≈ 272 GB/s →
ridge point ≈ 500 FLOP/byte."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is GPU occupancy and why does it matter?",
        "a": """\
Occupancy = active warps on an SM / maximum warps the SM can hold simultaneously.

High occupancy lets the SM hide memory latency: when one warp stalls waiting for
data, the scheduler switches to another ready warp.
Low occupancy → SM idles while waiting for the stalled warp.

Measure:
  ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active script.py

Occupancy limiters: register count per thread (fewer threads can fit), shared
memory usage per block. Not always the bottleneck — some kernels achieve high
throughput at low occupancy if arithmetic intensity is high."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is memory coalescing?",
        "a": """\
When 32 warp threads access 32 contiguous addresses (stride-1), the GPU hardware
coalesces them into ONE 128-byte memory transaction. With stride > 1, each thread
needs a separate transaction → wasted bandwidth proportional to stride.

Check:
  ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum script.py

Fix non-coalesced access by transposing data layouts, using shared memory for
intermediate storage (tiling), or restructuring the access pattern."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What are Tensor Cores and when does PyTorch use them?",
        "a": """\
Tensor Cores are dedicated matrix multiply-accumulate units introduced in Volta
(CC 7.0). They accelerate FP16 GEMM by 8× over CUDA cores.

PyTorch activates them when:
  • Tensor dtype is float16 or bfloat16
  • Matrix dimensions are multiples of 8 (or 16 for some operations)
  • TF32 mode is enabled (default on A100+): still uses FP32 accumulators

Check:
  ncu --metrics sm__sass_thread_inst_executed_op_hfma_pred_on.avg script.py"""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is the difference between L1, L2, and HBM on a GPU?",
        "a": """\
L1 cache (shared memory): ~64–192KB per SM, programmer-managed in CUDA C,
~1 cycle latency. Holds tiles for tiled matmuls, reused values.

L2 cache: shared across all SMs, ~40–50MB on recent GPUs, ~10-20 cycle latency.
Caches frequently-accessed global memory.

HBM (High Bandwidth Memory): off-chip DRAM, 40–3200GB depending on GPU tier,
~600 cycle latency, 272–3500 GB/s bandwidth.

Most AI workloads are HBM-bandwidth-limited. Every kernel-level optimisation
(tiling, fusion, reuse) aims to maximise computation per byte of HBM access."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is SM occupancy vs SM utilisation, and why are they different?",
        "a": """\
SM occupancy: ratio of active warps to peak-possible warps. A scheduling
metric — measures how well the SM can hide latency.

SM utilisation (sm__throughput.avg.pct_of_peak_sustained_elapsed): fraction
of cycles where at least one instruction is executing. A throughput metric.

A kernel can have low occupancy but high utilisation if its arithmetic intensity
is high enough that the SM never stalls (e.g., small, register-heavy kernels).
A kernel can have high occupancy but low utilisation if warps frequently stall
waiting for memory.

Profile both and cross-reference."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is PCIe bandwidth and when does it become a bottleneck?",
        "a": """\
PCIe 4.0 x16 provides ~32 GB/s bidirectional per slot; PCIe 3.0 x16 ≈ 16 GB/s.
Consumer GPUs often use x8 (RTX 4060: PCIe 4.0 x8 → ~16 GB/s).

PCIe becomes the bottleneck when:
  • H2D data transfers are frequent (DataLoader without pin_memory)
  • Model weights must be moved per inference step
  • Multi-GPU training uses PCIe instead of NVLink for gradient sync

Fix: pin_memory=True (faster DMA), non_blocking=True H2D, keep tensors on GPU
across iterations, use NVLink for multi-GPU setups."""
    },
    {
        "cat": 1, "cat_name": "GPU Architecture",
        "q": "What is TF32 and how does it affect training accuracy?",
        "a": """\
TF32 (TensorFloat-32) is a format internal to Tensor Cores on Ampere+ GPUs.
It uses the 8-bit exponent of FP32 and the 10-bit mantissa of FP16.

When TF32 is enabled (default on A100+), PyTorch runs FP32 matmuls with TF32
Tensor Cores internally while keeping FP32 accumulation — giving 3–8× speedup
over naive FP32 CUDA cores with < 0.1% accuracy degradation on most models.

To disable:
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

Always keep TF32 enabled unless you need IEEE FP32 precision (scientific work)."""
    },

    # ── Category 2: LLM Inference Systems ────────────────────────────────────
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "Explain prefill vs decode phases and their different bottlenecks.",
        "a": """\
Prefill: process all input tokens in one parallel forward pass. Large batch of
tokens → large GEMM → compute-bound (Tensor Cores near 100%).
Latency dominated by TTFT (Time To First Token).

Decode: generate one token per step, autoregressive. Batch=1 → GEMM degenerates
to matrix-vector product → memory-bandwidth-bound (reads all weights each step).
Throughput dominated by how fast HBM can deliver KV cache + weights.

They need different optimisation strategies:
  Prefill:  larger batch, chunked prefill, FlashAttention
  Decode:   continuous batching, KV quantisation, speculative decoding"""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "How does the KV cache work and what is its memory formula?",
        "a": """\
During prefill, the transformer computes and caches Keys (K) and Values (V) for
every input token. In decode, each new token computes only its own Q and attends
over all cached K, V — avoiding recomputation of past context.

Memory formula:
  KV bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes

Llama-7B FP16, 2048-token context, 1 request:
  2 × 32 × 32 × 128 × 2048 × 1 × 2 = 1 GB

At 8 requests: 8 GB plus ~14 GB weights = 22 GB → exceeds 24 GB GPU on large
batches. KV memory scales linearly with seq_len and batch."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What is continuous batching and why does it outperform static batching?",
        "a": """\
Static batching (HuggingFace generate()): all requests in a batch must finish
before new ones can join. Short requests pad with zeros until the longest
finishes → wasted GPU compute proportional to padding fraction.

Continuous batching (vLLM): when a request finishes a decode step, it is
immediately removed and a new request is inserted. No padding, no waiting.
GPU is always doing real work.

Typical improvement: 3–10× throughput on variable-length request distributions.
Works because decode is sequential — inserting at a step boundary costs nothing."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What is PagedAttention?",
        "a": """\
PagedAttention (vLLM) treats the KV cache like virtual memory.

Problem: pre-allocating max_seq_len × n_layers × ... per request wastes memory
when most requests are shorter than the maximum (internal fragmentation).

Solution: divide KV memory into fixed-size pages (e.g., 16 tokens each),
allocated on demand. Logical KV pages are mapped to physical blocks via a page
table — identical to OS virtual memory management.

Benefits:
  • Near-zero internal fragmentation
  • Prefix caching: share pages for common system prompts across requests
  • Enables more concurrent requests than preallocated approaches"""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "Explain speculative decoding and when it helps.",
        "a": """\
A small draft model generates K candidate tokens cheaply. The large target model
verifies all K tokens in ONE parallel forward pass.

If all K accepted: K tokens generated for roughly the cost of 1 target step →
K× latency improvement. If rejected at position j: tokens 0..j-1 are accepted
and generation continues from j.

Acceptance rate (α) and K determine speedup:
  expected_tokens_per_step = (1 - α^(K+1)) / (1 - α)
  At α=0.8, K=5: ~3.6× improvement

Works when: draft and target have the same tokenizer and aligned distributions
(e.g., CodeLlama-7B drafting for Llama-70B on code tasks).
Fails when: distributions diverge (open-ended generation, α < 0.6)."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What are the tradeoffs of FP16 vs INT8 vs INT4 inference?",
        "a": """\
              FP16        INT8         INT4 (GPTQ/AWQ)
Memory      2B/weight   1B/weight    0.5B/weight
Accuracy    Near-exact  <0.5% loss   1–3% loss (task-dependent)
Speed       Tensor Core INT8 TC      Dequant overhead on most GPUs
Compat      Universal   Calibration  Weight-only (activations stay FP16)
VRAM        Baseline    50% saving   75% saving

Rule of thumb:
  FP16   → fits in VRAM, highest quality
  INT8   → large model on small GPU, modest quality trade
  INT4   → 70B models on consumer GPUs; recheck perplexity"""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What limits LLM decode throughput as batch size grows?",
        "a": """\
Three limits that saturate in sequence as batch grows:

1. HBM bandwidth: at batch=1, decode is weight-read-limited. Adding requests
   amortises weight reads but increases KV traffic linearly.

2. KV cache memory: each new concurrent request adds KV bytes proportional to
   seq_len. The GPU OOMs when kv_cache_bytes + weights > VRAM.

3. Compute (prefill): at large batch, prefill becomes the bottleneck before
   decode, shifting the system from memory-bound to compute-bound.

Practical tool: nvitop while running vLLM shows which resource saturates first."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "Why does decode latency scale linearly with context length?",
        "a": """\
Each decode step reads the ENTIRE KV cache from HBM — all n_layers × n_heads ×
head_dim × current_seq_len bytes. As seq_len grows, the HBM read per step grows
proportionally.

Latency model:
  bytes_per_step = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes
  latency_ms ≈ bytes_per_step / (bandwidth_GB_s × 1e9) × 1000

At 2048 tokens vs 512 tokens, each decode step takes 4× longer.
This is why long-context inference is slow — and why KV quantisation and
sliding window attention are important at large seq_len."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What is chunked prefill and when should you use it?",
        "a": """\
Without chunked prefill, a single long prefill (e.g., 4096 tokens) monopolises
the GPU for hundreds of milliseconds, stalling decode for other requests.

Chunked prefill splits a long prefill into fixed-size chunks (e.g., 512 tokens
per iteration). Each iteration interleaves a prefill chunk with decode steps for
other requests, bounding TTFT tail latency.

vLLM flag: --enable-chunked-prefill --max-num-batched-tokens 2048

Use when: mix of short (interactive) and long (RAG, summarisation) requests in
the same serving pool. Latency-sensitive short requests benefit; throughput for
long requests is slightly reduced due to the interleaving overhead."""
    },
    {
        "cat": 2, "cat_name": "LLM Inference Systems",
        "q": "What is TTFT vs TPS vs E2E latency?",
        "a": """\
TTFT (Time To First Token): time from request submission until the first output
token arrives. Dominated by prefill compute. Target < 500ms for interactive use.

TPS (Tokens Per Second per request): decode throughput measured at the client.
Determined by decode latency × tokens to generate. Target > 20 tok/s visible.

E2E latency (end-to-end): TTFT + (n_output_tokens / TPS). Total time a user
waits. What SLOs are written against.

Monitor:
  vllm benchmark_serving.py → outputs P50/P99 TTFT, P50/P99 TPS, E2E latency"""
    },

    # ── Category 3: Profiling Tools ───────────────────────────────────────────
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "When do you use nsys vs ncu vs torch.profiler?",
        "a": """\
nsys (Nsight Systems): system-level GPU timeline. Use first to find WHICH
kernel or transfer category is slow. Millisecond granularity. Shows CPU/GPU
overlap, H2D transfers, idle time. Cost: ~5% overhead.

torch.profiler: PyTorch op-level attribution. Bridges Python ops to CUDA kernels.
Use to find which PyTorch layer (embedding, attention, FFN) is the hotspot, and
to export Chrome traces for visual inspection.

ncu (Nsight Compute): hardware counter granularity inside a specific kernel.
Use last, once you know exactly which kernel to optimise. Measures occupancy,
DRAM utilisation, shared memory efficiency. Cost: 10–100× slowdown."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "How do you annotate your own code regions in a profiler trace?",
        "a": """\
torch.profiler: use record_function as a context manager
  with torch.profiler.record_function("my_region"):
      # code here

The region appears as a named op in key_averages() and in Chrome traces.

nsys: use NVTX markers
  import nvtx
  with nvtx.annotate("my_region", color="blue"):
      # code here

  Or:  torch.cuda.nvtx.range_push("my_region") / range_pop()

These appear as coloured bands on the nsys GPU timeline."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What does 'self_cuda_time_total' mean in torch.profiler output?",
        "a": """\
The torch.profiler table shows two time columns:
  cuda_time_total: total CUDA time including time spent in child ops
  self_cuda_time_total: CUDA time attributed to THIS op, excluding children

The difference matters for fused ops and nested calls. An embedding layer may
show high cuda_time_total because it includes the memory transfer, but
self_cuda_time_total reveals the true kernel cost.

Sort by self_cuda_time_total to find genuine hotspots:
  prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15)"""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What is the profiler schedule pattern and why does it matter?",
        "a": """\
torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

wait: skip N iterations (let JIT compile, data cache warm)
warmup: start profiling but discard data (HW counters need a few steps to settle)
active: record these N iterations (the actual data you want)
repeat: how many wait/warmup/active cycles before stopping

Without scheduling, profiling starts immediately (including JIT compilation)
and the overhead changes the workload you're measuring. Always skip at least
2 warmup iterations; 5 is safer for compiled models."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "How do you generate a differential flamegraph?",
        "a": """\
A differential flamegraph shows which functions got faster (blue) or slower (red)
between two profiles.

Steps:
  1. py-spy record -o before.folded --format raw -- python slow_train.py
  2. py-spy record -o after.folded  --format raw -- python fast_train.py
  3. ~/FlameGraph/difffolded.pl before.folded after.folded \\
         | ~/FlameGraph/flamegraph.pl > diff.svg

Open diff.svg in a browser. Red stacks = more time after the change (regression).
Blue stacks = less time after the change (improvement). Width = time magnitude.

Use to confirm that a fix actually reduced time in the expected function."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What are NVTX markers and how do they appear in nsys?",
        "a": """\
NVTX (NVIDIA Tools eXtension) lets you inject named ranges and instant markers
into the CUDA timeline. nsys captures them and displays them as coloured bands
overlaid on the GPU kernel timeline.

Usage:
  torch.cuda.nvtx.range_push("forward_pass")
  output = model(x)
  torch.cuda.nvtx.range_pop()

  # Or with the nvtx package:
  with nvtx.annotate("attention", color="blue"):
      attn_out = self_attention(x)

In nsys-ui you can visually align PyTorch ops (NVTX bands) with the exact CUDA
kernels they launch — critical for identifying launch overhead."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What does high 'iowait' mean and how do you fix it?",
        "a": """\
iowait (wa column in vmstat) = fraction of CPU time where the CPU is idle AND
there is at least one outstanding I/O request. High iowait (>20%) means the
storage system is the bottleneck, not the CPU or GPU.

In AI training context:
  DataLoader reads images/tokens from disk → wa spikes → GPU starved.

Diagnosis:
  vmstat 1 | awk '{print $16}'       # wa column
  iostat -xz 1                        # disk %util

Fix:
  1. num_workers > 0 (parallel prefetch)
  2. pin_memory=True (faster H2D after prefetch)
  3. Cache dataset to RAM (tmpfs) or local NVMe
  4. Preprocess and save as memory-mapped tensors (.pt shards)"""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "How do you measure GPU idle time during training?",
        "a": """\
GPU idle time (DataLoader starvation) is measured as the fraction of total wall
time not attributed to GPU compute:

  idle_pct = (wall_time_ms - gpu_time_ms) / wall_time_ms × 100

In code:
  start = time.perf_counter()
  for batch in loader:
      # DataLoader fetch time is included in wall_time
      gpu_t0 = perf_counter_after_sync()
      model(batch)
      torch.cuda.synchronize()
      gpu_time += time since gpu_t0
  wall_time = time.perf_counter() - start
  idle_pct = (wall_time - gpu_time) / wall_time * 100

idle_pct > 10% → DataLoader is the bottleneck."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What does `python -m torch.utils.bottleneck` report?",
        "a": """\
torch.utils.bottleneck runs your script with three profilers simultaneously:
  1. cProfile (Python-level call counts and time)
  2. torch.autograd.profiler (CPU op timing)
  3. torch.autograd.profiler with CUDA events (GPU op timing)

And prints a combined report sorted by bottleneck.

Usage:
  python -m torch.utils.bottleneck train.py --epochs 1

Useful as a first-pass diagnostic before committing to a full nsys session.
Does NOT support record_shapes or Chrome trace export — use torch.profiler for
those. Output is text-only."""
    },
    {
        "cat": 3, "cat_name": "Profiling Tools",
        "q": "What is the .item() synchronisation trap?",
        "a": """\
tensor.item() forces a GPU→CPU synchronisation: PyTorch must wait for all
pending GPU work to finish, copy the scalar value to CPU, and return it.

If called inside the training loop (e.g., to accumulate a running loss), it
serialises the CPU and GPU — the GPU sits idle while Python does the copy.

Wrong:
  running_loss += loss.item()   # sync every step

Fix — accumulate on GPU, sync once per epoch:
  running_loss_tensor = torch.tensor(0.0, device=device)
  running_loss_tensor += loss.detach()   # stays on GPU
  epoch_loss = running_loss_tensor.item() / n_batches  # one sync

Other sneaky syncs: tensor.tolist(), print(tensor), logging tensor values."""
    },

    # ── Category 4: Distributed Systems ──────────────────────────────────────
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "Why does NCCL matter for multi-GPU training?",
        "a": """\
NCCL (NVIDIA Collective Communications Library) provides optimised collective
operations (AllReduce, AllGather, ReduceScatter) that use NVLink (GPU-to-GPU),
PCIe, and InfiniBand — keeping data on GPU throughout.

Without NCCL: gradients would go GPU → CPU → network → CPU → GPU each step —
2–4× slower than NVLink AllReduce.

Key metrics:
  AllReduce bandwidth = 2(n-1)/n × message_bytes / time
  On NVLink: can reach 300+ GB/s bidirectional (A100 SXM)
  On PCIe 4.0 x16: ~32 GB/s → gradient sync dominates for large models

Check bandwidth: nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 2"""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "What are the tradeoffs between tensor parallelism and pipeline parallelism?",
        "a": """\
                  Tensor Parallel         Pipeline Parallel
Splits       Each layer across GPUs    Layers to different GPUs (stages)
Communication AllReduce every layer   Point-to-point between stages
Latency      +AllReduce per layer     +pipeline bubble (1/n_stages efficiency)
Memory       1/N of every layer       N/total_layers per GPU
NVLink need  Yes (high bandwidth)     No (low bandwidth OK)
Best for     Wide layers (attention)  Sequential layer stacks

In practice: Megatron-LM uses 3D parallelism = tensor × pipeline × data.
Start with data parallelism; add tensor parallel when a single layer doesn't
fit in VRAM; add pipeline parallel when even with TP the model is too large."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "How does FSDP differ from DDP?",
        "a": """\
DDP (DistributedDataParallel): every GPU holds the FULL model. After backward,
gradients are AllReduced across GPUs. Memory per GPU:
  ~params + grads + optimiser_states ≈ 3× params in FP32

FSDP (FullyShardedDataParallel): shards params, grads, AND optimiser states
across N GPUs. Before each forward, AllGather reconstructs the full layer on each
GPU. After backward, ReduceScatter distributes gradients back.
  Memory per GPU ≈ 3× params / N

FSDP enables training models larger than one GPU's VRAM. DDP is simpler and has
lower communication overhead for models that fit in GPU memory."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "How do you overlap communication and computation in distributed training?",
        "a": """\
DDP bucketing: PyTorch DDP groups parameters into buckets. As soon as a bucket's
gradients are computed during backward, AllReduce starts for that bucket — while
backward continues computing gradients for earlier layers. Communication and
computation overlap.

FSDP prefetching: AllGather for layer N+1 can be initiated while layer N is in
forward compute. Overlap requires enough GPU memory headroom for two layer shards.

In nsys timeline: look for NCCL rows (orange) overlapping with CUDA compute
rows (green). No overlap → communication is a serial bottleneck.

Tune: FSDP's forward_prefetch and backward_prefetch flags."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "What is the pipeline bubble and how is it reduced?",
        "a": """\
In pipeline parallelism, the first micro-batch must flow through ALL stages
before the last stage begins backward — during this ramp-up, stages 1..n-1 sit
idle. Similarly, there's a ramp-down at the end of backward. These idle periods
are the "pipeline bubble".

Bubble fraction ≈ (n_stages - 1) / n_micro_batches

Fix: increase n_micro_batches (chunk the batch more finely). At 4 stages and
8 micro-batches: bubble = 3/8 = 37.5%. At 16 micro-batches: 3/16 = 18.75%.

Interleaved pipeline schedules (1F1B, Chimera) further reduce the bubble."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "What NCCL collectives are used in which parallelism strategy?",
        "a": """\
AllReduce: sum gradients across all data-parallel replicas. Used in DDP.
  Cost: 2(n-1)/n × message_bytes / BW

AllGather: reconstruct sharded params on all GPUs. Used in FSDP forward pass.
  Cost: (n-1)/n × message_bytes / BW

ReduceScatter: scatter reduced gradients back to owning GPU. Used in FSDP backward.
  Cost: (n-1)/n × message_bytes / BW

AllGather + ReduceScatter ≈ AllReduce in total data moved, but allows computation
to overlap with scatter — which is why FSDP can hide communication better."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "When would you choose inference over training parallelism?",
        "a": """\
Training: large batch, repeat computation, can afford bubble. Pipeline parallel
suited. Gradient communication dominates → NCCL throughput critical.

Inference: latency-sensitive, variable request sizes, no gradient sync needed.
Tensor parallelism preferred (lowest per-token latency for a single request).

For a 70B model serving at <500ms TTFT on 4 × A100 80GB:
  • Tensor parallel 4-way: model split across GPUs, every request uses all 4
  • Pipeline parallel: would add pipeline latency per decode step → worse TTFT

For batch inference (offline): pipeline parallel acceptable — throughput matters
more than latency."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "How do you diagnose slow NCCL AllReduce?",
        "a": """\
1. Measure baseline:
   nccl-tests/build/all_reduce_perf -b 8 -e 512M -f 2 -g N
   Compare to theoretical NVLink / PCIe bandwidth.

2. Profile with nsys:
   NCCL_DEBUG=INFO nsys profile --trace=cuda,nvtx,nccl torchrun --nproc_per_node=N train.py
   Look for NCCL rows in the timeline; check overlap with compute.

3. Common causes:
   • PCIe instead of NVLink: nvidia-smi topo -m → confirm P2P via NV2 not PIX
   • Ring scheduling with odd topologies: set NCCL_SOCKET_IFNAME
   • Bucket size too small: DDP default 25MB; increase with bucket_cap_mb
   • OOM causing contiguous allocation fallback: check memory headroom"""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "What is gradient checkpointing and what is the memory/compute tradeoff?",
        "a": """\
Gradient checkpointing (activation recomputation): instead of storing all
intermediate activations during forward for use in backward, store only
checkpoint boundaries. Recompute activations between checkpoints during backward.

Memory: O(√n) instead of O(n) for n layers (checkpoint every √n layers)
Compute: +33% FLOPs (one extra forward per checkpointed segment)

Enable in PyTorch:
  torch.utils.checkpoint.checkpoint(model.forward, *inputs)

Use when: model fits in GPU but activation memory causes OOM at target batch size.
Note: incompatible with some custom CUDA ops; test correctness after enabling."""
    },
    {
        "cat": 4, "cat_name": "Distributed Systems",
        "q": "What is the difference between synchronous and asynchronous AllReduce?",
        "a": """\
Synchronous AllReduce: all GPUs wait at a barrier until all gradients are
aggregated. Simple to implement; all GPUs always train on the same model.
Used by PyTorch DDP by default.

Asynchronous AllReduce (Hogwild, local SGD): GPUs continue training while
communication happens in the background. Gradients are stale by the time they
are applied. Can be faster in wall clock but may converge more slowly.

In practice: synchronous is standard for LLM training (correctness matters).
Asynchronous is used in federated learning or extremely slow interconnect scenarios."""
    },

    # ── Category 5: Benchmarking ──────────────────────────────────────────────
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "What are the five most common benchmarking mistakes?",
        "a": """\
1. No warmup: first run triggers cuBLAS autotuning and JIT compilation →
   always slower than steady state. Minimum 5 warmup iterations; 20+ for compiled.

2. Using time.time() / time.perf_counter() for GPU ops: measures CPU submission
   time. GPU work is asynchronous. Use CUDA events and torch.cuda.synchronize().

3. Single sample: no variance estimation. One fast outlier (cache pre-warmed)
   or one slow outlier (thermal throttle) ruins the result. Run 20+ iterations.

4. Changing multiple variables at once: batch size AND precision AND workers.
   You can't attribute the speedup. Change ONE variable per experiment.

5. Reporting peak, not steady-state: burst throughput may be 2× sustained due
   to thermal throttling. Run for 60+ seconds and report the last 30 seconds."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "How do you build a throughput–latency curve and what does it reveal?",
        "a": """\
Sweep request rate (RPS) from very low (1/10 of capacity) to well above
saturation (2×). At each rate, measure:
  • Mean, P50, P95, P99 latency
  • Actual achieved throughput

Plot achieved RPS vs P99 latency.

What it reveals:
  • Linear region: system is unsaturated, latency is roughly constant
  • Knee: where latency starts rising steeply — this is maximum useful load
  • Saturation tail: queue grows unboundedly; latency spikes

The knee identifies your operating point. SLOs (e.g., P99 < 500ms) define the
safe operating range. Always provision for 2× the expected peak to stay left of
the knee."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "What is arithmetic intensity and how do you measure it for a kernel?",
        "a": """\
Arithmetic intensity (AI) = FLOPs executed / bytes transferred from DRAM.

High AI → compute-bound (benefits from faster FP16, fusion, Tensor Cores)
Low AI → memory-bound (benefits from reduced data movement, quantisation)

Measure with ncu:
  FLOPs:  sm__sass_thread_inst_executed_op_hfma_pred_on.sum × 2
  DRAM:   dram__bytes.sum (total bytes read+written to HBM)
  AI = FLOPs / DRAM_bytes

For Llama-7B decode at batch=1:
  AI ≈ 2 (very low) — memory-bandwidth limited.
For prefill at batch=32, seq=512:
  AI ≈ 200 (high) — compute-bound."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "How do you characterise an AI workload before optimising?",
        "a": """\
Standard workload characterisation protocol:

1. Hardware inventory: GPU model, VRAM, peak BW, peak FP16 FLOPs
2. Roofline anchors: measure peak BW (copy benchmark), peak FLOPs (matmul sweep)
3. Throughput sweep: batch size × precision → throughput-latency curve
4. Latency distribution: P50/P95/P99 at target batch size, 20+ iterations
5. Bottleneck identification: ncu roofline per hotspot kernel
6. Memory footprint: weights + activations + KV cache + gradient buffers
7. CPU utilisation: should be <20% during GPU compute (DataLoader not starving)
8. Thermal check: sustained vs burst throughput — look for >10% throughput drop
9. Cross-hardware: same script on baseline and optimised config

Report every experiment with: GPU, batch, dtype, seq_len, warmup, percentiles."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "What is the difference between synthetic and production benchmarks?",
        "a": """\
Synthetic benchmarks (MLPerf, fixed-shape matmuls):
  + Reproducible, comparable across hardware and time
  + Isolates a specific op or scenario
  − May not reflect production request distribution
  − Misses variable-length, mixed-task, and concurrency effects

Production (real-workload) benchmarks:
  + Captures actual traffic distribution (ShareGPT, real API logs)
  + Includes concurrency, memory pressure, and thermal effects
  − Harder to reproduce; depends on traffic sampling

Best practice: run both. Synthetic benchmarks for hardware selection and op
optimisation; real-workload benchmarks for capacity planning and SLO validation.
Always report which dataset/distribution you used."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "How do you report a speedup correctly?",
        "a": """\
A correct speedup report requires:
  1. Baseline: what you're comparing against (model, GPU, batch, dtype, seq_len)
  2. Optimised: what changed (ONE change per experiment)
  3. Metric: throughput (higher=better) or latency (lower=better)
  4. Statistical summary: mean ± std and P99, not just mean
  5. Conditions: warmup iterations, number of measurement iterations, duration
  6. Reproducibility: random seed, torch.backends.cudnn.deterministic if needed

Speedup = optimised_throughput / baseline_throughput
       or baseline_latency / optimised_latency

Do NOT report: speedup without baseline numbers, cherry-picked best run,
metric that wasn't the bottleneck (reporting FLOPs when you're memory-bound)."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "What is the Python overhead floor and why does it limit batch=1 latency?",
        "a": """\
Every GPU kernel launch requires a CPU-side call through the CUDA driver. For a
model with N kernel launches per forward pass, the minimum achievable latency is:

  min_latency ≥ N × kernel_launch_overhead_us  (≈1–5 µs per launch)

At batch=1, compute kernels may finish in microseconds but the CPU is still
busy launching the next kernel. The wall time is dominated by CPU-side Python
and CUDA driver overhead — not GPU compute.

torch.compile (with mode="reduce-overhead" or "max-autotune") reduces kernel
launches by fusing ops, cutting CPU overhead.

Measure: wall_time_ms - gpu_time_ms at batch=1 = Python overhead floor."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "How do you diagnose thermal throttling during a benchmark?",
        "a": """\
Thermal throttling: GPU reduces clock speed when temperature exceeds TDP limit
(typically 83°C for consumer GPUs), reducing throughput by up to 50%.

Diagnose:
  nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.mem \\
             --format=csv -l 1

  If clocks.sm drops during benchmark → throttling.
  Compare burst throughput (first 10s) vs sustained (30–60s).

Fix:
  • nvidia-smi -i 0 -pm 1     # persistent mode (reduces re-init overhead)
  • Improve cooling (undervolting, better airflow)
  • Always report sustained throughput, not burst

For benchmarks: run for 120s, report mean of last 60s."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "What is the M/D/1 queuing model and what does it predict?",
        "a": """\
M/D/1: Poisson arrivals (M), deterministic service time (D), single server (1).

  Average latency = service_time × (1 + utilisation / (2 × (1 - utilisation)))

Where utilisation = arrival_rate / service_rate = RPS / max_RPS.

At 50% utilisation: latency ≈ 1.5× service_time
At 80% utilisation: latency ≈ 3× service_time
At 90% utilisation: latency ≈ 5.5× service_time

Lesson: a GPU serving system at 80% utilisation has 3× higher tail latency than
at low load. Provision for maximum 70–75% utilisation to meet P99 SLOs.

Real systems are closer to M/M/1 (random service) which is worse; M/D/1 gives
the best-case queue latency bound."""
    },
    {
        "cat": 5, "cat_name": "Benchmarking",
        "q": "How do you benchmark torch.compile correctly?",
        "a": """\
torch.compile triggers JIT compilation on the first call with a new input shape.
Subsequent calls are fast (compiled kernel). A naive benchmark times compilation.

Correct procedure:
  1. Build the model
  2. compiled = torch.compile(model, mode="default")
  3. Warm up with the exact input shape you will benchmark:
     for _ in range(10):  # minimum 3; more for max-autotune
         _ = compiled(x)
  4. THEN time steady-state performance with CUDA events

Report both:
  • Compilation cost: first_call_ms - steady_mean_ms (one-time cost)
  • Steady-state speedup: compiled_mean_ms / eager_mean_ms

Dynamic shapes require separate compilation per shape — benchmark with the
exact shapes used in production."""
    },
]

# ──────────────────────────────────────────────────────────────────────────────
# Presentation
# ──────────────────────────────────────────────────────────────────────────────

def print_questions(questions, quiz_mode=False):
    last_cat = None
    for i, qa in enumerate(questions, 1):
        if qa["cat"] != last_cat:
            last_cat = qa["cat"]
            print(f"\n{'═' * 68}")
            print(f"  Category {qa['cat']}: {qa['cat_name']}")
            print(f"{'═' * 68}")

        print(f"\nQ{i:02d}. {qa['q']}")
        if not quiz_mode:
            wrapped = textwrap.indent(textwrap.dedent(qa["a"]).strip(), "    ")
            print(f"\n{wrapped}\n")
        else:
            print("    [answer hidden — think before reading]\n")


def main():
    parser = argparse.ArgumentParser(description="AI Perf Interview Q&A")
    parser.add_argument("--quiz", action="store_true",
                        help="Hide answers (self-quiz mode)")
    parser.add_argument("--cat", type=int, choices=[1,2,3,4,5], default=None,
                        help="Show only category N (1-5)")
    args = parser.parse_args()

    questions = QA
    if args.cat is not None:
        questions = [q for q in QA if q["cat"] == args.cat]

    print("=" * 68)
    print("  APPENDIX C — Interview Preparation")
    print(f"  {'50 Questions and Model Answers' if not args.quiz else '50 Questions (quiz mode — answers hidden)'}")
    cat_names = {1: "GPU Architecture", 2: "LLM Inference Systems",
                 3: "Profiling Tools", 4: "Distributed Systems", 5: "Benchmarking"}
    for k, v in cat_names.items():
        if args.cat is None or args.cat == k:
            count = sum(1 for q in QA if q["cat"] == k)
            print(f"  Category {k}: {v} ({count} questions)")
    print("=" * 68)

    print_questions(questions, quiz_mode=args.quiz)

    print("\n" + "=" * 68)
    if not args.quiz:
        print("  ALL QUESTIONS PRINTED")
        print()
        print("  Study tip: run with --quiz to self-test without seeing answers.")
        print("  Run with --cat N to focus on one category at a time.")
    else:
        print("  QUIZ MODE COMPLETE")
        print("  Run without --quiz to check your answers.")
    print("=" * 68)


if __name__ == "__main__":
    main()
