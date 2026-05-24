# Interview Preparation — AI Systems Performance Engineering

Questions to answer fluently before interviewing for any AI systems performance role.

---

## GPU Architecture

**Q: What causes warp divergence and how do you detect it?**

When threads in a 32-thread warp take different code paths (if/else), the warp must serialise both paths, with inactive threads masked off. The warp width effectively halves for divergent paths. Detect with:
```bash
ncu --metrics sm__sass_average_branch_targets_threads_uniform.pct python script.py
```
Low uniform branch % → high divergence.

**Q: What is the difference between compute-bound and memory-bound kernels?**

A kernel is **compute-bound** when FLOPs are the bottleneck — the GPU's ALUs are always busy and throughput scales with FP16 tensor core speed. A kernel is **memory-bound** when data loading is the bottleneck — the GPU sits waiting for reads from HBM/GDDR6.

Measure with:
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\
             sm__throughput.avg.pct_of_peak_sustained_elapsed
```
- `dram ≈ 100%, sm < 50%` → memory-bound
- `sm ≈ 100%, dram < 50%` → compute-bound

LLM decode (batch=1) is almost always memory-bound. LLM prefill (large batches) is compute-bound.

**Q: What is the roofline model?**

A performance model with two ceilings:
1. **Memory bandwidth ceiling:** `throughput ≤ bandwidth × arithmetic_intensity`
2. **Compute ceiling:** `throughput ≤ peak_FLOPS`

The **ridge point** = peak_FLOPS / bandwidth tells you the minimum arithmetic intensity to be compute-bound.

A kernel is memory-bound if its AI (FLOPs/byte) is left of the ridge. Fix: increase reuse (tiling, caching). A kernel is compute-bound if its AI is right of the ridge. Fix: faster GPU or better kernel implementation.

**Q: What is GPU occupancy and why does it matter?**

Occupancy = active warps on SM / maximum warps an SM can hold.

High occupancy lets the GPU hide memory latency: when one warp stalls waiting for data, the SM switches to another ready warp.

Low occupancy → SM idles while waiting for stalled warp.

Measure: `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`

**Q: What is memory coalescing?**

When 32 warp threads access 32 contiguous addresses (stride-1), the hardware coalesces into ONE 128-byte memory transaction. With stride > 1, multiple transactions are needed → memory bandwidth wasted proportionally.

Check: `ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`

---

## AI Runtime

**Q: How does the KV cache work in autoregressive generation?**

In the prefill phase, the transformer computes and stores Keys (K) and Values (V) for every input token in a cache. During decode, each new token only needs to compute Q for itself, then attends over the full cached K, V. This avoids recomputing past context on every step.

**Why it's memory-bandwidth bound:** At batch=1, decode reads the entire KV cache from GPU DRAM every token step. Throughput ≈ GPU_BW / (bytes_per_token_in_kv_cache).

KV size per token = `2 × n_layers × n_heads × head_dim × dtype_bytes`

**Q: What is PagedAttention?**

vLLM's virtual memory system for KV cache. Instead of pre-allocating a fixed max_seq_len block per request (wasteful when response is short), PagedAttention divides KV memory into fixed-size pages (e.g., 16 tokens each) allocated on-demand.

Benefits: no internal fragmentation, supports variable-length sequences, allows more concurrent requests.

**Q: What is continuous batching?**

In static batching (HuggingFace generate()), all requests in a batch must finish before new ones join. Short requests pad with zeros until the longest finishes → wasted GPU compute.

Continuous batching removes finished requests immediately and admits new ones at the next decode step. GPU always has real work → higher utilisation and throughput.

Typically 3–5× throughput improvement over static batching on variable-length workloads.

**Q: What limits LLM throughput?**

1. **Decode:** memory bandwidth (reads entire KV cache per token)
2. **Prefill:** compute (large matrix multiplies over all input tokens)
3. **Concurrency:** KV cache VRAM limits how many requests are in flight
4. **CPU overhead:** Python/tokeniser latency creates GPU stalls

**Q: What are the tradeoffs of FP16 vs INT8 inference?**

| | FP16 | INT8 |
|--|------|------|
| Memory | 2 bytes/weight | 1 byte/weight |
| Accuracy | Near-perfect | <1% accuracy loss (with calibration) |
| Speed | Tensor cores | INT8 tensor cores (faster on A100+) |
| Compatibility | Universal | Requires calibration data |
| VRAM savings | 0 | 50% vs FP16 |

Use INT8 when: model doesn't fit in VRAM at FP16, or decode is memory-bandwidth limited and you want more throughput.

**Q: What is speculative decoding?**

Use a small draft model to generate K candidate tokens cheaply, then verify all K in ONE target model forward pass. If all accepted: K tokens for the cost of ~1 target step.

Expected speedup = `(1 - α^(K+1)) / (1 - α)` where α = acceptance rate.

At α=0.8, K=5: ~3.6× throughput improvement. Only works when: draft and target are aligned in distribution, and the target model is the bottleneck.

---

## Linux Systems

**Q: Why does NUMA topology matter for GPU workloads?**

On multi-socket servers, memory attached to CPU socket 0 is 2× faster for processes on socket 0 than for processes on socket 1 (cross-QPI access). A GPU physically connected to socket 0's PCIe lanes means:
- DataLoader workers on socket 0 → direct memory path → fast
- DataLoader workers on socket 1 → cross-QPI → slow H2D transfers

Fix: `numactl --cpunodebind=$(cat /sys/bus/pci/devices/0000:01:00.0/numa_node) python train.py`

**Q: What causes cache misses and how do you measure them?**

Cache miss = the CPU looked up an address not in L1/L2/L3, forcing a slow DRAM fetch (~200 cycles). Causes:
- Working set larger than L3 cache
- Random/non-sequential access patterns (stride > cache line)
- Multiple threads evicting each other's data (thrashing)

Measure:
```bash
perf stat -e cache-misses,cache-references python train.py
# cache-miss % = cache-misses / cache-references
```

**Q: How do flamegraphs work?**

A flamegraph visualises stack samples:
- X axis = time (width proportional to CPU time spent)
- Y axis = call stack depth (bottom = main, top = leaf function)
- Wide bars at the top = hot leaf functions
- Wide bars with narrow children = the function itself is expensive

Generated with: `py-spy record -o flame.svg -- python script.py`

**Q: What is iowait and what does it tell you?**

`iowait` = percentage of CPU time where the CPU is idle AND there's an outstanding I/O request. High iowait (>20%) means disk/NVMe is the bottleneck, not the CPU or GPU. Fix: faster storage (NVMe), RAM disk (tmpfs), or better prefetching.

Check: `vmstat 1 | awk '{print $16}'` (wa column)

---

## Distributed Systems

**Q: Why does NCCL matter for multi-GPU training?**

NCCL (NVIDIA Collective Communications Library) provides optimised implementations of collective operations (all-reduce, broadcast, all-gather) that exploit NVLink (GPU-GPU), PCIe, and InfiniBand.

Without NCCL, gradients would have to go GPU → CPU → network → CPU → GPU. NCCL keeps data on GPU throughout.

Key fact: all-reduce time scales with gradient tensor size / interconnect bandwidth. For large models, gradient sync can be 20–50% of training time.

**Q: What are the tradeoffs between tensor parallelism and pipeline parallelism?**

| | Tensor Parallelism | Pipeline Parallelism |
|--|------------------|---------------------|
| Splits | Each layer across GPUs | Different layers to different GPUs |
| Communication | All-reduce after EVERY layer | Point-to-point between stages |
| Latency | Adds all-reduce latency per layer | Adds pipeline bubble |
| Memory | Each GPU holds 1/N of every layer | Each GPU holds N/total_layers |
| Best for | Attention (fast NVLink needed) | Sequential layer stacks |
| Library | Megatron-LM, tensor_parallel | DeepSpeed, GPipe |

**Q: How does FSDP differ from DDP?**

DDP (DistributedDataParallel): each GPU holds the FULL model. Gradients all-reduced after backward. Memory per GPU = full model × 3 (params + grads + optim states).

FSDP (FullyShardedDataParallel): each GPU holds 1/N of model params, gradients, AND optimizer states. Before each forward pass, all-gather reconstructs the full layer; after backward, reduce-scatter distributes gradients back.

Memory per GPU ≈ full_model × 3 / N. Enables training models larger than one GPU's VRAM.

**Q: How do you overlap communication and computation in distributed training?**

Bucket gradients: PyTorch DDP groups parameters into buckets and starts all-reduce for each bucket as soon as it's complete during backward, overlapping communication with computation of later layers.

FSDP: all-gather for layer N+1 can overlap with forward compute of layer N using prefetching.

Check overlap in nsys: look for NCCL (orange) and compute (green) rows overlapping on the timeline.

---

## Workload Characterisation

**Q: What is the methodology for characterising an AI workload?**

1. **Hardware inventory:** GPU model, VRAM, peak bandwidth, peak FLOPs
2. **Roofline anchors:** measure peak BW (copy benchmark) and peak FLOPs (matmul)
3. **Throughput sweep:** batch size × precision → T-B curve
4. **Latency distribution:** P50/P95/P99 at target batch size
5. **Bottleneck identification:** ncu roofline analysis per kernel
6. **Memory footprint:** weights + activations + KV cache
7. **CPU utilisation:** should be <20% during GPU compute
8. **Thermal check:** sustained vs burst throughput (look for throttling)
9. **Compare across hardware:** same script on CPU/single GPU/multi-GPU/quantized

**Q: What is the difference between synthetic and real workload benchmarks?**

Synthetic benchmarks (MLPerf offline, fixed batch matmuls) are reproducible and comparable across hardware but may not reflect production behaviour.

Real workload benchmarks capture: variable-length sequences, mixed input distributions, concurrent users, memory pressure, and thermal effects.

For capacity planning: use real workloads or realistic synthetic distributions (ShareGPT dataset for LLM serving).

Always report: batch size, sequence length, GPU, dtype, warmup strategy, and percentiles — not just average throughput.
