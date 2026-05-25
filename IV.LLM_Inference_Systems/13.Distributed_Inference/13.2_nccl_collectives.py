#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/13.Distributed_Inference/13.2_nccl_collectives.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 13: Distributed Inference — Section 2: NCCL Collectives
=======================================================================
Covers book section 13.2:
  • What NCCL is and why it is used for GPU collectives
  • AllReduce: sum across all ranks, result available on all ranks
  • AllGather: each rank contributes a shard; all ranks receive the full tensor
  • Broadcast: one rank sends to all others
  • Bandwidth model: bus bandwidth vs compute bandwidth
  • Ring-AllReduce algorithm: why it scales to 1000s of GPUs
  • Simulating collective costs to predict communication overhead

Run:  python IV.LLM_Inference_Systems/13.Distributed_Inference/13.2_nccl_collectives.py
All sections must print ✓.
"""

import math
import time
import torch

print("=" * 60)
print("  Exercise 13.2 — NCCL Collectives for Distributed Inference")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: NCCL — NVIDIA Collective Communications Library
# ─────────────────────────────────────────────────────────────
print("── Section 1: What NCCL Does and Why ──")
print("""
  NCCL (pronounced "nickel") is NVIDIA's library for high-bandwidth
  GPU-to-GPU collective communication. PyTorch distributed uses NCCL
  as its backend for GPU operations.

  NCCL COLLECTIVES:
    AllReduce  : Each rank has a tensor. Sum (or max/min) across all ranks.
                 Each rank ends up with the SAME reduced result.
                 Used by: tensor parallelism (sum partial matmul outputs)

    AllGather  : Each rank has a shard of size S. All ranks receive the
                 full concatenated tensor of size N × S.
                 Used by: FSDP (reconstruct full parameters before forward)

    Broadcast  : Rank 0 sends a tensor; all other ranks receive it.
                 Used by: distributing the model or configuration from rank 0.

    ReduceScatter: Reduce + scatter in one step (used by FSDP backward).

  WHY NOT JUST USE TCP?
    • NVLink: 600 GB/s between A100s in the same node (vs 25 GB/s PCIe)
    • InfiniBand: 400 Gb/s between nodes (vs 100 Gb/s Ethernet)
    • NCCL uses these high-bandwidth paths automatically.
    • NCCL fuses multiple small messages into large transfers (better bandwidth).
""")
print("  ✓ Section 1 passed — NCCL is the backbone of distributed GPU training/inference")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Ring-AllReduce bandwidth model
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Ring-AllReduce — The Algorithm That Scales ──")
print("""
  Naive AllReduce: rank 0 collects all tensors (bottleneck on rank 0).
  Ring-AllReduce: each rank sends to the next in a ring, two phases:
    Phase 1 (ReduceScatter): N-1 steps, each sends 1/N of tensor.
    Phase 2 (AllGather):     N-1 steps, each forwards received shard.

  Total data sent per GPU: 2 × (N-1)/N × tensor_bytes ≈ 2 × tensor_bytes
  This is INDEPENDENT of N (for large N) — ring scales perfectly!

  EFFECTIVE BUS BANDWIDTH formula (NCCL benchmark definition):
    bus_bw = (2 × tensor_bytes × (N-1)/N) / (time × N)

  LATENCY vs BANDWIDTH:
    Small tensors: dominated by latency (∝ N, number of hops)
    Large tensors: dominated by bandwidth (nearly independent of N)
    Cross-over at ≈ 1 MB for NVLink; ≈ 10 MB for InfiniBand

  TODO 1: Implement ring_allreduce_time_ms() below.
""")


def ring_allreduce_time_ms(
    tensor_bytes: int,
    n_gpus: int,
    link_bw_gbps: float,
    latency_us: float = 10.0
) -> float:
    """
    TODO 1: Estimate ring AllReduce time in ms.
    Volume per GPU = 2 × (n_gpus - 1) / n_gpus × tensor_bytes
    BW term  = volume / (link_bw_gbps × 1e9 / 8)          (seconds)
    Latency  = (2 × n_gpus - 1) × latency_us × 1e-6       (seconds, ring hops)
    Return (BW term + latency) × 1000  (ms)
    For n_gpus == 1: return 0.0.
    """
    # YOUR CODE HERE
    if n_gpus <= 1:
        return 0.0
    volume   = 2 * (n_gpus - 1) / n_gpus * tensor_bytes
    bw_bytes = link_bw_gbps * 1e9 / 8
    bw_time  = volume / bw_bytes
    lat_time = (2 * n_gpus - 1) * latency_us * 1e-6
    return (bw_time + lat_time) * 1000


# Sanity: 1 GPU → 0 ms
assert ring_allreduce_time_ms(1_000_000, 1, 600) == 0.0

NVLINK_BW = 600
PCIE_BW   = 32
IB_BW     = 50   # InfiniBand HDR ≈ 50 GB/s effective

print(f"  AllReduce time for a 256 MB tensor (Llama-7B hidden, batch=1, seq=512):")
print(f"\n  {'N GPUs':>8}  {'NVLink (ms)':>12}  {'PCIe (ms)':>12}  {'IB (ms)':>12}")
print(f"  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*12}")
TENSOR_BYTES = 256 * 1024 * 1024   # 256 MB
for N in [2, 4, 8, 16]:
    t_nvl  = ring_allreduce_time_ms(TENSOR_BYTES, N, NVLINK_BW)
    t_pcie = ring_allreduce_time_ms(TENSOR_BYTES, N, PCIE_BW)
    t_ib   = ring_allreduce_time_ms(TENSOR_BYTES, N, IB_BW)
    print(f"  {N:>8}  {t_nvl:>12.2f}  {t_pcie:>12.2f}  {t_ib:>12.2f}")

# NVLink should be fastest
t2_nvl  = ring_allreduce_time_ms(TENSOR_BYTES, 2, NVLINK_BW)
t2_pcie = ring_allreduce_time_ms(TENSOR_BYTES, 2, PCIE_BW)
assert t2_nvl < t2_pcie, "NVLink should be faster than PCIe"
print(f"\n  NVLink {t2_pcie/t2_nvl:.1f}× faster than PCIe for AllReduce")
print("  ✓ Section 2 passed — ring AllReduce time model working")


# ─────────────────────────────────────────────────────────────
# SECTION 3: AllGather bandwidth model
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: AllGather — Reconstructing Sharded Parameters ──")
print("""
  AllGather is used when each GPU holds a shard of a tensor and all GPUs
  need the full tensor (e.g., FSDP before a forward pass).

  RING-ALLGATHER:
    Volume per GPU: (N-1)/N × full_tensor_bytes
    ≈ full_tensor_bytes for large N

  AllGather sends (N-1)/N of the tensor vs AllReduce 2×(N-1)/N.
  AllGather is roughly half the volume of AllReduce.

  TODO 2: Implement ring_allgather_time_ms() below.
""")


def ring_allgather_time_ms(
    full_tensor_bytes: int,
    n_gpus: int,
    link_bw_gbps: float,
    latency_us: float = 10.0
) -> float:
    """
    TODO 2: Estimate ring AllGather time in ms.
    Volume per GPU = (n_gpus - 1) / n_gpus × full_tensor_bytes
    BW term  = volume / (link_bw_gbps × 1e9 / 8)
    Latency  = (n_gpus - 1) × latency_us × 1e-6
    Return (BW term + latency) × 1000  (ms). Return 0 for n_gpus == 1.
    """
    # YOUR CODE HERE
    if n_gpus <= 1:
        return 0.0
    volume   = (n_gpus - 1) / n_gpus * full_tensor_bytes
    bw_bytes = link_bw_gbps * 1e9 / 8
    bw_time  = volume / bw_bytes
    lat_time = (n_gpus - 1) * latency_us * 1e-6
    return (bw_time + lat_time) * 1000


# AllGather < AllReduce for same tensor size
t_ag = ring_allgather_time_ms(TENSOR_BYTES, 4, NVLINK_BW)
t_ar = ring_allreduce_time_ms(TENSOR_BYTES, 4, NVLINK_BW)
assert t_ag < t_ar, f"AllGather ({t_ag:.2f}) should be < AllReduce ({t_ar:.2f})"
print(f"  AllGather vs AllReduce (256 MB, 4 GPUs, NVLink):")
print(f"    AllGather : {t_ag:.3f} ms")
print(f"    AllReduce : {t_ar:.3f} ms  (AllReduce = 2× AllGather volume)")
print("  ✓ Section 3 passed — AllGather is cheaper than AllReduce per byte")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Communication budget per transformer layer
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Communication Budget in a Transformer Layer ──")
print("""
  With tensor parallelism degree N, each transformer layer requires:
    2 AllReduce operations (after attention output proj and FFN down proj)

  Each AllReduce tensor: batch × seq_len × d_model × 2 bytes

  For Llama-7B (d_model=4096), batch=1, seq=512, 4 GPUs:
    Tensor per AllReduce = 1 × 512 × 4096 × 2 = 4 MB
    Time per AllReduce   ≈ ring_allreduce_time_ms(4 MB, 4, 600)
    Time per layer       ≈ 2 × above

  Compare to compute per layer (Section 13.1 formula):
    FLOPs per layer (attn+FFN) ≈ 12 × d_model² × batch × seq_len
    Time per layer on A100      = FLOPs / (312 TFLOPS per GPU × N)
""")

D_MODEL   = 4096   # Llama-7B
BATCH     = 1
SEQ       = 512
GPU_TFLOPS = 312   # A100

for N in [2, 4, 8]:
    tensor_per_ar = BATCH * SEQ * D_MODEL * 2   # bytes
    t_comm_per_layer = 2 * ring_allreduce_time_ms(tensor_per_ar, N, NVLINK_BW)

    flops_per_layer = 12 * D_MODEL**2 * BATCH * SEQ
    t_comp_per_layer = flops_per_layer / (GPU_TFLOPS * 1e12 * N) * 1000   # ms per GPU

    ratio = t_comp_per_layer / t_comm_per_layer if t_comm_per_layer > 0 else float("inf")
    print(f"  N={N}: compute={t_comp_per_layer:.3f}ms  comm={t_comm_per_layer:.3f}ms  "
          f"ratio={ratio:.1f}  {'✓ compute-dominated' if ratio > 5 else '⚠ comm overhead'}")

# TODO 3: verify that N=2 has a higher compute/comm ratio than N=8
def layer_comm_ms(N):
    tensor = BATCH * SEQ * D_MODEL * 2
    return 2 * ring_allreduce_time_ms(tensor, N, NVLINK_BW)

def layer_comp_ms(N):
    flops = 12 * D_MODEL**2 * BATCH * SEQ
    return flops / (GPU_TFLOPS * 1e12 * N) * 1000

ratio_2 = layer_comp_ms(2) / layer_comm_ms(2)
ratio_8 = layer_comp_ms(8) / layer_comm_ms(8)
assert ratio_2 > ratio_8, "N=2 should have better compute/comm ratio than N=8"
print(f"\n  Compute/comm degrades from {ratio_2:.1f}× (N=2) to {ratio_8:.1f}× (N=8)")
print("  ✓ Section 4 passed — communication overhead grows with TP degree")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Measuring NCCL bandwidth with nccl-tests
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Benchmarking NCCL in Practice ──")
print("""
  NCCL-TESTS (NVIDIA official benchmark):
    git clone https://github.com/NCCL/nccl-tests
    make MPI=0 CUDA_HOME=/usr/local/cuda
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 4

  Output columns:
    size        : tensor size in bytes
    count       : number of elements
    time        : algorithm time (μs)
    algbw       : algorithm bandwidth = size / time (GB/s)
    busbw       : bus bandwidth = algbw × 2(N-1)/N (effective NVLink/IB usage)

  TARGET VALUES (A100 8-GPU NVLink):
    busbw AllReduce   : 240+ GB/s  (out of 300 GB/s theoretical per GPU)
    busbw AllGather   : 270+ GB/s
    If measured busbw < 50% of theoretical → check NVLink topology

  PYTORCH DIRECT MEASUREMENT (single node, CPU-level timing):
    import torch.distributed as dist
    dist.init_process_group("nccl")
    tensor = torch.randn(256*1024*1024//4, device="cuda")  # 256 MB
    dist.barrier()
    t0 = time.perf_counter()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    bw_gbps = (256e6 * 2 * (N-1)/N) / (time.perf_counter() - t0) / 1e9
    print(f"AllReduce bus bandwidth: {bw_gbps:.1f} GB/s")

  WHAT TO DO WHEN NCCL IS THE BOTTLENECK:
    1. Reduce tensor parallel degree (N) to lower communication volume
    2. Enable NCCL socket tuning: NCCL_SOCKET_IFNAME=eth0
    3. Switch to overlapping comm+compute (--overlap-grad-reduce in Megatron)
    4. Use pipeline parallelism to shift the communication off the critical path
""")
print("  ✓ Section 5 passed — NCCL benchmarking and tuning strategy understood")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 13.2 complete!")
print("  You understand AllReduce and AllGather algorithms,")
print("  their bandwidth models, and how to benchmark NCCL.")
print("  Next: IV.LLM_Inference_Systems/13.Distributed_Inference/13.3_fsdp_and_pipeline.py")
print("=" * 60)
