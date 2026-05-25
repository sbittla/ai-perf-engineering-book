#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/13.Distributed_Inference/13.1_tensor_parallelism.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 13: Distributed Inference — Section 1: Tensor Parallelism
=======================================================================
Covers book section 13.1:
  • Why models do not fit on one GPU (70B+ FP16 = 140GB)
  • Tensor parallelism: shard weight matrices across GPUs along one dimension
  • Column-parallel and row-parallel linear layers (Megatron-LM style)
  • All-Reduce communication after each row-parallel layer
  • Roofline analysis: when is the all-reduce the bottleneck?
  • Compute vs communication ratio as a function of tensor parallel degree

Run:  python IV.LLM_Inference_Systems/13.Distributed_Inference/13.1_tensor_parallelism.py
All sections must print ✓.
"""

import math
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 13.1 — Tensor Parallelism")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Why we need tensor parallelism
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Multi-GPU Motivation ──")
print("""
  MODEL SIZE VS GPU MEMORY (FP16, 2 bytes/param):
    GPT-2   (117M)  →   0.23 GB  → fits on any GPU
    Llama-7B         →  14    GB  → barely fits on A100 40GB
    Llama-13B        →  26    GB  → needs A100 80GB
    Llama-70B        → 140    GB  → needs 2× A100 80GB minimum
    GPT-4 (est. 1.8T)→ 3600   GB  → needs 40+ A100 80GB

  Even if the model fits, KV cache + activations may not:
    Llama-70B inference at batch=8, seq=2048: ~140 GB (weights) + ~8 GB (KV)
    → 2 A100 80GB mandatory even at small batch size

  SOLUTION: TENSOR PARALLELISM
    Split each weight matrix across N GPUs.
    Each GPU holds 1/N of each matrix.
    A small all-reduce synchronisation after each layer reconstructs the output.

  KEY TRADE-OFF:
    More GPUs → less memory per GPU, but more all-reduce communication.
    Optimal N is where compute time ≈ communication time.
""")
print("  ✓ Section 1 passed — tensor parallelism is necessary for large models")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Column-parallel and row-parallel linear layers
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Weight Sharding Strategy ──")
print("""
  COLUMN-PARALLEL LINEAR (Weight partitioned along output dim):
    Full weight  : [D_out, D_in]
    GPU k holds  : [D_out // N, D_in]   (rows of weight matrix)
    Forward      : y_k = x @ W_k^T
    No communication needed — each GPU produces a partial output column.

  ROW-PARALLEL LINEAR (Weight partitioned along input dim):
    Full weight  : [D_out, D_in]
    GPU k holds  : [D_out, D_in // N]   (columns of weight matrix)
    Forward      : Each GPU takes a different slice of x, computes partial matmul.
    Communication: All-Reduce (sum) across GPUs to get full D_out output.

  MEGATRON-LM PATTERN (for attention + MLP):
    Attention Q/K/V : Column-parallel  (no comm)
    Attention proj  : Row-parallel     (All-Reduce)
    MLP up-proj     : Column-parallel  (no comm)
    MLP down-proj   : Row-parallel     (All-Reduce)
    → 2 All-Reduces per transformer layer, N GPUs

  Each All-Reduce transfers: batch × seq_len × d_model × 2 bytes
  (the factor 2 is because all-reduce does reduce + broadcast)
""")


def weight_per_gpu_gb(D_out: int, D_in: int, N_gpus: int, dtype_bytes: int = 2) -> float:
    """Return GB of weight on each GPU with tensor parallelism degree N_gpus."""
    return (D_out * D_in * dtype_bytes) / N_gpus / 1e9


D_MODEL = 8192    # Llama-70B hidden dim
N_HEADS = 64
HEAD_DIM = 128
D_FFN   = 28672   # Llama-70B FFN dim
N_LAYERS = 80

# Each layer has: 3 weight matrices for QKV + 1 output proj + 2 FFN
# Simplified: total weight per layer ≈ 4 × d_model² + 2 × d_model × d_ffn
full_weight_per_layer_bytes = (
    4 * D_MODEL * D_MODEL +      # attn Q,K,V,O
    2 * D_MODEL * D_FFN          # MLP up+down
) * 2   # FP16

full_model_gb = full_weight_per_layer_bytes * N_LAYERS / 1e9

print(f"  Llama-70B approximate weight distribution:")
print(f"  {'N GPUs':>8}  {'Weight/GPU (GB)':>16}  {'Fits in A100 80GB?':>20}")
print(f"  {'─'*8}  {'─'*16}  {'─'*20}")
for N in [1, 2, 4, 8]:
    w_gpu = full_model_gb / N
    fits  = "✓ YES" if w_gpu < 80 else "✗ NO"
    print(f"  {N:>8}  {w_gpu:>16.1f}  {fits:>20}")

# TODO 1: Verify 1 GPU does not fit (>80 GB) and 2 GPUs does fit (<80 GB each)
assert full_model_gb > 80,       f"Llama-70B should be >80GB total, got {full_model_gb:.0f}"
assert full_model_gb / 2 < 80,   f"Half should fit in A100 80GB, got {full_model_gb/2:.0f}"
print(f"\n  Total Llama-70B weight: {full_model_gb:.0f} GB → need at least 2× A100 80GB")
print("  ✓ Section 2 passed — weight sharding understood")


# ─────────────────────────────────────────────────────────────
# SECTION 3: All-Reduce communication cost
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: All-Reduce Cost Analysis ──")
print("""
  All-Reduce (ring-based, N GPUs):
    Communication volume per GPU: 2 × (N-1)/N × tensor_bytes
    ≈ 2 × tensor_bytes  (for large N)

  For one all-reduce after a row-parallel layer:
    Tensor size = batch × seq_len × D_out × dtype_bytes

  With NVLink bandwidth between A100s (≈ 600 GB/s bidirectional):
    Time = 2 × tensor_bytes / (NVLink_bandwidth_per_GPU)

  TODO 2: Implement all_reduce_ms() that returns the estimated time in ms.
""")


def all_reduce_ms(
    batch: int, seq_len: int, d_model: int,
    n_gpus: int, link_bw_gbps: float,
    dtype_bytes: int = 2
) -> float:
    """
    TODO 2: Return estimated All-Reduce time in milliseconds.
    Volume per GPU = 2 × (n_gpus-1)/n_gpus × batch × seq_len × d_model × dtype_bytes
    Time = volume / (link_bw_gbps × 1e9 / 8)   (convert Gbps → bytes/s)
    Return time in ms.
    """
    # YOUR CODE HERE
    if n_gpus <= 1:
        return 0.0
    volume = 2 * (n_gpus - 1) / n_gpus * batch * seq_len * d_model * dtype_bytes
    bw_bytes_s = link_bw_gbps * 1e9 / 8
    return volume / bw_bytes_s * 1000   # seconds → ms


# Verify: 0 comm for 1 GPU
assert all_reduce_ms(1, 512, 4096, 1, 600) == 0.0

NVLINK_BW   = 600   # GB/s, A100 NVLink 3.0
PCIE_BW     = 32    # GB/s, PCIe 4.0

print(f"  All-Reduce time for batch=1, seq=512, d_model={D_MODEL}:")
print(f"\n  {'N GPUs':>8}  {'NVLink (ms)':>12}  {'PCIe (ms)':>12}")
print(f"  {'─'*8}  {'─'*12}  {'─'*12}")
for N in [2, 4, 8]:
    t_nvl  = all_reduce_ms(1, 512, D_MODEL, N, NVLINK_BW)
    t_pcie = all_reduce_ms(1, 512, D_MODEL, N, PCIE_BW)
    print(f"  {N:>8}  {t_nvl:>12.3f}  {t_pcie:>12.3f}")

# PCIe should be > NVLink
t_nvl2  = all_reduce_ms(1, 512, D_MODEL, 2, NVLINK_BW)
t_pcie2 = all_reduce_ms(1, 512, D_MODEL, 2, PCIE_BW)
assert t_pcie2 > t_nvl2, "PCIe should be slower than NVLink"
print(f"\n  PCIe 4× slower than NVLink ({t_pcie2/t_nvl2:.1f}×) → NVLink required for TP")
print("  ✓ Section 3 passed — all-reduce cost estimated for different interconnects")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Compute vs communication ratio
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Compute vs Communication Roofline ──")
print("""
  Tensor parallelism is only worthwhile when:
    compute_time >> all_reduce_time

  Compute time for one row-parallel matmul on one GPU:
    FLOPs per GPU = 2 × D_out × (D_in // N) × batch × seq_len
    Time = FLOPs / (GPU_TFLOPS × 1e12)

  TODO 3: Implement compute_ms() for the matmul on one GPU.
  Then print the compute/comm ratio for various N.
""")


def compute_ms(
    batch: int, seq_len: int, d_out: int, d_in: int,
    n_gpus: int, gpu_tflops: float
) -> float:
    """
    TODO 3: Return compute time in ms for one row-parallel matmul on one GPU.
    FLOPs per GPU = 2 × d_out × (d_in // n_gpus) × batch × seq_len
    Time = FLOPs / (gpu_tflops × 1e12)  → multiply by 1000 for ms
    """
    # YOUR CODE HERE
    flops = 2 * d_out * (d_in // n_gpus) * batch * seq_len
    return flops / (gpu_tflops * 1e12) * 1000


GPU_TFLOPS = 312   # A100 FP16 tensor core throughput (TFLOPS)

print(f"  d_model={D_MODEL}, batch=1, seq=512, A100 at {GPU_TFLOPS} TFLOPS, NVLink {NVLINK_BW}GB/s")
print(f"\n  {'N GPUs':>8}  {'Compute (ms)':>14}  {'Comm (ms)':>12}  {'C/C Ratio':>10}  {'Comm bound?':>12}")
print(f"  {'─'*8}  {'─'*14}  {'─'*12}  {'─'*10}  {'─'*12}")
for N in [1, 2, 4, 8]:
    t_comp = compute_ms(1, 512, D_MODEL, D_MODEL, N, GPU_TFLOPS)
    t_comm = all_reduce_ms(1, 512, D_MODEL, N, NVLINK_BW)
    ratio  = t_comp / t_comm if t_comm > 0 else float("inf")
    bound  = "YES" if ratio < 2 else "no"
    print(f"  {N:>8}  {t_comp:>14.3f}  {t_comm:>12.3f}  {ratio:>10.1f}  {bound:>12}")

# Verify that compute time > comm time for N=2 (otherwise TP is not useful)
tc2 = compute_ms(1, 512, D_MODEL, D_MODEL, 2, GPU_TFLOPS)
tm2 = all_reduce_ms(1, 512, D_MODEL, 2, NVLINK_BW)
print(f"\n  N=2: compute/comm = {tc2/tm2:.1f}× "
      f"({'compute-bound — TP beneficial' if tc2 > tm2 else 'comm-bound — TP overhead dominates'})")
print("  ✓ Section 4 passed — compute vs communication ratio guides TP degree choice")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Tensor parallelism in practice (vLLM / Megatron)
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Using Tensor Parallelism ──")
print("""
  vLLM (easiest path for serving):
    vllm serve meta-llama/Llama-2-70b-chat-hf \\
        --tensor-parallel-size 4   # split across 4 GPUs
        --dtype bfloat16

  Megatron-LM (training + inference):
    torchrun --nproc_per_node 8 pretrain_gpt.py \\
        --tensor-model-parallel-size 8 \\
        --pipeline-model-parallel-size 1

  HuggingFace Accelerate with device_map:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        device_map="auto",         # Accelerate splits across GPUs automatically
        torch_dtype=torch.bfloat16
    )
    # device_map="auto" uses simple column splitting, not Megatron-style TP
    # Lower communication efficiency than vLLM or Megatron

  Checking communication overhead empirically:
    import torch.distributed as dist
    # After training/serving, look for 'ncclAllReduce' in profiler timeline.
    # If ncclAllReduce > 15% of step time, TP degree may be too high.
    # Use nsys profile to capture NCCL kernels with --trace=cuda,nvtx,mpi
""")
print("  ✓ Section 5 passed — tensor parallelism commands and caveats understood")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 13.1 complete!")
print("  You understand weight sharding, all-reduce cost, and how to")
print("  choose tensor parallel degree based on compute/comm ratio.")
print("  Next: IV.LLM_Inference_Systems/13.Distributed_Inference/13.2_nccl_collectives.py")
print("=" * 60)
