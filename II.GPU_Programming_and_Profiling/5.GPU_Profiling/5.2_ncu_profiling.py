#!/usr/bin/env python3
"""
5.GPU_Profiling/5.2_ncu_profiling.py  ─  Chapter 5: Nsight Compute and Kernel Classification
=======================================================================
Covers book section 5.2:
  • Arithmetic intensity calculation
  • Classifying kernels as memory-bound or compute-bound
  • Key ncu metrics reference
  • Building a roofline classification table

Run:
  python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py
  ncu --set basic --kernel-name gemm python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py

All sections must print ✓.
"""

import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 5.2 — Nsight Compute and Kernel Classification")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# HOW TO USE ncu
# ─────────────────────────────────────────────────────────────
print("""
  ── How to use Nsight Compute (ncu) ──

  Nsight Compute profiles individual GPU kernels with hardware counters:
    - Memory throughput as % of peak HBM bandwidth
    - SM occupancy (% of theoretical maximum warps)
    - Tensor Core utilization (% of peak)
    - L1/L2 cache hit rates
    - Warp divergence cost

  Basic usage — profile all kernels, show bandwidth and compute metrics:
    ncu --set basic python 5.2_ncu_profiling.py

  Profile only GEMM (matrix multiply) kernels:
    ncu --set basic --kernel-name gemm \\
        python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py

  Full report (slower — runs multiple passes per kernel):
    ncu --set full --kernel-name gemm \\
        -o /tmp/5.2_ncu_report \\
        python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py

  Key metrics to look for in the ncu output:
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed
      → Fraction of peak HBM bandwidth achieved (high = memory-bound)
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active
      → Tensor Core utilization
    sm__warps_active.avg.pct_of_peak_sustained_active
      → SM occupancy
""")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Arithmetic Intensity Calculation
# ─────────────────────────────────────────────────────────────
print("── Section 1: Arithmetic Intensity Calculation ──")
print("""
  Arithmetic Intensity (AI) = FLOPs / bytes_transferred

  For a matrix multiply (M, K) @ (K, N):
    FLOPs  = 2 * M * K * N         (one multiply + one add per element pair)
    bytes  = element_size * (M*K + K*N + M*N)
               input A    + input B + output C

  High AI → compute-bound.  Low AI → memory-bound.
  The ridge point = peak_tflops * 1e12 / (peak_bandwidth_gbs * 1e9)
  For A100: 312e12 / 2000e9 = 156 FLOPs/byte
""")

def compute_ai(M: int, K: int, N: int, dtype: torch.dtype) -> float:
    """
    TODO 1: Implement this function.
    Compute arithmetic intensity (FLOPs / bytes) for (M,K) @ (K,N) matmul.

    Steps:
      element_size = 2 if dtype in (torch.float16, torch.bfloat16) else 4
      flops = 2 * M * K * N
      bytes_io = element_size * (M*K + K*N + M*N)
      return flops / bytes_io
    """
    pass  # YOUR CODE HERE → return flops / bytes_io


# Verify the function was implemented
ai_test = compute_ai(1024, 1024, 1024, torch.float16)
assert ai_test is not None and ai_test > 100, (
    f"compute_ai(1024, 1024, 1024, float16) should be > 100, got {ai_test}. "
    "Did you implement the TODO?"
)
print(f"  AI for (1024,1024) @ (1024,1024) FP16: {ai_test:.1f} FLOPs/byte")

# Show the intuition across sizes
for M in [1, 8, 64, 512, 2048]:
    ai = compute_ai(M, 4096, 4096, torch.float16)
    print(f"  M={M:>5}, K=N=4096, FP16 → AI = {ai:.1f} FLOPs/byte")

print("  ✓ Section 1 passed — arithmetic intensity formula implemented")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Memory-Bound vs Compute-Bound Classification
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Memory-Bound vs Compute-Bound ──")
print("""
  Given a GPU's peak FP16 TFLOP/s and peak memory bandwidth (GB/s),
  the ridge point is the AI where the workload transitions:

    ridge_point = peak_tflops * 1e12 / (peak_bw_gbs * 1e9)

  If AI < ridge_point → memory-bound (more bandwidth would help)
  If AI > ridge_point → compute-bound (more compute or FP16 would help)

  A100 ridge point: 312e12 / 2000e9 = 156 FLOPs/byte
  H100 ridge point: 989e12 / 3350e9 = 295 FLOPs/byte

  This means a kernel that is compute-bound on A100 might be
  memory-bound on H100 (because H100's compute grew more than its BW).
""")

def classify_kernel(ai: float, ridge_point: float) -> str:
    """
    TODO 2: Return "memory-bound" if ai < ridge_point, else "compute-bound".
    """
    pass  # YOUR CODE HERE → return "memory-bound" or "compute-bound"


assert classify_kernel(1.0,   156) == "memory-bound",  \
    "AI=1.0 vs ridge=156 → should be memory-bound"
assert classify_kernel(200.0, 156) == "compute-bound", \
    "AI=200 vs ridge=156 → should be compute-bound"
assert classify_kernel(156.0, 156) == "compute-bound", \
    "AI exactly at ridge → classify as compute-bound (boundary)"

print(f"  classify_kernel(  1.0, 156) = {classify_kernel(1.0,   156)}")
print(f"  classify_kernel(200.0, 156) = {classify_kernel(200.0, 156)}")
print(f"  classify_kernel(156.0, 156) = {classify_kernel(156.0, 156)}")
print("  ✓ Section 2 passed — classification function correct")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Key ncu Metrics Explained
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Key ncu Metrics ──")
print("""
  The five most important ncu metrics for diagnosing GPU kernel performance:

  1. Memory throughput % (l1tex__t_bytes...pct_of_peak)
     What it means: fraction of peak HBM bandwidth the kernel achieved.
     How to use: if >80%, you are hitting the memory wall; the only fix
     is to reduce bytes transferred (kernel fusion, quantisation).

  2. Tensor Core utilization (sm__pipe_tensor_op_hmma...pct_of_peak)
     What it means: fraction of cycles Tensor Cores were active.
     How to use: if <50% on a FP16 GEMM, check alignment (must be ×8).

  3. SM occupancy (sm__warps_active.avg.pct_of_peak_sustained_active)
     What it means: fraction of theoretical maximum warps that were active.
     How to use: low occupancy (<30%) may indicate register pressure or
     too little work. Increase batch size.

  4. L1 cache hit rate (l1tex__t_sector_hit_rate)
     What it means: fraction of memory requests served from L1.
     How to use: low rate on a non-GEMM kernel suggests strided or random
     access. Investigate memory layout.

  5. Warp efficiency / divergence (smsp__thread_inst_executed_pred_on...pct)
     What it means: fraction of instruction slots that did useful work
     (vs masked by divergent branches).
     How to use: <80% efficiency suggests warp divergence.
     Fix: restructure data to avoid per-thread branches.
""")

# Now measure actual DRAM throughput proxy with a large tensor copy
print("  Measuring DRAM bandwidth proxy via large tensor copy...")

if DEVICE == "cuda":
    # TODO 3: Measure bandwidth for copying a 512MB tensor.
    #   src = torch.randn(128*1024*1024, device=DEVICE)
    #   dst = torch.empty_like(src)
    #   Use CUDA events to time dst.copy_(src) for 5 iterations.
    #   achieved_bw = (src.numel() * src.element_size() * 2) / (ms/1000) / 1e9
    #   (factor of 2: one read + one write)

    src = None  # YOUR CODE HERE → torch.randn(128*1024*1024, device=DEVICE)
    dst = None  # YOUR CODE HERE → torch.empty_like(src)

    # YOUR CODE HERE: warmup and timing loop with CUDA events
    achieved_bw = None  # YOUR CODE HERE → compute bandwidth

    assert achieved_bw is not None and achieved_bw > 0, \
        "achieved_bw must be > 0 GB/s"
    print(f"  Achieved DRAM bandwidth: {achieved_bw:.1f} GB/s")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  (GPU: {props.name})")
else:
    achieved_bw = 50.0  # CPU DRAM reference
    print(f"  CPU DRAM bandwidth proxy: {achieved_bw:.1f} GB/s (CPU fallback)")

print("  ✓ Section 3 passed — ncu metrics explained and bandwidth measured")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Identifying Bottlenecks with AI
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Identifying Bottlenecks with AI ──")
print("""
  A100 reference specs (used for classification below):
    Peak FP16 Tensor Core throughput: 312 TFLOP/s
    Peak HBM2e bandwidth:            2000 GB/s
    Ridge point:                      156 FLOPs/byte

  We build a table for the same K=N=4096 matmul at different batch sizes M.
  As M grows, AI grows proportionally — the kernel shifts from memory-bound
  to compute-bound.  This table explains why LLM inference at small batch
  sizes is always memory-bound.
""")

A100_TFLOPS = 312.0     # FP16 Tensor Core peak
A100_BW_GBS = 2000.0    # HBM bandwidth
a100_ridge  = A100_TFLOPS * 1e12 / (A100_BW_GBS * 1e9)

print(f"  A100 ridge point: {a100_ridge:.1f} FLOPs/byte\n")
print(f"  {'M':>6}  {'AI (FLOP/byte)':>16}  {'Classification':>16}")
print(f"  {'-'*6}  {'-'*16}  {'-'*16}")

classifications = []

# TODO 4: For each M in [1, 32, 256, 2048], call compute_ai(M, 4096, 4096, torch.float16)
#   and classify_kernel(ai, a100_ridge).  Print the result.
#   Append each classification to the `classifications` list.

shapes = [(1, 4096, 4096), (32, 4096, 4096), (256, 4096, 4096), (2048, 4096, 4096)]
for M, K, N in shapes:
    # YOUR CODE HERE: compute ai and classification, print, and append to classifications
    pass  # replace with: ai = compute_ai(...); cls = classify_kernel(...); print(...)

assert len(classifications) > 0, "Must populate classifications list in TODO 4"
assert any(c == "memory-bound"  for c in classifications), \
    "At least one shape should be memory-bound (small M)"
assert any(c == "compute-bound" for c in classifications), \
    "At least one shape should be compute-bound (large M)"

print("\n  Interpretation:")
print("    Small M (inference, batch=1) → memory-bound: adding compute doesn't help")
print("    Large M (training)           → compute-bound: Tensor Cores are the bottleneck")
print("  ✓ Section 4 passed — roofline classification table built")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 5.2 complete!")
print("  You can now compute arithmetic intensity, classify kernels on")
print("  the roofline, interpret ncu metrics, and build diagnosis tables.")
print("  Next: II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.3_torch_profiler.py")
print("=" * 60)
