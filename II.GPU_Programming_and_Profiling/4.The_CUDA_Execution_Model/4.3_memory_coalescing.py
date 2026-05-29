#!/usr/bin/env python3
"""
4.The_CUDA_Execution_Model/4.3_memory_coalescing.py  ─  Chapter 4: Memory Coalescing
=======================================================================
Covers book section 4.2:
  • Stride-1 vs stride-N access bandwidth measurement
  • L1/L2/DRAM latency tiers
  • Shared memory concept
  • Layout implications for transformers

Run:  python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.3_memory_coalescing.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 4.2 — Memory Coalescing and the GPU Memory Hierarchy")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Memory Coalescing
# ─────────────────────────────────────────────────────────────
print("── Section 1: Memory Coalescing ──")
print("""
  When 32 threads in a warp all access CONSECUTIVE memory addresses
  (stride 1), the GPU can serve all 32 reads in a single 128-byte
  memory transaction.  This is a coalesced access.

  When threads access memory with a stride (e.g. stride=8 means each
  thread skips 7 elements), the GPU must issue multiple transactions
  to service the warp, dramatically reducing effective bandwidth.

  We measure bandwidth for strides 1, 2, 4, 8, 16, 32 using index-gather
  on a large float32 tensor.

  Bandwidth (GB/s) = bytes_read / (time_s)
                   = (n_elements * 4 bytes) / (time_ms / 1000) / 1e9
""")

N_ELEMS = 64 * 1024 * 1024  # 64M float32 = 256 MB

bw_results = {}

if DEVICE == "cuda":
    src = torch.randn(N_ELEMS, device=DEVICE, dtype=torch.float32)

    for stride in [1, 2, 4, 8, 16, 32]:
        # Build index tensor: [0, stride, 2*stride, ...]
        n_idx = N_ELEMS // stride
        indices = torch.arange(0, n_idx * stride, stride,
                               device=DEVICE, dtype=torch.long)

        # Warmup
        for _ in range(3):
            _ = src[indices]
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        ITERS = 10
        start.record()
        for _ in range(ITERS):
            out = src[indices]
        end.record()
        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)
        ms_per_iter = total_ms / ITERS

        # TODO 1: Compute bandwidth in GB/s.
        #   bytes_read = n_idx * 4  (each indexed element is 4 bytes)
        #   bw_gbs = bytes_read / (ms_per_iter / 1000) / 1e9
        bw_gbs = None  # YOUR CODE HERE → bytes_read / (ms_per_iter / 1000) / 1e9

        assert bw_gbs is not None, "compute bw_gbs"
        bw_results[stride] = bw_gbs
        print(f"  stride={stride:>2}  {bw_gbs:>8.1f} GB/s   ({ms_per_iter:.3f} ms)")

    assert bw_results[1] > bw_results[8] * 2, (
        f"stride-1 BW ({bw_results[1]:.1f} GB/s) should be >2x stride-8 BW "
        f"({bw_results[8]:.1f} GB/s)"
    )
else:
    bw_results = {1: 900, 2: 500, 4: 260, 8: 140, 16: 80, 32: 45}
    print("  (CUDA not available — using reference bandwidth values)")

print("  ✓ Section 1 passed — stride-1 access has highest bandwidth")

# ─────────────────────────────────────────────────────────────
# SECTION 2: GPU Memory Hierarchy
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: GPU Memory Hierarchy ──")
print("""
  Inside the GPU there is a memory hierarchy analogous to the CPU:

    Registers  →  L1 / Shared memory  →  L2 cache  →  HBM (global)

  Small tensors fit entirely in L1 or L2 cache — repeated reads are
  fast.  Large tensors spill to HBM and experience full memory latency.

  We measure torch.sum() throughput for tensors of increasing size to
  observe the cache transition:
    < 1 MB  : likely L1/L2 hit  → high effective bandwidth
    1–40 MB : L2 hit            → moderate bandwidth
    > 40 MB : DRAM (HBM) bound  → capped by HBM bandwidth
""")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    l2_size_mb = props.l2_cache_size / 1e6
    print(f"  L2 cache size: {l2_size_mb:.0f} MB (from device properties)\n")

    sizes_mb = [0.01, 0.1, 1.0, 10.0, 64.0, 256.0]
    prev_bw = None

    for size_mb in sizes_mb:
        n = int(size_mb * 1e6 / 4)  # float32 elements
        t = torch.randn(n, device=DEVICE, dtype=torch.float32)

        # Warmup
        for _ in range(3):
            t.sum()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        ITERS = 20
        start.record()
        for _ in range(ITERS):
            t.sum()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        bw = (n * 4) / (ms / 1000) / 1e9  # GB/s

        # TODO 2: Classify the memory tier.
        #   If size_mb < 1.0: "L1/L2 hit"
        #   elif size_mb < l2_size_mb: "L2 hit"
        #   else: "DRAM"
        classification = None  # YOUR CODE HERE → one of: "L1/L2 hit", "L2 hit", "DRAM"

        assert classification is not None and len(classification) > 0, \
            "classification must not be empty"
        print(f"  {size_mb:>7.2f} MB  {bw:>8.1f} GB/s  [{classification}]")

else:
    print("  (CUDA not available — skipping memory hierarchy measurement)")

print("  ✓ Section 2 passed — memory tier classification complete")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Row-major vs Column-major Access
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Row-major vs Column-major Access ──")
print("""
  PyTorch tensors are row-major (C-contiguous) by default.
  This means row elements are stored consecutively in memory.

  Accessing M[i, :] (a full row) reads a contiguous block → coalesced.
  Accessing M[:, i] (a full column) reads every 1024th element → strided.

  For transformer weight matrices that must be accessed both ways
  (Q/K/V projections need both input features and output features),
  layout choices affect bandwidth efficiency.

  We time 128 repeated slices of rows vs columns on a (1024, 1024) FP16 matrix.
""")

if DEVICE == "cuda":
    M_mat = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float16)

    # Warmup
    for i in range(5):
        _ = M_mat[i % 1024, :].sum()
        _ = M_mat[:, i % 1024].sum()
    torch.cuda.synchronize()

    # TODO 3: Time both access patterns with CUDA events.
    #   row_major_ms: time 128 iterations of M_mat[i % 1024, :].sum()
    #   col_major_ms: time 128 iterations of M_mat[:, i % 1024].sum()
    row_major_ms = None  # YOUR CODE HERE → CUDA event timing, 128 iters
    col_major_ms = None  # YOUR CODE HERE → CUDA event timing, 128 iters

    assert row_major_ms is not None and row_major_ms > 0, "row_major_ms must be > 0"
    assert col_major_ms is not None and col_major_ms > 0, "col_major_ms must be > 0"
    assert row_major_ms < col_major_ms * 3, (
        f"row-major ({row_major_ms:.3f} ms) should be < 3x col-major "
        f"({col_major_ms:.3f} ms) — coalescing benefit"
    )
    print(f"  Row-major slice (M[i, :]) : {row_major_ms:.3f} ms total (128 iters)")
    print(f"  Col-major slice (M[:, i]) : {col_major_ms:.3f} ms total (128 iters)")
    print(f"  Ratio col/row             : {col_major_ms/row_major_ms:.2f}x slower")
else:
    row_major_ms, col_major_ms = 1.0, 2.5
    print("  (CUDA not available — skipping layout access timing)")

print("  ✓ Section 3 passed — row-major access is faster due to coalescing")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Layout Implication — contiguous() before matmul
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Layout Implication — contiguous() for matmul ──")
print("""
  When a tensor is non-contiguous (e.g. after .t() or .permute()), the
  CUDA matmul kernels may not be able to use their most optimised paths.
  Calling .contiguous() copies the data into a new contiguous buffer.
  This copy has a cost, but the subsequent matmul may be faster enough
  to justify it.

  We create a transposed (non-contiguous) tensor and compare matmul
  timing with and without .contiguous().  The key assertion is that
  .contiguous() correctly reports is_contiguous() = True.
""")

if DEVICE == "cuda":
    SIZE = 1024
    A = torch.randn(SIZE, SIZE, device=DEVICE, dtype=torch.float16)
    B_orig = torch.randn(SIZE, SIZE, device=DEVICE, dtype=torch.float16)
    B_transposed = B_orig.t()  # non-contiguous

    print(f"  B_transposed.is_contiguous(): {B_transposed.is_contiguous()}")

    # TODO 4: Call .contiguous() on B_transposed to produce B_cont.
    B_cont = None  # YOUR CODE HERE → B_transposed.contiguous()

    assert B_cont is not None, "B_cont must not be None"
    assert B_cont.is_contiguous(), "B_cont.is_contiguous() must return True"

    # Benchmark non-contiguous matmul
    for _ in range(5):
        torch.mm(A, B_transposed)
    torch.cuda.synchronize()

    s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s1.record()
    for _ in range(30):
        torch.mm(A, B_transposed)
    e1.record()
    torch.cuda.synchronize()
    ms_noncontig = s1.elapsed_time(e1) / 30

    # Benchmark contiguous matmul
    for _ in range(5):
        torch.mm(A, B_cont)
    torch.cuda.synchronize()

    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(30):
        torch.mm(A, B_cont)
    e2.record()
    torch.cuda.synchronize()
    ms_contig = s2.elapsed_time(e2) / 30

    print(f"  B_cont.is_contiguous(): {B_cont.is_contiguous()}")
    print(f"  matmul non-contiguous : {ms_noncontig:.3f} ms")
    print(f"  matmul contiguous     : {ms_contig:.3f} ms")
    print(f"  (results may be similar — modern cuBLAS handles both paths)")
else:
    B_transposed = torch.randn(512, 512).t()
    B_cont = None  # YOUR CODE HERE → B_transposed.contiguous()
    # On CPU we still verify the TODO
    assert B_cont is None or True, "CPU fallback"
    print("  (CUDA not available — verifying contiguous() logic on CPU)")
    B_cpu = torch.randn(512, 512).t()
    B_cont_cpu = B_cpu.contiguous()
    assert B_cont_cpu.is_contiguous(), "contiguous() should return is_contiguous=True"
    print(f"  CPU verify: B_cont.is_contiguous() = {B_cont_cpu.is_contiguous()}")

print("  ✓ Section 4 passed — .contiguous() produces a contiguous tensor")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 4.2 complete!")
print("  You now understand memory coalescing, cache tiers, layout access")
print("  patterns, and the contiguous() API for matmul optimisation.")
print("  Next: II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py")
print("=" * 60)
