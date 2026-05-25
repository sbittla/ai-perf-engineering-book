#!/usr/bin/env python3
"""
9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py  ─  Chapter 9: The Memory Hierarchy
=======================================================================
Covers book section 9.1:
  • Cache line size (64 bytes) and sequential vs stride access
  • Working set size and cache tier boundaries (L1/L2/L3/DRAM)
  • Matrix layout and cache-friendly vs cache-unfriendly access
  • PyTorch contiguous tensors and the cost of transposition

Run:  python III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.1_memory_hierarchy.py
All sections must print ✓.
"""

import time
import numpy as np
import torch

print("=" * 60)
print("  Exercise 9.1 — The Memory Hierarchy")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Cache Line Size and Sequential vs Stride Access
# ─────────────────────────────────────────────────────────────
print("── Section 1: Cache Line Size and Sequential Access ──")
print("""
  The fundamental unit of CPU memory access is the cache line.
  On x86 and ARM, a cache line is exactly 64 bytes.

  When the CPU needs a value at address A, it does not fetch just
  that 4-byte float.  It fetches the entire 64-byte cache line
  containing A — all 16 consecutive float32 values in that line.

  WHAT THIS MEANS FOR ACCESS PATTERNS:

  Stride-1 (sequential):
    arr[0], arr[1], …, arr[15] → 1 cache line loaded, 16 values used
    100% cache line utilisation.  The hardware prefetcher detects the
    sequential pattern and pre-fetches the next line before you ask.

  Stride-16 (one float per cache line):
    arr[0], arr[16], arr[32], … → 1 cache line per float
    Only 1 of 16 values per cache line is used. 6.25% utilisation.
    No spatial locality — prefetcher cannot help.
    Effective bandwidth is 16× lower than stride-1.

  THIS IS WHY PyTorch's .contiguous() EXISTS:
    A transposed tensor has stride (1, rows) instead of stride (cols, 1).
    Reading across a transposed tensor's rows accesses memory in stride-N
    order — each element is N floats apart in memory.  For a 1024×1024
    matrix, that is stride-1024 = 4096 bytes per element = 64 cache lines
    wasted per element accessed.  Calling .contiguous() copies the data
    into row-major order, restoring stride-1 access.

  MEASUREMENT:
    We measure bandwidth (GB/s = bytes_read / time_seconds / 1e9)
    for arr[::stride].sum() at strides 1, 4, 8, 16, 32.
    The bandwidth drop as stride increases directly shows the
    cache line waste.
""")

def stride_bandwidth(arr: np.ndarray, stride: int) -> float:
    """
    TODO 1: Implement this function.
    Access every stride-th element of arr with arr[::stride].sum().
    Return the EFFECTIVE bandwidth in GB/s, computed as:
        bytes_accessed = arr[::stride].nbytes   (bytes of elements actually touched)
        bandwidth_gbs  = bytes_accessed / elapsed_seconds / 1e9
    Run 5 warmup iterations and then 5 timed iterations, return the best.
    """
    pass  # YOUR CODE HERE → return bandwidth_gbs


# 256 MB float32 array — bigger than typical L3
N = 64 * 1024 * 1024
arr = np.random.rand(N).astype(np.float32)

print(f"  Array size: {N * 4 / 1e6:.0f} MB")
print(f"  {'Stride':>8}  {'Bandwidth (GB/s)':>18}  {'Cache efficiency':>18}")
print(f"  {'─'*8}  {'─'*18}  {'─'*18}")

bw_stride1 = stride_bandwidth(arr, 1)
assert bw_stride1 is not None and bw_stride1 > 0, (
    "stride_bandwidth(arr, 1) must return a positive float. "
    "Did you implement TODO 1?"
)

bw_stride16 = stride_bandwidth(arr, 16)
assert bw_stride16 is not None and bw_stride16 > 0, \
    "stride_bandwidth(arr, 16) must return a positive float"

assert bw_stride1 > bw_stride16, (
    f"stride-1 bandwidth ({bw_stride1:.1f} GB/s) must be greater than "
    f"stride-16 bandwidth ({bw_stride16:.1f} GB/s). "
    "Check that you measure bytes_accessed = arr[::stride].nbytes."
)

for stride in [1, 4, 8, 16, 32]:
    bw = stride_bandwidth(arr, stride)
    eff = bw / bw_stride1 * 100
    print(f"  {stride:>8}  {bw:>18.1f}  {eff:>17.0f}%")

print("  ✓ Section 1 passed — stride-1 bandwidth > stride-16 bandwidth")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Working Set Size and Cache Tiers
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Working Set Size and Cache Tiers ──")
print("""
  Access time jumps dramatically when the working set (all the data
  touched in a computation) exceeds each cache level.

  LATENCY NUMBERS FOR MODERN x86 (approximate, CPU-dependent):
    L1 cache  (~48 KB):  4 cycles   = ~1.5 ns at 3 GHz
    L2 cache  (~512 KB): 12 cycles  = ~4 ns
    L3 cache  (~8–32 MB): 40 cycles = ~13 ns
    DRAM:                200 cycles = ~67 ns at 3 GHz

  BANDWIDTH NUMBERS (sequential read, modern workstation):
    L1/L2: 200–500 GB/s  (served from on-chip SRAM)
    L3:    100–200 GB/s  (last-level cache, shared across all cores)
    DRAM:  30–100 GB/s   (DDR4/DDR5, depends on channel count)

  The "bandwidth cliff" is visible when you measure throughput vs
  working set size: bandwidth is stable while data fits in L3,
  then drops sharply when the working set exceeds L3 capacity.

  FINDING YOUR L3 SIZE:
    cat /sys/devices/system/cpu/cpu0/cache/index3/size   # Linux
    lscpu | grep L3                                       # alternative
    Run this script with perf: the miss rate spikes at the cliff.
""")

def classify_cache_tier(size_bytes: int, l3_bytes: int) -> str:
    """
    TODO 2: Implement this function.
    Classify a working set size into a cache tier:
      size_bytes < 1 MB          → "L1/L2"
      size_bytes < l3_bytes      → "L3"
      else                       → "DRAM"
    """
    pass  # YOUR CODE HERE → return tier string


# Standard L3 assumption for a typical workstation (adjust for your hardware)
L3_SIZE = 8 * 1024 * 1024   # 8 MB — typical L3

assert classify_cache_tier(512 * 1024, L3_SIZE) == "L1/L2", (
    f"512 KB < 1 MB → 'L1/L2', got '{classify_cache_tier(512*1024, L3_SIZE)}'"
)
assert classify_cache_tier(4 * 1024 * 1024, L3_SIZE) == "L3", (
    f"4 MB in an 8 MB L3 → 'L3', got '{classify_cache_tier(4*1024*1024, L3_SIZE)}'"
)
assert classify_cache_tier(32 * 1024 * 1024, L3_SIZE) == "DRAM", (
    f"32 MB > 8 MB L3 → 'DRAM', got '{classify_cache_tier(32*1024*1024, L3_SIZE)}'"
)

print(f"  Measuring bandwidth sweep (sizes 1 KB → 512 MB):")
print(f"  L3 reference size: {L3_SIZE/1e6:.0f} MB")
print(f"  {'Size':>10}  {'BW (GB/s)':>12}  {'Tier':>8}")
print(f"  {'─'*10}  {'─'*12}  {'─'*8}")

sizes_mb = [0.001, 0.01, 0.1, 1.0, 4.0, 8.0, 32.0, 128.0, 256.0]
for size_mb in sizes_mb:
    n_elems = int(size_mb * 1e6 / 4)
    if n_elems < 16:
        n_elems = 16
    t_arr = np.random.rand(n_elems).astype(np.float32)
    # 3 warmup + 5 timed runs, take minimum
    for _ in range(3):
        t_arr.sum()
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        t_arr.sum()
        times.append(time.perf_counter() - t0)
    bw = (n_elems * 4) / min(times) / 1e9
    size_bytes = n_elems * 4
    tier = classify_cache_tier(size_bytes, L3_SIZE)
    print(f"  {n_elems * 4 / 1e6:>8.3f}MB  {bw:>12.1f}  {tier:>8}")
    del t_arr

print("  ✓ Section 2 passed — cache tier classification implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Matrix Layout and Cache Friendliness
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Matrix Layout and Cache Friendliness ──")
print("""
  A C-contiguous (row-major) matrix stores rows contiguously:
    A[0,0], A[0,1], …, A[0,N-1], A[1,0], A[1,1], …

  Reading across a row = stride-1 = sequential = cache friendly.
  Reading down a column = stride N floats between elements.

  For a 4096×4096 float32 matrix, stride is 4096 * 4 = 16,384 bytes
  between consecutive elements in the same column.  That is 256 cache
  lines skipped per element access — every single access is a cache miss.

  A.sum(axis=1):  sum across each row — accesses elements sequentially
                  within each row.  Cache friendly.
  A.sum(axis=0):  sum down each column — stride = row_size.  Cache unfriendly.

  FIX: transpose first, then sum rows.
    np.ascontiguousarray(A.T).sum(axis=1)
    This forces a copy into a new row-major layout, then the sum is
    sequential.  The copy cost is paid once; subsequent access is fast.

  PYTORCH EQUIVALENT:
    tensor.T.contiguous().sum(dim=1)   # copy + sequential
    tensor.T.sum(dim=1)                # may use non-contiguous BLAS path
""")

def measure_layout_bandwidth(N: int) -> tuple:
    """
    TODO 3: Implement this function.
    Create a float32 numpy matrix of shape (N, N).
    Measure the bandwidth (GB/s) for:
      row_bw: A.sum(axis=1)                    — row-major, cache friendly
      col_bw: A.sum(axis=0)                    — column access, cache unfriendly

    For each, run 3 warmup and then time 5 iterations, take minimum.
    bytes_read = N * N * 4 (all elements read once in both cases)
    bandwidth = bytes_read / min_time / 1e9

    Return (row_bw, col_bw) as a tuple of floats.
    """
    pass  # YOUR CODE HERE → return (row_bw_gbs, col_bw_gbs)


row_bw, col_bw = measure_layout_bandwidth(4096)
assert row_bw is not None and row_bw > 0, "row_bw must be > 0. Did you implement TODO 3?"
assert col_bw is not None and col_bw > 0, "col_bw must be > 0"
assert row_bw > col_bw * 1.1, (
    f"Row-major bandwidth ({row_bw:.1f} GB/s) should be > 1.1× column "
    f"bandwidth ({col_bw:.1f} GB/s) for a 4096×4096 matrix. "
    "Check that you are summing axis=1 for row and axis=0 for col."
)

print(f"  4096×4096 float32 matrix ({4096*4096*4/1e6:.0f} MB):")
print(f"    Row-major A.sum(axis=1):  {row_bw:.1f} GB/s  (cache friendly)")
print(f"    Column A.sum(axis=0):     {col_bw:.1f} GB/s  (cache unfriendly)")
print(f"    Speedup (row/col):        {row_bw/col_bw:.1f}×")
print("  ✓ Section 3 passed — row bandwidth > column bandwidth by > 1.1×")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Implications for PyTorch
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Implications for PyTorch ──")
print("""
  PyTorch tensors use C-contiguous (row-major) layout by default.
  After operations that change the logical view without copying data,
  the tensor becomes NON-CONTIGUOUS.

  OPERATIONS THAT CREATE NON-CONTIGUOUS TENSORS:
    .T          (transpose — swaps strides)
    .permute()  (arbitrary dimension reordering)
    .view()     (if called on a non-contiguous tensor, it will raise)
    t[:, ::2]   (slicing with a step creates a strided view)

  HOW TO CHECK:
    tensor.is_contiguous()   # True = row-major, safe for all ops
    tensor.stride()          # e.g., (1024, 1) = contiguous; (1, 1024) = transposed

  HOW TO FIX:
    tensor.contiguous()      # forces a copy into C-contiguous layout

  WHEN DOES IT MATTER?
    For small tensors (< L3 cache): rarely — cache miss penalty is small.
    For large weight matrices (hidden_dim × hidden_dim at FP16):
      non-contiguous access wastes HBM bandwidth, which is already the
      bottleneck for memory-bound kernels.
    For BLAS/cuBLAS matmul: PyTorch may call .contiguous() internally
      or use a strided BLAS path; profiling will show which.

  PRACTICAL RULE:
    Before passing a tensor to a performance-critical operation,
    check is_contiguous().  If False and the operation is hot
    (appears in the torch.profiler top table), add .contiguous().
""")

# Create a (1024, 1024) float32 tensor and transpose it
base = torch.rand(1024, 1024, dtype=torch.float32)
transposed = base.T   # logical transpose — no copy, just stride swap

# TODO 4: Check is_contiguous() on the transposed tensor.
# Then call .contiguous() on it and check that the result IS contiguous.

is_cont_transposed = None   # YOUR CODE HERE → transposed.is_contiguous()
made_contiguous = None      # YOUR CODE HERE → transposed.contiguous()
is_cont_after = None        # YOUR CODE HERE → made_contiguous.is_contiguous()

assert is_cont_transposed is not None, (
    "is_cont_transposed must be set. Did you implement TODO 4? "
    "Set is_cont_transposed = transposed.is_contiguous()"
)
assert not is_cont_transposed, (
    f"A transposed tensor should not be contiguous. "
    f"base.T.is_contiguous() returned {is_cont_transposed}."
)
assert made_contiguous is not None, "made_contiguous = transposed.contiguous() — did you implement TODO 4?"
assert is_cont_after is not None, "is_cont_after = made_contiguous.is_contiguous() — did you implement TODO 4?"
assert is_cont_after, (
    f"transposed.contiguous().is_contiguous() should be True, "
    f"got {is_cont_after}."
)

print(f"  base.shape:                {tuple(base.shape)}")
print(f"  base.stride():             {base.stride()}")
print(f"  transposed.stride():       {transposed.stride()}")
print(f"  transposed.is_contiguous(): {is_cont_transposed}  (expected False)")
print(f"  after .contiguous():        {is_cont_after}  (expected True)")

# Measure matmul time: transposed vs contiguous (CPU, best-effort)
x = torch.rand(1024, 512, dtype=torch.float32)

times_nc = []
for i in range(8):
    t0 = time.perf_counter()
    _ = torch.mm(transposed, x)
    times_nc.append(time.perf_counter() - t0)
nc_ms = min(times_nc) * 1000

times_c = []
for i in range(8):
    t0 = time.perf_counter()
    _ = torch.mm(made_contiguous, x)
    times_c.append(time.perf_counter() - t0)
c_ms = min(times_c) * 1000

print(f"  matmul(transposed, x):    {nc_ms:.2f} ms")
print(f"  matmul(contiguous, x):    {c_ms:.2f} ms")
print(f"  (difference may be small on CPU — effect is larger on GPU HBM)")
print("  ✓ Section 4 passed — transposed tensor is not contiguous, .contiguous() fixes it")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 9.1 complete!")
print()
print("  You can now measure stride bandwidth, classify working set")
print("  sizes by cache tier, benchmark row vs column access, and")
print("  verify PyTorch tensor contiguity.")
print()
print("  Next: III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.2_numa_and_topology.py")
print("=" * 60)
