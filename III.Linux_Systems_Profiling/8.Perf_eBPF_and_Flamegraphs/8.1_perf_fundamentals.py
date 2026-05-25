#!/usr/bin/env python3
"""
8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py  ─  Chapter 8: perf stat and Hardware Counters
=======================================================================
Covers book section 8.1:
  • IPC (Instructions Per Cycle) and what it reveals about CPU efficiency
  • Cache miss rate — sequential vs random access patterns
  • Branch misprediction cost and the sorted-array trick
  • perf stat command reference for AI workloads

Run:  python III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.1_perf_fundamentals.py
All sections must print ✓.
"""

import time
import numpy as np
import torch

print("=" * 60)
print("  Exercise 8.1 — perf stat and Hardware Counters")
print("=" * 60)

DEVICE = "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: IPC and CPU Efficiency
# ─────────────────────────────────────────────────────────────
print("── Section 1: IPC and CPU Efficiency ──")
print("""
  IPC (Instructions Per Cycle) measures how efficiently the CPU
  executes your code.  Modern superscalar CPUs can issue 3–5
  instructions per cycle when their pipelines are full — executing
  multiple independent instructions in parallel on different
  execution units.

  Memory-bound code has LOW IPC because the CPU stalls waiting for
  data to arrive from cache or DRAM.  While waiting for one load
  to complete, the CPU cannot make progress on instructions that
  depend on that value.

  The rule:
    IPC > 2.0 → compute-bound (pipelines are full, CPU is busy)
    IPC < 1.0 → memory-bound  (CPU stalls waiting for data)
    IPC 1–2   → balanced      (mixed memory and compute pressure)

  We measure IPC indirectly by timing two access patterns:
    small_array.sum()  → fits in L1 cache → high IPC (no stalls)
    large_random.sum() → random DRAM access → low IPC (constant stalls)

  perf stat shows the raw instruction and cycle counts:
    sudo perf stat -e instructions,cycles python 8.1_perf_fundamentals.py
""")

def measure_access_time(arr: np.ndarray, indices: np.ndarray) -> float:
    """
    TODO 1: Implement this function.
    Sum arr[indices] using a Python loop (not numpy vectorisation).
    This exposes the per-access cost because Python cannot batch-optimise it.
    Return elapsed time in milliseconds.

    Steps:
      t0 = time.perf_counter()
      total = 0
      for i in indices:
          total += arr[i]
      elapsed_ms = (time.perf_counter() - t0) * 1000
      return elapsed_ms
    """
    pass  # YOUR CODE HERE → return elapsed_ms


# Build test arrays
small_arr = np.arange(1024, dtype=np.float32)          # 4 KB — fits in L1
small_idx = np.arange(len(small_arr), dtype=np.int64)

large_arr = np.random.rand(4 * 1024 * 1024).astype(np.float32)  # 16 MB — larger than L2
rng = np.random.default_rng(42)
random_idx = rng.integers(0, len(large_arr), size=2048, dtype=np.int64)

small_time_ms = measure_access_time(small_arr, small_idx)
large_time_ms = measure_access_time(large_arr, random_idx)

assert small_time_ms is not None and small_time_ms > 0, (
    "measure_access_time must return a positive float in ms. "
    "Did you implement the TODO?"
)
assert large_time_ms is not None and large_time_ms > 0, (
    "measure_access_time on large random array must return > 0 ms."
)

print(f"  Small array (L1 fit, 1024 elements):  {small_time_ms:.3f} ms")
print(f"  Large array (DRAM, 2048 random):       {large_time_ms:.3f} ms")
print(f"  Ratio (large/small per element):       "
      f"{(large_time_ms/len(random_idx)) / (small_time_ms/len(small_idx)):.1f}×")
print("  ✓ Section 1 passed — IPC cost measured via Python loop timing")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Cache Miss Rate — Sequential vs Random
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Cache Miss Rate — Sequential vs Random ──")
print("""
  Each cache miss stalls the CPU for roughly 200 cycles — the time
  needed to read a 64-byte cache line from DRAM.

  Sequential access lets the hardware prefetcher predict which cache
  lines will be needed next.  When it detects a stride-1 access
  pattern (arr[0], arr[1], arr[2], …) it begins fetching future
  cache lines before the CPU requests them.  The load arrives before
  the stall begins — the CPU never waits.

  Random access destroys prefetching.  The prefetcher cannot predict
  which cache line will be needed next, so the CPU must request it
  on demand and wait ~200 cycles for the DRAM to respond.

  The miss ratio = random_time / sequential_time measures the
  empirical penalty of random vs sequential access on your hardware.
  Values of 3× to 10× are typical for large arrays.

  perf stat can show the hardware cache miss count:
    sudo perf stat -e cache-misses,cache-references python 8.1_perf_fundamentals.py
""")

def sequential_sum(arr: np.ndarray) -> float:
    """
    TODO 2a: Implement this function.
    Sum the entire array using numpy (arr.sum()).
    Return the elapsed time in milliseconds.
    Run the sum 5 times and return the minimum (to reduce OS noise).
    """
    pass  # YOUR CODE HERE → return min_elapsed_ms


def random_gather(arr: np.ndarray, idx: np.ndarray) -> float:
    """
    TODO 2b: Implement this function.
    Gather arr[idx] and sum the result (arr[idx].sum()).
    Return the elapsed time in milliseconds (minimum over 5 runs).
    """
    pass  # YOUR CODE HERE → return min_elapsed_ms


# 256 MB float32 array — larger than any typical L3 cache
N = 64 * 1024 * 1024   # 64M elements = 256 MB
big_arr = np.random.rand(N).astype(np.float32)

# Random index covering the full array (defeats prefetcher)
rand_idx_large = rng.integers(0, N, size=N // 16, dtype=np.int64)

seq_ms = sequential_sum(big_arr)
rnd_ms = random_gather(big_arr, rand_idx_large)

assert seq_ms is not None and seq_ms > 0, (
    "sequential_sum must return a positive elapsed time in ms. "
    "Did you implement TODO 2a?"
)
assert rnd_ms is not None and rnd_ms > 0, (
    "random_gather must return a positive elapsed time in ms. "
    "Did you implement TODO 2b?"
)

miss_ratio = rnd_ms / seq_ms
assert miss_ratio > 2.0, (
    f"miss_ratio = random_ms / sequential_ms should be > 2.0, got {miss_ratio:.2f}. "
    f"seq_ms={seq_ms:.1f} rnd_ms={rnd_ms:.1f}. "
    "Check that big_arr is 256 MB (larger than L3 cache) and idx covers the full range."
)

print(f"  Array size:           {N * 4 / 1e6:.0f} MB (larger than L3)")
print(f"  Sequential sum:       {seq_ms:.1f} ms")
print(f"  Random gather sum:    {rnd_ms:.1f} ms")
print(f"  Miss ratio:           {miss_ratio:.1f}× (random is {miss_ratio:.1f}× slower)")
print("  ✓ Section 2 passed — cache miss ratio > 2× (random much slower than sequential)")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Branch Misprediction Cost
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Branch Misprediction Cost ──")
print("""
  Modern CPUs predict branches before knowing which way they go.
  The branch predictor uses history — if the last 100 iterations
  of a loop always took the same path, it predicts the same again.

  When the prediction is wrong, the CPU must flush the pipeline and
  discard all speculatively-executed instructions.  This costs
  15–20 cycles per misprediction on a modern CPU.

  The classic example: summing elements where value > 0.5.
  On an unsorted random array, half the elements satisfy the
  condition, half do not — the branch alternates unpredictably,
  causing many mispredictions.  After sorting, all elements below
  0.5 appear first, then all above.  The branch flips exactly once
  at the threshold — the predictor gets almost everything right.

  In practice: sorting input data before a conditional operation
  is one of the few pure-software cures for branch misprediction.
  DataLoader shuffling trades training accuracy benefits against
  branch predictor efficiency in your preprocessing transforms.

  perf stat measures mispredictions directly:
    sudo perf stat -e branch-misses,branches python 8.1_perf_fundamentals.py
""")

SIZE = 4 * 1024 * 1024   # 4M elements — large enough for statistically visible timing

unsorted_arr = rng.random(SIZE).astype(np.float32)
sorted_arr = np.sort(unsorted_arr)

# Measure time to sum elements > 0.5 on both arrays
THRESHOLD = 0.5

def time_conditional_sum(arr: np.ndarray, threshold: float = 0.5) -> tuple:
    """
    TODO 3: Implement this function.
    Measure time (minimum over 5 runs) to compute:
        result = arr[arr > threshold].sum()
    Return (elapsed_ms, result_value) so we can verify both arrays
    give the same sum (within floating point tolerance).
    """
    pass  # YOUR CODE HERE → return (elapsed_ms, result_value)


unsorted_result = time_conditional_sum(unsorted_arr, THRESHOLD)
sorted_result = time_conditional_sum(sorted_arr, THRESHOLD)

assert unsorted_result is not None, "time_conditional_sum must return (ms, value). Did you implement TODO 3?"
assert sorted_result is not None,   "time_conditional_sum must return (ms, value). Did you implement TODO 3?"

unsorted_ms, unsorted_sum = unsorted_result
sorted_ms,   sorted_sum   = sorted_result

assert unsorted_ms > 0, "unsorted_ms must be > 0"
assert sorted_ms > 0,   "sorted_ms must be > 0"
assert abs(unsorted_sum - sorted_sum) < 1.0, (
    f"Both arrays have the same elements — sums must match. "
    f"unsorted={unsorted_sum:.2f}, sorted={sorted_sum:.2f}"
)

print(f"  Array size:           {SIZE * 4 / 1e6:.0f} MB")
print(f"  Unsorted conditional: {unsorted_ms:.2f} ms  sum={unsorted_sum:.1f}")
print(f"  Sorted conditional:   {sorted_ms:.2f} ms   sum={sorted_sum:.1f}")
print(f"  Sums match:           {abs(unsorted_sum - sorted_sum) < 1.0}")
print("  ✓ Section 3 passed — both arrays produce same sum; timing measured")


# ─────────────────────────────────────────────────────────────
# SECTION 4: perf stat Command Reference
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: perf stat Command Reference ──")
print("""
  perf stat is Linux's performance counter tool.  It wraps a command
  and reports hardware-level event counts when the command exits.
  It reads CPU performance monitoring units (PMUs) — actual hardware
  counters in silicon, not estimates.

  The five most important perf stat events for AI workloads:

  ┌─────────────────────┬──────────────────────────────┬───────────┬────────────────┐
  │ Event               │ What it measures             │ Good      │ Bad            │
  ├─────────────────────┼──────────────────────────────┼───────────┼────────────────┤
  │ cache-misses        │ LLC (L3) cache misses         │ < 0.1%    │ > 5% of refs   │
  │ cache-references    │ All LLC cache accesses        │ baseline  │ —              │
  │ branch-misses       │ Branch predictor failures     │ < 1%      │ > 5% of branches│
  │ instructions        │ Total instructions retired    │ high      │ low for work done│
  │ cycles              │ Total CPU cycles elapsed      │ baseline  │ —              │
  └─────────────────────┴──────────────────────────────┴───────────┴────────────────┘

  IPC = instructions / cycles

  Commands to run in a separate terminal while your AI workload runs:

    # Five core counters — IPC, cache, and branch:
    sudo perf stat -e cache-misses,cache-references,branch-misses,instructions,cycles \\
        python 8.1_perf_fundamentals.py

    # Detailed cache hierarchy — L1, L2, LLC miss rates:
    sudo perf stat -e L1-dcache-misses,L2-dcache-misses,LLC-misses \\
        python 8.1_perf_fundamentals.py

    # For a running process (attach without restarting):
    sudo perf stat -p <PID> sleep 5

    # Full system-wide snapshot (all CPUs, 10 seconds):
    sudo perf stat -a sleep 10

  Reading the output:
    instructions / cycles → IPC.  A100-class CPU code often sees IPC 1.5–2.5
    during data preprocessing.  Values below 1.0 indicate severe memory stalls.

    cache-misses / cache-references → miss rate.  Above 5% on a DataLoader
    workload suggests the dataset does not fit in L3 and every item fetch
    is a DRAM access.

    branch-misses / branches → misprediction rate.  Above 3% on conditional
    augmentation code suggests the branches are data-dependent and unpredictable.
    Consider re-ordering data or vectorising the condition with numpy.
""")

def interpret_ipc(instructions: float, cycles: float) -> str:
    """
    TODO 4: Implement this function.
    Compute IPC = instructions / cycles and return a classification string:
      IPC > 2.0  → "compute-bound"
      IPC < 1.0  → "memory-bound"
      else       → "balanced"
    """
    pass  # YOUR CODE HERE → return classification string


assert interpret_ipc(3e9, 1e9) == "compute-bound", (
    f"interpret_ipc(3e9, 1e9): IPC=3.0 → should be 'compute-bound', "
    f"got '{interpret_ipc(3e9, 1e9)}'"
)
assert interpret_ipc(5e8, 1e9) == "memory-bound", (
    f"interpret_ipc(5e8, 1e9): IPC=0.5 → should be 'memory-bound', "
    f"got '{interpret_ipc(5e8, 1e9)}'"
)
assert interpret_ipc(1.5e9, 1e9) == "balanced", (
    f"interpret_ipc(1.5e9, 1e9): IPC=1.5 → should be 'balanced', "
    f"got '{interpret_ipc(1.5e9, 1e9)}'"
)

print(f"  interpret_ipc(3e9, 1e9)   = '{interpret_ipc(3e9, 1e9)}'   (IPC=3.0)")
print(f"  interpret_ipc(5e8, 1e9)   = '{interpret_ipc(5e8, 1e9)}'  (IPC=0.5)")
print(f"  interpret_ipc(1.5e9, 1e9) = '{interpret_ipc(1.5e9, 1e9)}'      (IPC=1.5)")
print("  ✓ Section 4 passed — IPC interpretation function correct")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 8.1 complete!")
print()
print("  You can now measure IPC cost with Python timing, explain")
print("  sequential vs random cache miss rates, understand branch")
print("  misprediction, and interpret perf stat output.")
print()
print("  Next: III.Linux_Systems_Profiling/8.Perf_eBPF_and_Flamegraphs/8.2_cpu_flamegraphs.py")
print("=" * 60)
