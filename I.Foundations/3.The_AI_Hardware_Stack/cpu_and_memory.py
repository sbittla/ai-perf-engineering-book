#!/usr/bin/env python3
"""
exercise_01_cpu_and_memory.py  ─  Chapter 3: CPU Architecture & PCIe
=====================================================================
Covers book sections 3.1 and 3.2:
  • CPU cache hierarchy: L1 → L2 → L3 → DRAM latency and bandwidth
  • Cache miss effects on DataLoader-style random access patterns
  • PCIe transfers: CPU→GPU and GPU→CPU bandwidth measurement
  • Pinned (page-locked) memory vs pageable memory transfer speed
  • non_blocking=True for overlapping copies with compute

This exercise runs on CPU-only machines for Sections 1–2.
Sections 3–4 require CUDA.

Run:  python exercise_01_cpu_and_memory.py
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("=" * 65)
print("  Exercise 01 — CPU Architecture & PCIe Transfers")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: CPU cache hierarchy — sequential vs random access
# ─────────────────────────────────────────────────────────────
print("── Section 1: CPU Cache Effects ──")
print("""
  The CPU has a multilevel cache: L1 (~32 KB, ~4 cycles) →
  L2 (~512 KB, ~12 cycles) → L3 (~32 MB, ~40 cycles) → DRAM (~200 cycles).

  Sequential access (stride-1) stays in L1/L2 cache → fast.
  Random access scatters reads across DRAM → slow (cache misses).

  For DataLoaders: random shuffling of large datasets causes L3 misses
  and reduces throughput.  Prefetching and larger shuffle buffers help.
""")

def time_access(tensor: torch.Tensor, sequential: bool, n_accesses: int) -> float:
    """Return time in ms for n_accesses reads from tensor."""
    size = tensor.numel()
    if sequential:
        indices = torch.arange(n_accesses) % size
    else:
        indices = torch.randint(0, size, (n_accesses,))

    t0 = time.perf_counter()
    _ = tensor[indices].sum()   # force evaluation
    return (time.perf_counter() - t0) * 1000

N_ACCESSES = 500_000

# Small tensor: fits in L1/L2 cache (~512 KB = 128K float32 elements)
small = torch.randn(32_768)   # 128 KB

# Large tensor: exceeds L3 cache (~256 MB)
large = torch.randn(64_000_000)   # 256 MB

# Warmup
for _ in range(3):
    time_access(small, sequential=True, n_accesses=10_000)
    time_access(large, sequential=True, n_accesses=10_000)

t_small_seq  = time_access(small, sequential=True,  n_accesses=N_ACCESSES)
t_small_rand = time_access(small, sequential=False, n_accesses=N_ACCESSES)
t_large_seq  = time_access(large, sequential=True,  n_accesses=N_ACCESSES)
t_large_rand = time_access(large, sequential=False, n_accesses=N_ACCESSES)

# TODO 1: Compute speedup: how much faster is sequential vs random for the large tensor?
speedup_large = None  # YOUR CODE HERE  → t_large_rand / t_large_seq

assert speedup_large is not None, "compute speedup_large"
print(f"  Small tensor (128 KB — fits in L1/L2):")
print(f"    Sequential : {t_small_seq:.1f} ms   Random: {t_small_rand:.1f} ms")
print(f"  Large tensor (256 MB — exceeds L3, hits DRAM):")
print(f"    Sequential : {t_large_seq:.1f} ms   Random: {t_large_rand:.1f} ms")
print(f"  Sequential vs random speedup (large) : {speedup_large:.1f}×")
print(f"  Interpretation: random access on DRAM-resident data is {speedup_large:.1f}× slower")
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 2: DataLoader access pattern impact
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: DataLoader Sequential vs Shuffled Access ──")
print("""
  shuffle=True in DataLoader randomises the sample order each epoch.
  For large datasets that don't fit in RAM, this causes DRAM misses.
  Practical mitigations: pre-shuffle once, use WebDataset (sequential
  streaming), or buffer-shuffle (shuffle only within a window).
""")

# Simulate a dataset that partially fits in CPU cache vs one that does not
N_SAMPLES = 10_000
FEAT_DIM  = 128

data   = torch.randn(N_SAMPLES, FEAT_DIM)   # ~5 MB — fits in L3
labels = torch.randint(0, 10, (N_SAMPLES,))
ds     = TensorDataset(data, labels)

# TODO 2: Create two DataLoaders — one with shuffle=False, one with shuffle=True
loader_seq  = None  # YOUR CODE HERE  → DataLoader(ds, batch_size=64, shuffle=False)
loader_shuf = None  # YOUR CODE HERE  → DataLoader(ds, batch_size=64, shuffle=True)

assert loader_seq  is not None, "create loader_seq"
assert loader_shuf is not None, "create loader_shuf"

def time_loader(loader) -> float:
    t0 = time.perf_counter()
    for xb, yb in loader:
        _ = xb.sum()   # force read
    return (time.perf_counter() - t0) * 1000

# Warmup
for _ in range(2): time_loader(loader_seq)

t_seq  = time_loader(loader_seq)
t_shuf = time_loader(loader_shuf)

print(f"  Sequential loader : {t_seq:.1f} ms  (cache-friendly)")
print(f"  Shuffled loader   : {t_shuf:.1f} ms  (random access — more cache misses)")
print(f"  Note: this small dataset fits in L3 — difference is small.")
print(f"  On a 100 GB dataset the difference can be 10–50×.")
print("  ✓ Section 2 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 3: PCIe transfer bandwidth
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: PCIe Transfer Bandwidth ──")

if DEVICE != "cuda":
    print("  (CUDA not available — Sections 3 and 4 skipped)")
    print("  When you have a GPU, re-run this exercise to measure PCIe bandwidth.")
else:
    print("""
  PCIe 4.0 ×16: theoretical 32 GB/s bidirectional.
  Practical effective bandwidth: 12–20 GB/s (protocol overhead).
  Measuring it directly shows you how expensive tensor.to('cuda') is.
    """)

    SIZES_MB = [1, 8, 32, 128, 512]

    print(f"  {'Size':>8}  {'HtoD (GB/s)':>12}  {'DtoH (GB/s)':>12}  {'Time HtoD':>10}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*12}  {'-'*10}")

    for size_mb in SIZES_MB:
        n_floats = size_mb * 1_000_000 // 4   # float32
        cpu_t  = torch.randn(n_floats)        # pageable CPU memory
        gpu_t  = torch.empty(n_floats, device=DEVICE)

        # Warmup
        for _ in range(3):
            tmp = cpu_t.cuda(non_blocking=False)
            torch.cuda.synchronize()

        # Host-to-Device (HtoD)
        ITERS = 5
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(ITERS):
            tmp = cpu_t.cuda()
        e.record()
        torch.cuda.synchronize()
        htod_ms  = s.elapsed_time(e) / ITERS
        # TODO 3: Compute HtoD bandwidth in GB/s
        #   bandwidth = size_mb / 1000 / (htod_ms / 1000)
        htod_bw = None  # YOUR CODE HERE

        # Device-to-Host (DtoH)
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s2.record()
        for _ in range(ITERS):
            tmp2 = gpu_t.cpu()
        e2.record()
        torch.cuda.synchronize()
        dtoh_ms = s2.elapsed_time(e2) / ITERS
        # TODO 4: Compute DtoH bandwidth in GB/s
        dtoh_bw = None  # YOUR CODE HERE

        assert htod_bw is not None, f"compute htod_bw for {size_mb}MB"
        assert dtoh_bw is not None, f"compute dtoh_bw for {size_mb}MB"
        print(f"  {size_mb:>6} MB  {htod_bw:>12.2f}  {dtoh_bw:>12.2f}  {htod_ms:>9.2f} ms")

    print("  ✓ Section 3 passed — compare your numbers to PCIe spec")

    # ─────────────────────────────────────────────────────────
    # SECTION 4: Pinned memory vs pageable memory
    # ─────────────────────────────────────────────────────────
    print("\n── Section 4: Pinned Memory (pin_memory=True) ──")
    print("""
  Pageable memory: CPU can swap pages → GPU DMA must copy to a pinned
                   staging buffer first, then DMA. Two copies total.
  Pinned memory:  page-locked → GPU DMA reads directly.  One copy.

  DataLoader pin_memory=True pre-allocates output tensors as pinned.
  Combined with non_blocking=True, copies overlap with GPU compute.
    """)

    SIZE_MB = 64
    n_floats = SIZE_MB * 1_000_000 // 4

    # Pageable (normal)
    pageable = torch.randn(n_floats)

    # TODO 5: Create a pinned CPU tensor of the same size
    #   torch.empty(n_floats).pin_memory()
    pinned = None  # YOUR CODE HERE
    pinned.copy_(pageable)   # fill with same data

    assert pinned is not None,         "create pinned tensor"
    assert pinned.is_pinned(),         "tensor should be pinned"

    def measure_htod(cpu_tensor, iters=10):
        # Warmup
        for _ in range(3):
            tmp = cpu_tensor.cuda(); torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            tmp = cpu_tensor.cuda()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_pageable_ms = measure_htod(pageable)
    t_pinned_ms   = measure_htod(pinned)

    # TODO 6: Compute speedup of pinned over pageable
    pinned_speedup = None  # YOUR CODE HERE  → t_pageable_ms / t_pinned_ms

    assert pinned_speedup is not None, "compute pinned_speedup"
    print(f"  Transfer size          : {SIZE_MB} MB")
    print(f"  Pageable → GPU         : {t_pageable_ms:.2f} ms")
    print(f"  Pinned   → GPU         : {t_pinned_ms:.2f} ms")
    print(f"  Pinned speedup         : {pinned_speedup:.2f}×")
    print(f"  Expected: ~1.5–2× faster with pinned memory")
    print("  ✓ Section 4 passed")

print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Exercise 01 (Chapter 3) done!")
print()
print("  Key takeaways:")
print("    • Random memory access on DRAM-resident data is 5–20× slower")
print("      than sequential — explains DataLoader shuffle overhead")
print("    • PCIe effective bandwidth: 12–20 GB/s (not the 32 GB/s spec)")
print("    • Pinned memory cuts CPU→GPU transfer time by ~2×")
print("    • Use pin_memory=True + non_blocking=True in every DataLoader")
print("=" * 65)
