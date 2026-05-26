#!/usr/bin/env python3
"""
9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py  ─  Chapter 9: Memory Bandwidth
=======================================================================
Covers book section 9.3:
  • Measuring CPU DRAM bandwidth — the ceiling for memory-bound CPU work
  • PCIe bandwidth — the bottleneck between DataLoader and GPU
  • The full bandwidth hierarchy from GPU registers to SSD
  • Applying the roofline model with measured (not theoretical) bandwidth

Run:  python III.Linux_Systems_Profiling/9.Memory_Hierarchy_and_NUMA/9.3_memory_bandwidth.py
All sections must print ✓.
"""

import time
import torch
import numpy as np

print("=" * 60)
print("  Exercise 9.3 — Memory Bandwidth as a Bottleneck")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: CPU DRAM Bandwidth
# ─────────────────────────────────────────────────────────────
print("── Section 1: CPU DRAM Bandwidth ──")
print("""
  CPU DRAM bandwidth is the maximum rate at which a single CPU can
  read or write main memory.  This is the hard ceiling for any
  memory-bandwidth-limited operation that runs on the CPU.

  TYPICAL VALUES:
    DDR4 dual-channel:  ~50 GB/s  (common workstations, 2021 era)
    DDR5 dual-channel:  ~80 GB/s  (modern workstations, 2022+)
    DDR5 8-channel:     ~300 GB/s (high-end server, 4th-gen Xeon)
    HBM2e on-package:   ~1 TB/s   (some AI accelerators)

  WHY IT IS A HARD LIMIT:
    If your DataLoader preprocessing reads 256 MB per batch and your
    CPU DRAM bandwidth is 50 GB/s, the minimum possible preprocessing
    time is 256 MB / 50 GB/s = 5.1 ms per batch regardless of how
    many CPU cores you use.  Adding more workers does not help once
    you saturate the memory bus.

  MEASURING IT:
    The most accurate method is a streaming copy benchmark:
    allocate two large arrays (>> L3 cache), copy src to dst,
    measure bytes / time.
    Bytes transferred = 2× array size (one read + one write).

    torch.Tensor.clone() performs this copy.  The operation reads all
    bytes from src and writes all bytes to dst — a pure memory bandwidth
    benchmark with no compute overhead.

  CHECKING AGAINST THE LIMIT:
    After measuring, compare to your DRAM spec.
    Typical achievement ratio: 70–90% of theoretical peak.
    If you see < 50%, check for NUMA effects (wrong node) or
    that the array is large enough to overflow L3.
""")

def measure_cpu_bandwidth(size_mb: float) -> float:
    """
    TODO 1: Implement this function.
    Allocate a float32 CPU tensor of the given size in megabytes.
    Time t.clone() for 5 runs (after 3 warmup runs) and return
    the bandwidth in GB/s for the BEST run.

    bytes_transferred = size_mb * 1e6 * 2  (read src + write dst)
    bandwidth = bytes_transferred / elapsed_seconds / 1e9

    Note: use torch.Tensor on CPU (not numpy) to be consistent with
    DataLoader pipeline benchmarks.  size_mb should be large enough
    to overflow the L3 cache.
    """
    pass  # YOUR CODE HERE → return bandwidth_gbs


bw = measure_cpu_bandwidth(256.0)
assert bw is not None, "measure_cpu_bandwidth must return a float. Did you implement TODO 1?"
assert bw > 5.0, (
    f"measure_cpu_bandwidth(256) should return > 5.0 GB/s on any modern system, "
    f"got {bw:.1f} GB/s. Check that size_mb=256 overflows L3 and that "
    f"bytes_transferred = size_mb * 1e6 * 2."
)

print(f"  CPU DRAM bandwidth (256 MB clone): {bw:.1f} GB/s")
print(f"  (Typical: 30–100 GB/s depending on DDR generation)")
print("  ✓ Section 1 passed — CPU DRAM bandwidth > 5 GB/s measured")


# ─────────────────────────────────────────────────────────────
# SECTION 2: PCIe Bandwidth Measurement
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: PCIe Bandwidth Measurement ──")
print("""
  The PCIe bus connects CPU RAM to GPU VRAM.  Every DataLoader batch
  crosses this bus when you call tensor.to(device) or tensor.cuda().

  PCIe THEORETICAL BANDWIDTH:
    PCIe 3.0 x16: 16 GB/s  (older systems, GTX/RTX 20xx era)
    PCIe 4.0 x16: 32 GB/s  (current mainstream, RTX 30xx/A100)
    PCIe 5.0 x16: 64 GB/s  (enterprise, H100 NVL, 2023+)

  PRACTICAL BANDWIDTH:
    Practical rates are lower than theoretical due to protocol overhead
    (8b/10b encoding, TLPs, credit management):
    PCIe 3.0 x16 practical: ~12–14 GB/s
    PCIe 4.0 x16 practical: ~18–26 GB/s

  PINNED vs PAGEABLE MEMORY:
    Pinned (page-locked) memory achieves the practical maximum because
    the GPU DMA engine can read it directly without a staging copy.
    Pageable memory requires the OS to first copy to a pinned buffer
    before DMA can proceed — this roughly halves the effective rate.

  DETECTING A PCIe BOTTLENECK:
    Symptom: GPU sm% stays low even with many DataLoader workers.
    Diagnosis: nsys trace shows a large HtoD (host-to-device) memcpy
    bar between each batch of kernels.
    Fix: pin_memory=True in DataLoader, non_blocking=True in .to(device),
    pre-load batches asynchronously with a background thread.
""")

def measure_h2d_bandwidth(size_mb: float, pinned: bool) -> float:
    """
    TODO 2: Implement this function.
    If CUDA is available:
      Allocate a float32 CPU tensor of size_mb megabytes.
      If pinned=True, call .pin_memory() on it.
      Time tensor.to('cuda') for 10 runs (after 3 warmup runs).
        For pinned: use tensor.to('cuda', non_blocking=True) and
        then torch.cuda.synchronize() to get accurate timing.
      Return bandwidth in GB/s = size_mb * 1e6 / best_elapsed / 1e9.

    If CUDA is not available:
      Return -1.0 (caller will skip the assertion).
    """
    pass  # YOUR CODE HERE → return bandwidth_gbs (or -1.0 if no CUDA)


if DEVICE == "cuda":
    pageable_bw = measure_h2d_bandwidth(256.0, pinned=False)
    pinned_bw   = measure_h2d_bandwidth(256.0, pinned=True)

    assert pageable_bw is not None and pageable_bw > 0, (
        "measure_h2d_bandwidth must return > 0 when CUDA available. "
        "Did you implement TODO 2?"
    )
    assert pinned_bw is not None and pinned_bw > 0, \
        "Pinned H2D bandwidth must be > 0"

    print(f"  Pageable H2D (256 MB): {pageable_bw:.1f} GB/s")
    print(f"  Pinned   H2D (256 MB): {pinned_bw:.1f} GB/s")
    print(f"  Pinned speedup:        {pinned_bw/pageable_bw:.2f}×")
else:
    print("  CUDA not available — skipping PCIe measurement.")
    print("  On a system with a GPU, this section measures pinned vs")
    print("  pageable host-to-device transfer bandwidth.")
    # Create placeholders for the later assert
    pageable_bw = -1.0
    pinned_bw   = -1.0

assert pinned_bw >= -1.0, "measure_h2d_bandwidth must return a float"
print("  ✓ Section 2 passed — PCIe bandwidth measurement complete")


# ─────────────────────────────────────────────────────────────
# SECTION 3: The Full Bandwidth Hierarchy
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: The Full Bandwidth Hierarchy ──")
print("""
  The complete memory bandwidth hierarchy for a modern AI server,
  from fastest to slowest:

  ┌────────────────────────────────┬──────────────┬──────────────────────────────┐
  │ Level                          │ Bandwidth    │ Notes                        │
  ├────────────────────────────────┼──────────────┼──────────────────────────────┤
  │ GPU registers                  │ >> 10 TB/s   │ Unlimited — per-SM local     │
  │ GPU L1 / shared memory         │ ~20 TB/s     │ Per-SM, 100–128 KB           │
  │ GPU L2 cache                   │ ~5–10 TB/s   │ A100: 40 MB shared           │
  │ GPU HBM (VRAM)                 │ ~1–3.5 TB/s  │ A100: 2 TB/s; H100: 3.35 TB/s│
  │ NVLink (GPU-to-GPU)            │ ~300–900 GB/s│ DGX A100: 600 GB/s bidirec.  │
  │ PCIe 4.0 x16 (CPU↔GPU)        │ ~20–26 GB/s  │ DataLoader H2D bottleneck    │
  │ CPU DRAM (DDR5 dual-ch.)       │ ~50–100 GB/s │ DataLoader preprocessing cap │
  │ CPU L3 cache                   │ ~100–500 GB/s│ Fits small working sets      │
  │ NVMe SSD (PCIe 4.0)            │ ~3–7 GB/s    │ Sequential reads             │
  │ SATA SSD                       │ ~0.5–0.6 GB/s│ Older systems                │
  │ HDD (7200 RPM)                 │ ~0.1–0.2 GB/s│ Avoid for DataLoader         │
  │ Network storage (NFS 10 GbE)   │ ~1.0–1.25 GB/s│ Subject to congestion       │
  │ S3 / object storage            │ ~0.1–2 GB/s  │ Per-file open overhead       │
  └────────────────────────────────┴──────────────┴──────────────────────────────┘

  WHICH BOTTLENECK DOES EACH AI STAGE HIT?

    DataLoader (SSD → DRAM):       Bottleneck is SSD read speed (3–7 GB/s).
                                    Fix: NVMe SSD, tmpfs RAM disk, or cache .pt files.

    DataLoader (DRAM → CPU):       Bottleneck is CPU DRAM bandwidth (~50 GB/s).
                                    Fix: reduce preprocessing bytes, use uint8 not float32.

    Batch transfer (CPU → GPU):    Bottleneck is PCIe bandwidth (~20 GB/s).
                                    Fix: pin_memory=True, non_blocking=True, smaller dtype.

    LLM decode inference:          Bottleneck is GPU HBM bandwidth (~2 TB/s).
                                    KV cache size × model layers must stream each step.
                                    Fix: quantise KV cache, reduce sequence length.

    Training matmul (large batch): Bottleneck is Tensor Core compute.
                                    The matmul is compute-bound above the ridge point.
                                    Fix: FP16/BF16, torch.compile, increase batch size.
""")

def identify_bottleneck(stage: str) -> str:
    """
    TODO 3: Implement this function.
    Return the bandwidth bottleneck string for each AI workload stage:
      "dataloader_ssd"          → "SSD (3–7 GB/s)"
      "dataloader_dram"         → "CPU DRAM (~50 GB/s)"
      "pcie_transfer"           → "PCIe (~20 GB/s)"
      "llm_decode"              → "GPU HBM (~2 TB/s)"
      "training_matmul_large_batch" → "Compute (Tensor Cores)"
    For unknown stages, return "unknown".
    """
    pass  # YOUR CODE HERE → return bottleneck string


assert identify_bottleneck("dataloader_ssd") is not None, \
    "identify_bottleneck must return a string. Did you implement TODO 3?"
assert "SSD" in identify_bottleneck("dataloader_ssd"), (
    f"identify_bottleneck('dataloader_ssd') must mention 'SSD', "
    f"got '{identify_bottleneck('dataloader_ssd')}'"
)
assert "HBM" in identify_bottleneck("llm_decode"), (
    f"identify_bottleneck('llm_decode') must mention 'HBM', "
    f"got '{identify_bottleneck('llm_decode')}'"
)
assert "PCIe" in identify_bottleneck("pcie_transfer"), \
    f"identify_bottleneck('pcie_transfer') must mention 'PCIe'"
assert "DRAM" in identify_bottleneck("dataloader_dram") or \
       "dram" in identify_bottleneck("dataloader_dram").lower(), \
    f"identify_bottleneck('dataloader_dram') must mention 'DRAM'"

stages = [
    "dataloader_ssd",
    "dataloader_dram",
    "pcie_transfer",
    "llm_decode",
    "training_matmul_large_batch",
]
print("  Bottleneck map:")
for stage in stages:
    print(f"    {stage:<30}: {identify_bottleneck(stage)}")
print("  ✓ Section 3 passed — bottleneck identification correct")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Applying the Roofline with Measured Bandwidth
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Applying the Roofline with Measured Bandwidth ──")
print("""
  Chapter 5 introduced the roofline model with THEORETICAL peak values
  from GPU spec sheets.  Section 9.3 adds an important refinement:
  use your MEASURED bandwidth instead of the theoretical maximum.

  WHY MEASURED BANDWIDTH MATTERS:
    The A100 spec says 2,000 GB/s HBM bandwidth.  Under real workloads,
    achieved bandwidth is typically 1,400–1,800 GB/s (70–90% efficiency).
    Using 2,000 GB/s in your roofline gives an optimistic ridge point.
    Using your measured value (say 1,600 GB/s) gives a more realistic
    ridge point — one that better predicts when a real kernel will be
    memory-bound vs compute-bound.

  THE RIDGE POINT WITH MEASURED BANDWIDTH:
    ridge_point = peak_tflops * 1e12 / (measured_bandwidth_gbs * 1e9)
                = FLOPs/byte

  INTERPRETATION:
    A kernel with arithmetic intensity BELOW the ridge point is
    memory-bound — it cannot use all available compute because it
    runs out of memory bandwidth first.
    A kernel ABOVE the ridge point is compute-bound.

  LOWER BANDWIDTH → HIGHER RIDGE POINT:
    If your HBM bandwidth degrades (due to bank conflicts, strided
    access, or ECC overhead), the ridge point rises.  Kernels that
    were previously compute-bound become memory-bound.
    This is why ECC-on vs ECC-off can shift kernel classification.

  EXAMPLE:
    A100 spec:      312 TFLOP/s FP16, 2000 GB/s HBM  → ridge = 156 FLOPs/byte
    A100 measured:  312 TFLOP/s FP16, 1600 GB/s HBM  → ridge = 195 FLOPs/byte
    Kernel AI = 160 FLOPs/byte:
      With spec ridge   (156): compute-bound
      With measured ridge (195): memory-bound
    The measured ridge gives the honest answer.
""")

def compute_ridge_point(peak_tflops: float, bandwidth_gbs: float) -> float:
    """
    TODO 4: Implement this function.
    Compute the roofline ridge point in FLOPs/byte:
        ridge_point = peak_tflops * 1e12 / (bandwidth_gbs * 1e9)

    Both arguments are already in the units named:
        peak_tflops    — TeraFLOPs per second (e.g., 312 for A100 FP16)
        bandwidth_gbs  — GB/s (e.g., 2000 for A100 HBM spec)

    Return the ridge point as a float (FLOPs per byte).
    """
    pass  # YOUR CODE HERE → return ridge_flops_per_byte


ridge_spec     = compute_ridge_point(312, 2000)
ridge_measured = compute_ridge_point(312, 1600)

assert ridge_spec is not None, "compute_ridge_point must return a float. Did you implement TODO 4?"
assert ridge_spec == 156.0, (
    f"compute_ridge_point(312, 2000) = 312e12 / 2000e9 = 156.0, "
    f"got {ridge_spec}"
)
assert compute_ridge_point(312, 1600) > compute_ridge_point(312, 2000), (
    f"Lower bandwidth → higher ridge point. "
    f"compute_ridge_point(312, 1600) = {ridge_measured:.1f} should be > "
    f"compute_ridge_point(312, 2000) = {ridge_spec:.1f}"
)

print(f"  A100 with spec bandwidth (2000 GB/s):     ridge = {ridge_spec:.1f} FLOPs/byte")
print(f"  A100 with measured bw   (1600 GB/s):      ridge = {ridge_measured:.1f} FLOPs/byte")
print(f"  H100 with spec bandwidth (3350 GB/s):     ridge = {compute_ridge_point(989, 3350):.1f} FLOPs/byte")

# Demonstrate the classification shift
test_ai = 160.0
print(f"\n  Kernel with AI = {test_ai} FLOPs/byte:")
for name, peak, bw in [("A100 spec", 312, 2000), ("A100 measured", 312, 1600), ("H100 spec", 989, 3350)]:
    ridge = compute_ridge_point(peak, bw)
    classification = "memory-bound" if test_ai < ridge else "compute-bound"
    print(f"    {name:<20}: ridge={ridge:.0f}  → {classification}")

print("  ✓ Section 4 passed — ridge point calculation correct and inverse-BW relationship holds")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Part III Exercise 9.3 complete!")
print()
print("  You have completed all exercises in Part III —")
print("  Linux Systems Profiling.")
print()
print("  Part III covered:")
print("    Chapter 8: perf stat, CPU flamegraphs, eBPF/bpftrace, lock contention")
print("    Chapter 9: Memory hierarchy, NUMA topology, memory bandwidth & roofline")
print("=" * 60)
