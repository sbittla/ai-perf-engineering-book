#!/usr/bin/env python3
"""
memory_bench.py  —  Phase 3, Module 7: Memory Bandwidth & NUMA Benchmarking
============================================================================

HOW TO RUN:
    python memory_bench.py
    python memory_bench.py --skip-numa   (on single-socket systems)

    # After running, diagnose live with:
    sudo perf stat -e cache-misses,cache-references python memory_bench.py
    numactl --hardware   (to see your NUMA topology)


"""

import argparse
import time
import torch
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--skip-numa", action="store_true",
                    help="Skip NUMA experiments (single-socket systems)")
parser.add_argument("--gpu-only",  action="store_true",
                    help="Skip CPU benchmarks, GPU only")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


def sep(title):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def wall_bw(fn, size_bytes, warmup=3, iters=10):
    """Measure bandwidth using wall-clock time (for CPU operations)."""
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = (time.perf_counter() - start) / iters
    return size_bytes / elapsed / 1e9   # GB/s


def cuda_bw(fn, size_bytes, warmup=5, iters=30):
    """Measure bandwidth using CUDA events (for GPU operations)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    return size_bytes / (ms / 1000) / 1e9   # GB/s


# =============================================================================
# SECTION 1 — CPU Memory Bandwidth vs Working Set Size
# =============================================================================
if not args.gpu_only:
    sep("SECTION 1: CPU Memory Bandwidth (Cache vs DRAM)")
    print("""
  We copy tensors of increasing size on the CPU.
  When working set < L3 cache → data served from cache → high bandwidth.
  When working set > L3 cache → data served from DRAM → bandwidth drops.
  The "knee" in the bandwidth curve = effective L3 cache size.
  """)

    print(f"  {'Size':>10}  {'Bandwidth (GB/s)':>18}  {'Tier':>14}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*14}")

    # Try to get L3 cache size
    try:
        import subprocess
        result = subprocess.run(["lscpu"], capture_output=True, text=True)
        l3_line = [l for l in result.stdout.split('\n') if 'L3' in l]
        l3_info = l3_line[0] if l3_line else "unknown"
        print(f"  CPU L3 cache: {l3_info.strip()}\n")
    except Exception:
        pass

    prev_bw = None
    for size_mb in [0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        n = int(size_mb * 1e6 / 4)
        t = torch.randn(n, dtype=torch.float32)   # CPU tensor
        bw = wall_bw(lambda: t.clone(), n * 4)
        actual_mb = n * 4 / 1e6

        # Detect the cache boundary (bandwidth drops significantly)
        if prev_bw and bw < prev_bw * 0.7:
            tier = "← DRAM boundary"
        elif actual_mb < 1:
            tier = "L1/L2"
        elif actual_mb < 8:
            tier = "L3"
        else:
            tier = "DRAM"
        
        print(f"  {actual_mb:>8.1f}MB  {bw:>18.1f}  {tier:>14}")
        prev_bw = bw

    print(f"\n  → Sharp bandwidth drop = L3 cache capacity exceeded")
    print(f"  → To see cache miss rate: sudo perf stat -e cache-misses python memory_bench.py")


# =============================================================================
# SECTION 2 — PCIe Host-to-Device Transfer (CPU RAM → GPU VRAM)
# =============================================================================
if device == "cuda":
    sep("SECTION 2: PCIe Host-to-Device Transfer Bandwidth")
    print("""
  DataLoader workers load data into CPU RAM.
  Before the model can use it, it must cross the PCIe bus to GPU VRAM.
  This is the bottleneck when num_workers is high but GPU is still idle.

  PINNED (page-locked) memory:
    Allocated with torch.cuda.pin_memory() or pin_memory=True in DataLoader.
    GPU DMA engine can read it directly — no OS copy needed.
    Enables ASYNCHRONOUS transfers: CPU doesn't wait.

  PAGEABLE memory:
    OS may swap these pages to disk.
    GPU must copy to a temporary pinned buffer first.
    Transfer is SLOWER and SYNCHRONOUS.
  """)

    print(f"  {'Size':>10}  {'Pageable (GB/s)':>18}  {'Pinned (GB/s)':>16}  {'Speedup':>10}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*16}  {'─'*10}")

    for size_mb in [1, 4, 16, 64, 256]:
        n = int(size_mb * 1e6 / 4)

        # Pageable CPU tensor → GPU
        cpu_pageable = torch.randn(n)   # Normal CPU allocation
        def h2d_pageable():
            return cpu_pageable.to(device)
        bw_pageable = cuda_bw(h2d_pageable, n * 4)

        # Pinned CPU tensor → GPU (faster path)
        # pin_memory() marks the allocation as page-locked
        cpu_pinned = torch.randn(n).pin_memory()
        def h2d_pinned():
            return cpu_pinned.to(device, non_blocking=True)
        bw_pinned = cuda_bw(h2d_pinned, n * 4)

        speedup = bw_pinned / bw_pageable
        actual_mb = n * 4 / 1e6

        print(f"  {actual_mb:>8.1f}MB  {bw_pageable:>18.1f}  {bw_pinned:>16.1f}  {speedup:>9.2f}x")
        del cpu_pageable, cpu_pinned

    print(f"\n  → Always use pin_memory=True in DataLoader for {speedup:.1f}x+ H2D speedup")
    print(f"  → PCIe 4.0 x16 theoretical: ~32 GB/s; practical: ~20-26 GB/s")


# =============================================================================
# SECTION 3 — GPU DRAM Bandwidth (device-to-device)
# =============================================================================
if device == "cuda":
    sep("SECTION 3: GPU DRAM Bandwidth (Device-to-Device)")
    print("""
  This measures how fast the GPU can read/write its own VRAM.
  This is the ceiling for MEMORY-BOUND kernels like LLM decode.

  LLM decode throughput = (KV cache bytes per token) / (GPU BW)
  Example: Llama-7B, 32 layers, FP16, batch=1:
    KV per token = 32 layers × 2 (K,V) × 32 heads × 128 dim × 2 bytes = 524KB
    At 272 GB/s: max ~500K tokens/sec... but overhead reduces this to ~100 tok/s
  """)

    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}")
    print(f"  Theoretical BW: {getattr(props, 'memory_bandwidth_gb_s', 'unknown')} GB/s\n")

    print(f"  {'Size':>10}  {'d2d copy (GB/s)':>18}  {'in-place op (GB/s)':>20}")
    print(f"  {'─'*10}  {'─'*18}  {'─'*20}")

    for size_mb in [1, 4, 16, 64, 256, 512]:
        try:
            n = int(size_mb * 1e6 / 4)
            src = torch.randn(n, device=device, dtype=torch.float32)
            dst = torch.empty_like(src)

            bw_copy = cuda_bw(lambda: dst.copy_(src), n * 4 * 2)   # read + write
            bw_op   = cuda_bw(lambda: src.mul_(2.0), n * 4 * 2)    # read + write

            actual_mb = n * 4 / 1e6
            print(f"  {actual_mb:>8.1f}MB  {bw_copy:>18.1f}  {bw_op:>20.1f}")
            del src, dst
        except RuntimeError:
            print(f"  {size_mb:>8.0f}MB  (OOM)")

    print(f"\n  → Peak GPU BW is the hard ceiling for memory-bound kernels (LLM decode)")
    print(f"  → Increasing batch size doesn't help if DRAM BW is the bottleneck")


# =============================================================================
# SECTION 4 — NUMA Topology and Cross-Socket Penalty
# =============================================================================
if not args.skip_numa and not args.gpu_only:
    sep("SECTION 4: NUMA Topology")
    print("""
  NUMA (Non-Uniform Memory Access): on multi-socket systems, each CPU has
  its own memory bank. Accessing the OTHER socket's memory takes 2x+ longer.

  For AI workloads:
    - GPU 0 is physically connected to CPU socket 0's PCIe lanes
    - If DataLoader workers run on socket 1, data fetches go:
        CPU1 RAM → CPU1↔CPU0 interconnect → CPU0 RAM → PCIe → GPU VRAM
      Instead of:
        CPU0 RAM → PCIe → GPU VRAM (direct)
    - NUMA-aware binding: numactl --cpunodebind=0 --membind=0 python train.py
  """)

    try:
        import subprocess
        result = subprocess.run(["numactl", "--hardware"],
                                capture_output=True, text=True, timeout=5)
        print("  numactl --hardware output:")
        print("  " + "\n  ".join(result.stdout.strip().split('\n')[:15]))

        # Test cross-socket access (only meaningful on multi-socket systems)
        result2 = subprocess.run(["numastat", "-p", str(os.getpid())],
                                 capture_output=True, text=True, timeout=5)
        if "numa_miss" in result2.stdout:
            print("\n  numastat for current process:")
            print("  " + "\n  ".join(result2.stdout.strip().split('\n')[:10]))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  numactl not found. Install: sudo apt install numactl")
        print("  On single-socket systems, NUMA is less relevant.")

    print("""
  COMMANDS TO USE IN PRACTICE:
    # Show full topology (CPU + cache + NUMA + GPU):
    lstopo --of ascii

    # Bind process to NUMA node 0:
    numactl --cpunodebind=0 --membind=0 python train.py

    # Show NUMA memory stats for a running process:
    numastat -p $(pgrep python)

    # Show which NUMA node each GPU connects to:
    cat /sys/bus/pci/devices/*/numa_node 2>/dev/null | head -5
  """)


# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'='*60}")
print(f"  MEMORY BANDWIDTH SUMMARY")
print(f"{'='*60}")
print(f"""
  Hierarchy from fastest to slowest:
  
  GPU L1/shared : ~19 TB/s (RTX 4060 per-SM)
  GPU L2 cache  : ~2 TB/s  (RTX 4060 total)
  GPU DRAM      : ~272 GB/s (RTX 4060 GDDR6)
  PCIe 4.0 x16  : ~32 GB/s (theoretical)  ← DataLoader bottleneck
  CPU DRAM      : ~50 GB/s (DDR5 dual-channel)
  CPU L3 cache  : ~200-500 GB/s (varies)
  SSD (NVMe)    : ~3-7 GB/s (PCIe 4.0 NVMe)
  HDD           : ~0.1-0.2 GB/s (rotational)

  KEY INSIGHT:
    LLM decode throughput is bounded by GPU DRAM bandwidth.
    DataLoader throughput is bounded by PCIe bandwidth (if disk → GPU).
    Always identify WHICH bottleneck before optimising.
""")
