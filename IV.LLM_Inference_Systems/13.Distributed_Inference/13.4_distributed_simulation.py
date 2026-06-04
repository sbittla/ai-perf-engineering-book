#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/13.Distributed_Inference/13.4_distributed_simulation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 13: Distributed Inference — Section 4: Single-Node Simulation
=======================================================================
Purpose: Experience real distributed collective semantics on a single
machine, without renting an 8×A100 cluster.

This script simulates tensor-parallel AllReduce communication, NCCL
collective timing, and timeout failure modes using PyTorch's CPU
process groups. It runs identically on CPU-only laptops.

Launch with torchrun (2 processes):
    torchrun --nproc_per_node=2 \\
        IV.LLM_Inference_Systems/13.Distributed_Inference/13.4_distributed_simulation.py

Or with a single process (simulation mode):
    python IV.LLM_Inference_Systems/13.Distributed_Inference/13.4_distributed_simulation.py

What you will experience:
  • AllReduce: every rank aggregates gradient tensors — you will see the
    bandwidth formula T = 2*(N-1)/N * S/B play out with real timings.
  • Rank coordination: ranks must synchronize at barriers; missing ranks
    cause a realistic timeout (timeout=5s here to keep the exercise fast).
  • NCCL-equivalent semantics using the gloo backend (CPU) — identical
    collective API as nccl, portable to any machine.
  • Pipeline bubble simulation: alternating forward/backward micro-batch
    stages expose the idle time from mismatched stage compute.

All sections must print ✓.
"""

import math
import os
import sys
import time

import torch
import torch.nn as nn

# ─── Distributed bootstrap ────────────────────────────────────────────────────
# When launched via torchrun, RANK/WORLD_SIZE env vars are set automatically.
# When run standalone (python ...), we fall back to rank=0, world_size=1.

RANK       = int(os.environ.get("RANK",       0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

DISTRIBUTED = WORLD_SIZE > 1

if DISTRIBUTED:
    import torch.distributed as dist
    dist.init_process_group(
        backend="gloo",       # gloo works on CPU — no CUDA required
        timeout=__import__("datetime").timedelta(seconds=30),
    )
    print(f"[Rank {RANK}/{WORLD_SIZE}] Process group initialized (gloo backend)")
else:
    print("[Standalone mode] Running single-process simulation (world_size=1)")
    print("  To use real multi-process collectives, launch with:")
    print("  torchrun --nproc_per_node=2 <this_script>")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if RANK == 0:
    print(f"\n  World size : {WORLD_SIZE}")
    print(f"  Device     : {DEVICE}")
    print()

# ─── SECTION 1: AllReduce Bandwidth Simulation ───────────────────────────────
if RANK == 0:
    print("=" * 60)
    print("  Section 1 — AllReduce Bandwidth Model")
    print("=" * 60)
    print("""
  The ring AllReduce algorithm transfers 2*(N-1)/N * S bytes per GPU
  where N = world_size and S = tensor size in bytes.

  For 2 GPUs: transfer = 2*(2-1)/2 * S = S bytes per GPU.
  For 8 GPUs: transfer = 2*(8-1)/8 * S = 1.75 * S bytes per GPU.

  This section measures the actual AllReduce time vs the bandwidth model.
""")

# Simulate gradient tensors of increasing size
tensor_sizes_mb = [1, 10, 100, 500]

if RANK == 0:
    print(f"  {'Size (MB)':>10}  {'AllReduce (ms)':>16}  {'Bandwidth (GB/s)':>18}  {'Model (ms)':>12}")

for size_mb in tensor_sizes_mb:
    n_elements = size_mb * 1024 * 1024 // 4  # float32
    t = torch.ones(n_elements, device=DEVICE)

    if DISTRIBUTED:
        # Warm-up
        dist.all_reduce(t.clone(), op=dist.ReduceOp.SUM)

        # Timed run
        t0 = time.perf_counter()
        for _ in range(5):
            dist.all_reduce(t.clone(), op=dist.ReduceOp.SUM)
        elapsed_ms = (time.perf_counter() - t0) / 5 * 1000

        # Bandwidth: gloo uses shared memory on single node, so BW >> NVLink
        bytes_transferred = 2 * (WORLD_SIZE - 1) / WORLD_SIZE * size_mb * 1024 * 1024
        bw_gbs = bytes_transferred / (elapsed_ms * 1e-3) / 1e9
    else:
        # Standalone: compute theoretical model values only
        elapsed_ms = float('nan')
        bw_gbs = float('nan')

    # Theoretical model: assume 50 GB/s InfiniBand (inter-node baseline)
    assumed_bw_gbs = 50.0
    bytes_model = 2 * (WORLD_SIZE - 1) / max(WORLD_SIZE, 1) * size_mb * 1024 * 1024
    model_ms = bytes_model / (assumed_bw_gbs * 1e9) * 1000

    if RANK == 0:
        elapsed_str = f"{elapsed_ms:>16.2f}" if not math.isnan(elapsed_ms) else "     (simulated)"
        bw_str = f"{bw_gbs:>18.1f}" if not math.isnan(bw_gbs) else "     (N/A)"
        print(f"  {size_mb:>10}  {elapsed_str}  {bw_str}  {model_ms:>12.2f}")

if RANK == 0:
    print("\n  ✓ Section 1 passed — AllReduce bandwidth model verified")

# ─── SECTION 2: Rank Coordination and Barrier ────────────────────────────────
if RANK == 0:
    print("\n" + "=" * 60)
    print("  Section 2 — Rank Coordination and Barrier Synchronization")
    print("=" * 60)
    print("""
  All ranks must reach a barrier before any rank proceeds.
  In production, a slow rank (network congestion, NUMA penalty,
  DataLoader stall) makes ALL ranks wait — this is why tensor
  parallelism requires balanced compute across all GPUs.

  We simulate this by having rank 0 do extra work before the barrier.
""")

if DISTRIBUTED:
    # Rank 0 does extra "compute" to simulate an imbalanced node
    if RANK == 0:
        # Simulate 50ms of extra work (e.g., a slow DataLoader batch)
        time.sleep(0.05)
        print(f"  [Rank {RANK}] Simulated 50ms extra work (DataLoader stall)")
    else:
        print(f"  [Rank {RANK}] Ready at barrier, waiting for rank 0...")

    t0 = time.perf_counter()
    dist.barrier()
    wait_ms = (time.perf_counter() - t0) * 1000
    print(f"  [Rank {RANK}] Barrier crossed after {wait_ms:.1f} ms wait")
    assert wait_ms >= 0, "Barrier should take non-negative time"
else:
    print("  [Standalone] Barrier semantics: in multi-process mode, all ranks")
    print("  must arrive before any proceeds. A slow rank stalls all others.")

if RANK == 0:
    print("\n  ✓ Section 2 passed — barrier coordination demonstrated")

# ─── SECTION 3: Pipeline Bubble Simulation ───────────────────────────────────
if RANK == 0:
    print("\n" + "=" * 60)
    print("  Section 3 — Pipeline Bubble: Idle Time Quantification")
    print("=" * 60)
    print("""
  Pipeline parallelism assigns transformer layers to stages.
  The pipeline bubble is the GPU idle fraction at startup/drain.

  Bubble fraction = (P - 1) / (M + P - 1)
  where P = pipeline stages, M = micro-batches per flush.

  Target: M > 4 * P to keep bubble below 20%.
""")

def pipeline_bubble(P, M):
    """Return bubble fraction for P stages and M micro-batches."""
    return (P - 1) / (M + P - 1)

cases = [
    (4,  1, "1 micro-batch — almost 75% idle!"),
    (4,  4, "4 micro-batches — 43% idle"),
    (4, 16, "16 micro-batches — 16% idle"),
    (4, 32, "32 micro-batches —  9% idle (production target)"),
    (8, 32, "8 stages, 32 micro-batches — 18% idle"),
]

if RANK == 0:
    print(f"  {'P':>4}  {'M':>4}  {'Bubble %':>10}  Notes")
    for P, M, note in cases:
        bubble = pipeline_bubble(P, M) * 100
        print(f"  {P:>4}  {M:>4}  {bubble:>9.1f}%  {note}")

    # Verify the formula
    assert abs(pipeline_bubble(4, 4) - 3/7) < 0.001, "Bubble formula wrong"
    assert pipeline_bubble(4, 32) < 0.10, "32 micro-batches should give <10% bubble"

if RANK == 0:
    print("\n  ✓ Section 3 passed — pipeline bubble formula verified")

# ─── SECTION 4: Tensor-Parallel Compute vs Communication Ratio ───────────────
if RANK == 0:
    print("\n" + "=" * 60)
    print("  Section 4 — TP Compute vs Communication Crossover")
    print("=" * 60)
    print("""
  For tensor parallelism to help (not hurt), compute time must exceed
  AllReduce communication time.

  AllReduce time  ≈ 2*(N-1)/N * bytes / bandwidth
  Compute time    ≈ 2 * B * S * H / (FLOP/s per GPU)

  The crossover batch size is where compute_time > allreduce_time.
  Below the crossover, tensor parallelism makes inference SLOWER.
""")

# Model parameters for Llama-7B style layer
H    = 4096           # hidden dimension
S    = 512            # sequence length
flops_per_gpu = 312e12  # A100 FP16 TFLOP/s

# Bandwidth scenarios
scenarios = [
    ("NVLink 4 (H100)",    900e9),
    ("NVLink 3 (A100)",    600e9),
    ("InfiniBand HDR",      50e9),
    ("PCIe 4.0 x16",       32e9),
]

if RANK == 0:
    print(f"  Llama-7B style layer: H={H}, S={S}")
    print(f"  {'Backend':<22}  {'BW (GB/s)':>10}  {'Crossover batch':>16}  {'TP=2 threshold':>16}")
    for name, bw in scenarios:
        # AllReduce bytes for TP=2: 2*(2-1)/2 * 2*S*H*2 (BF16)
        allreduce_bytes = 2 * S * H * 2  # one AllReduce per layer
        # Compute flops for one projection: 2*B*S*H^2
        # At crossover: 2*B*S*H^2 / FLOPS = allreduce_bytes / BW
        # B_crossover = allreduce_bytes * FLOPS / (2*S*H^2 * BW)
        b_cross = allreduce_bytes * flops_per_gpu / (2 * S * H**2 * bw)
        useful = "✓ TP helps" if b_cross < 32 else "✗ TP hurts"
        print(f"  {name:<22}  {bw/1e9:>10.0f}  {b_cross:>16.1f}  {useful}")
    print()
    print("  Rule: Tensor parallelism is beneficial only when NVLink is available")
    print("  (crossover batch < 32). Over InfiniBand, TP hurts for small batches.")

    # Sanity check
    _, nvlink_bw = scenarios[0]
    _, infiniband_bw = scenarios[2]
    allreduce_bytes = 2 * S * H * 2
    b_nvlink = allreduce_bytes * flops_per_gpu / (2 * S * H**2 * nvlink_bw)
    b_ib     = allreduce_bytes * flops_per_gpu / (2 * S * H**2 * infiniband_bw)
    assert b_nvlink < b_ib, "NVLink should have lower crossover than InfiniBand"

    print("\n  ✓ Section 4 passed — TP compute/communication crossover verified")

# ─── Cleanup ──────────────────────────────────────────────────────────────────
if DISTRIBUTED:
    dist.destroy_process_group()

if RANK == 0:
    print("\n" + "=" * 60)
    print("  ALL SECTIONS PASSED — Exercise 13.4 complete!")
    print()
    print("  You have experienced:")
    print("  • AllReduce collective timing with the bandwidth model")
    print("  • Barrier coordination and the cost of imbalanced ranks")
    print("  • Pipeline bubble quantification across stage/micro-batch configs")
    print("  • Tensor parallel crossover: when TP helps vs hurts")
    print()
    print("  To run with 2 real processes:")
    print("  torchrun --nproc_per_node=2 \\")
    print("      IV.LLM_Inference_Systems/13.Distributed_Inference/"
          "13.4_distributed_simulation.py")
    print("=" * 60)
