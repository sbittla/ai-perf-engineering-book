#!/usr/bin/env python3
"""
nccl_bench.py  —  Phase 4/5: NCCL Communication Bandwidth Benchmark
======================================================================

HOW TO RUN:
    # Requires at least 2 GPUs or simulates with 1:
    torchrun --nproc_per_node=2 nccl_bench.py

    # Single GPU mode (shows bandwidth ceiling analysis only):
    python nccl_bench.py --single

    # With NCCL debug output:
    NCCL_DEBUG=INFO torchrun --nproc_per_node=2 nccl_bench.py

    # Profile NCCL calls with nsys:
    nsys profile --trace=cuda,nvtx,nccl \\
        torchrun --nproc_per_node=2 nccl_bench.py


"""

import argparse
import os
import time
import torch
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument("--single",  action="store_true", help="Run single-GPU analysis only")
parser.add_argument("--warmup",  type=int, default=5)
parser.add_argument("--iters",   type=int, default=20)
args = parser.parse_args()

IS_DISTRIBUTED = "RANK" in os.environ and not args.single
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cuda_time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    e = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    if s:
        s.record()
    else:
        t0 = time.perf_counter()

    for _ in range(iters):
        fn()

    if IS_DISTRIBUTED:
        dist.barrier()   # sync before stopping clock

    if e:
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    else:
        return (time.perf_counter() - t0) / iters * 1000


# =============================================================================
# MAIN
# =============================================================================
if IS_DISTRIBUTED:
    dist.init_process_group(backend="nccl")
    rank  = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
else:
    rank, world = 0, 1

def is_main():
    return rank == 0

if is_main():
    print(f"\n{'='*65}")
    print(f"  NCCL Collective Benchmark")
    print(f"{'='*65}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU: {props.name}")
        print(f"  GPUs in this run: {world}")

    # Check GPU topology
    if torch.cuda.device_count() >= 2:
        try:
            import subprocess
            topo = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=5
            )
            print(f"\n  GPU Interconnect Topology:")
            print("  " + "\n  ".join(topo.stdout.strip().split('\n')[:8]))
        except Exception:
            pass
    print()


# ── Section 1: Single-GPU memory bandwidth (baseline for comparison) ──────────
if is_main():
    print(f"── GPU Memory Bandwidth (single-GPU baseline) ──────────────────")
    n = 128 * 1024 * 1024 // 4   # 128MB
    t = torch.randn(n, device=device, dtype=torch.float32)
    ms_bw = cuda_time_ms(lambda: t.clone(), warmup=10, iters=30)
    peak_bw = n * 4 * 2 / (ms_bw / 1000) / 1e9   # GB/s (read + write)
    print(f"  GPU DRAM bandwidth: {peak_bw:.1f} GB/s")
    print(f"  (This is the theoretical ceiling for any GPU collective)\n")
    del t

# ── Section 2: All-reduce bandwidth at different tensor sizes ─────────────────
if is_main():
    print(f"── All-Reduce Bandwidth (world_size={world}) {'─'*30}")
    print(f"  Measures: sum gradients across all GPUs and distribute back")
    print(f"  {'Size (MB)':>10}  {'Time (ms)':>10}  {'Bus BW (GB/s)':>14}  {'vs GPU DRAM':>12}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*12}")

for size_mb in [0.1, 1, 4, 16, 64, 256]:
    n = int(size_mb * 1e6 / 4)
    t = torch.randn(n, device=device, dtype=torch.float32)

    if IS_DISTRIBUTED:
        def do_allreduce():
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    else:
        # Single GPU: simulate with a copy (no actual collective)
        def do_allreduce():
            _ = t.clone()

    ms = cuda_time_ms(do_allreduce, args.warmup, args.iters)

    # Bus bandwidth for all-reduce: 2 × (N-1)/N × message_size / time
    # This accounts for the bidirectional ring algorithm used by NCCL
    bus_bytes = 2 * (world - 1) / world * (n * 4) if world > 1 else n * 4 * 2
    bus_bw = bus_bytes / (ms / 1000) / 1e9

    vs_dram = f"{bus_bw/peak_bw*100:.0f}%" if is_main() else ""

    if is_main():
        flag = ""
        if ms > 10 and size_mb >= 16:
            flag = " ← communication bottleneck"
        print(f"  {size_mb:>10.0f}  {ms:>10.3f}  {bus_bw:>14.1f}  {vs_dram:>12}{flag}")

    del t

# ── Section 3: Compare collectives at 64MB ────────────────────────────────────
if IS_DISTRIBUTED and is_main():
    print(f"\n── Collective Comparison (64MB tensor, world={world}) ──────────────")
    print(f"  {'Operation':>18}  {'Time (ms)':>10}  {'BW (GB/s)':>12}  Notes")
    print(f"  {'─'*18}  {'─'*10}  {'─'*12}  {'─'*30}")

    n = 64 * 1024 * 1024 // 4   # 64MB
    t = torch.randn(n, device=device)

    ops = {
        "all_reduce":    lambda: dist.all_reduce(t, op=dist.ReduceOp.SUM),
        "broadcast":     lambda: dist.broadcast(t, src=0),
        "reduce":        lambda: dist.reduce(t, dst=0, op=dist.ReduceOp.SUM),
    }

    for name, fn in ops.items():
        ms = cuda_time_ms(fn, args.warmup, args.iters)
        bw = n * 4 / (ms / 1000) / 1e9
        note = {
            "all_reduce": "DDP gradient sync (every step)",
            "broadcast":  "Parameter init / sync",
            "reduce":     "Reduce to rank 0 only",
        }.get(name, "")
        print(f"  {name:>18}  {ms:>10.3f}  {bw:>12.1f}  {note}")

    del t

# ── Section 4: Overlap analysis ───────────────────────────────────────────────
if is_main():
    print(f"""
── Communication-Compute Overlap ────────────────────────────────────
  In DDP training, NCCL all-reduce runs CONCURRENTLY with the backward
  pass for the previous layer (gradient overlap).

  torch.cuda.Stream enables this:
    compute_stream = torch.cuda.Stream()
    comm_stream    = torch.cuda.Stream()
    with torch.cuda.stream(compute_stream): backward_layer_N()
    with torch.cuda.stream(comm_stream):    allreduce_layer_N_minus_1()
    # Both streams run in parallel on the GPU

  PyTorch DDP does this automatically via bucket gradients.
  To see overlap in nsys timeline:
    - Open nsys-ui
    - Look for compute (green) and NCCL (orange) rows running simultaneously

  KEY COMMANDS:
    # Show GPU interconnect type (NVLink = fast, PIX/PHB = PCIe = slow):
    nvidia-smi topo -m

    # Check NVLink bandwidth:
    nvidia-smi nvlink --status

    # Run nccl-tests (must be compiled separately):
    nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g {world}
    """)

if IS_DISTRIBUTED:
    dist.destroy_process_group()
