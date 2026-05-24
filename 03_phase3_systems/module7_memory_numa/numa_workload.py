#!/usr/bin/env python3
"""
numa_workload.py  ─  Phase 3 / Module 7: NUMA-Aware vs Naive Allocation
========================================================================

HOW TO RUN
    python numa_workload.py
    numactl --cpunodebind=0 --membind=0 python numa_workload.py --label node0
    numactl --cpunodebind=1 --membind=1 python numa_workload.py --label node1

DIAGNOSE WITH
    numactl --hardware          # show topology
    numastat -p $$              # NUMA hits/misses for this process
    lstopo --of ascii           # full topology including GPU


"""

import argparse, os, subprocess, time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="default")
parser.add_argument("--size-mb", type=float, default=256.0,
                    help="Tensor size in MB for bandwidth tests")
args = parser.parse_args()

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 – Show current topology
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 0 – NUMA Topology")

pid = os.getpid()
print(f"  PID: {pid}   Label: {args.label}")

# Current CPU affinity
try:
    affinity = os.sched_getaffinity(0)
    print(f"  CPU affinity: cores {sorted(affinity)}")
except AttributeError:
    print("  CPU affinity: (not available on this OS)")

# numactl --hardware
try:
    r = subprocess.run(["numactl", "--hardware"],
                       capture_output=True, text=True, timeout=3)
    for line in r.stdout.strip().split('\n')[:12]:
        print(f"  {line}")
except FileNotFoundError:
    print("  numactl not found. Install: sudo apt install numactl")
except Exception as e:
    print(f"  numactl error: {e}")

# GPU NUMA node
for dev_path in ["/sys/bus/pci/devices/0000:01:00.0/numa_node",
                 "/sys/bus/pci/devices/0000:00:02.0/numa_node"]:
    try:
        with open(dev_path) as f:
            numa_node = f.read().strip()
        print(f"  GPU NUMA node: {numa_node}  ({dev_path})")
        break
    except FileNotFoundError:
        pass

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Memory allocation locality
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 1 – Memory Bandwidth (current NUMA binding)")
print(f"""
  Allocating {args.size_mb:.0f} MB tensor and measuring sequential read bandwidth.
  
  Run this script with different numactl settings to see the difference:
    python numa_workload.py --label default
    numactl --cpunodebind=0 --membind=0 python numa_workload.py --label node0
    numactl --cpunodebind=0 --membind=1 python numa_workload.py --label remote

  EXPECTED: --membind=1 (remote memory) will be 1.5–2.5× slower on
  multi-socket servers.  On single-socket: all nodes are local → same speed.
""")

n = int(args.size_mb * 1e6 / 4)   # number of float32 elements
arr = torch.rand(n, dtype=torch.float32)   # allocate on current NUMA node

# Bandwidth test: sequential read
ITERS = 10
times = []
for _ in range(ITERS + 3):
    t0 = time.perf_counter()
    _ = arr.sum()
    times.append(time.perf_counter() - t0)
times = times[3:]   # discard first 3 (cache warmup)

avg_ms = sum(times) / len(times) * 1000
bw_gbs = (n * 4) / (avg_ms / 1000) / 1e9
print(f"  [{args.label}]  {args.size_mb:.0f}MB  bandwidth: {bw_gbs:.1f} GB/s  ({avg_ms:.2f}ms)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – H2D transfer speed (CPU NUMA node → GPU)
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 2 – CPU→GPU Transfer Speed (NUMA effect on PCIe)")
print(f"""
  The GPU PCIe root complex is attached to ONE CPU socket.
  If CPU memory is on the SAME socket → data flows directly to PCIe.
  If CPU memory is on a DIFFERENT socket → data must cross QPI first.

  Measurable difference: typically 10–30% slower H2D on remote socket.
  With NVLink (DGX systems): less relevant (GPU-to-GPU directly).
""")

if torch.cuda.is_available():
    DEVICE = "cuda"
    cpu_pageable = torch.rand(n, dtype=torch.float32)
    cpu_pinned   = cpu_pageable.pin_memory()   # page-locked, faster DMA

    def h2d_pageable():
        return cpu_pageable.to(DEVICE)

    def h2d_pinned():
        return cpu_pinned.to(DEVICE, non_blocking=True)

    torch.cuda.synchronize()

    # Pageable
    times_p = []
    for _ in range(ITERS + 3):
        t0 = time.perf_counter()
        _ = h2d_pageable()
        torch.cuda.synchronize()
        times_p.append(time.perf_counter() - t0)
    times_p = times_p[3:]
    bw_page = (n*4) / (sum(times_p)/len(times_p)) / 1e9

    # Pinned
    times_pin = []
    for _ in range(ITERS + 3):
        t0 = time.perf_counter()
        _ = h2d_pinned()
        torch.cuda.synchronize()
        times_pin.append(time.perf_counter() - t0)
    times_pin = times_pin[3:]
    bw_pin = (n*4) / (sum(times_pin)/len(times_pin)) / 1e9

    print(f"  [{args.label}]  Pageable H2D : {bw_page:.1f} GB/s")
    print(f"  [{args.label}]  Pinned   H2D : {bw_pin:.1f} GB/s  ({bw_pin/bw_page:.2f}× vs pageable)")
    del cpu_pageable, cpu_pinned
else:
    print("  (CUDA not available — skipped)")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Save result for comparison
# ─────────────────────────────────────────────────────────────────────────────
result_file = f"numa_result_{args.label}.txt"
with open(result_file, "w") as f:
    f.write(f"label={args.label}\n")
    f.write(f"cpu_affinity={sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else 'unknown'}\n")
    f.write(f"mem_bandwidth_gbs={bw_gbs:.2f}\n")
    if torch.cuda.is_available():
        f.write(f"h2d_pageable_gbs={bw_page:.2f}\n")
        f.write(f"h2d_pinned_gbs={bw_pin:.2f}\n")

print(f"\n  Saved: {result_file}")
print(f"""
  COMPARE RESULTS:
    grep "mem_bandwidth" numa_result_*.txt
    grep "h2d_pinned"    numa_result_*.txt

  BINDING COMMANDS:
    numactl --cpunodebind=0 --membind=0 python numa_workload.py --label node0
    numactl --cpunodebind=1 --membind=0 python numa_workload.py --label cpu1_mem0
    numactl --cpunodebind=0 --membind=1 python numa_workload.py --label cpu0_mem1_remote

  TRAINING COMMAND (production):
    numactl --cpunodebind=$(cat /sys/bus/pci/devices/0000:01:00.0/numa_node) \\
            --membind=$(cat /sys/bus/pci/devices/0000:01:00.0/numa_node) \\
            python train.py
""")
