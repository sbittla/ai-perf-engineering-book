#!/usr/bin/env python3
"""
dataloader_worker.py  —  Phase 5, Module 10/11: CPU Affinity & NUMA Binding Demo
===================================================================================

HOW TO RUN:
    # Run the benchmark itself:
    python dataloader_worker.py

    # Run with taskset (restrict to cores 0-7):
    taskset -c 0-7 python dataloader_worker.py --label "taskset_cores_0-7"

    # Run with NUMA binding:
    numactl --cpunodebind=0 --membind=0 python dataloader_worker.py --label "numa_node0"

    # Compare all three manually:
    python dataloader_worker.py --label "default" &> results_default.txt
    taskset -c 0-7 python dataloader_worker.py --label "taskset" &> results_taskset.txt
    numactl --cpunodebind=0 --membind=0 python dataloader_worker.py --label "numa" &> results_numa.txt
    grep "Throughput" results_*.txt


"""

import argparse
import time
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

parser = argparse.ArgumentParser()
parser.add_argument("--workers",   type=int, default=4,    help="Number of DataLoader workers")
parser.add_argument("--batch",     type=int, default=32,   help="Batch size")
parser.add_argument("--iters",     type=int, default=100,  help="Iterations to benchmark")
parser.add_argument("--img-size",  type=int, default=224)
parser.add_argument("--pin",       action="store_true", default=True,
                    help="Use pin_memory (default: on)")
parser.add_argument("--no-pin",    dest="pin", action="store_false")
parser.add_argument("--label",     default="default",
                    help="Label for this run (for comparing results)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Show current CPU affinity ─────────────────────────────────────────────────
def get_cpu_affinity():
    """Return set of CPU cores this process is allowed to run on."""
    try:
        return os.sched_getaffinity(0)
    except AttributeError:
        return set()

def get_numa_info():
    """Return basic NUMA topology if numactl is available."""
    try:
        import subprocess
        r = subprocess.run(["numactl", "--hardware"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip()[:300]
    except Exception:
        return "numactl not available"

affinity = get_cpu_affinity()
print(f"\n{'='*60}")
print(f"  dataloader_worker.py — CPU Affinity & NUMA Benchmark")
print(f"{'='*60}")
print(f"  Label      : {args.label}")
print(f"  PID        : {os.getpid()}")
print(f"  CPU cores allowed : {sorted(affinity) if affinity else 'all (no restriction)'}")
print(f"  Num workers: {args.workers}")
print(f"  Batch size : {args.batch}")
print(f"  pin_memory : {args.pin}")
print(f"  Device     : {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU        : {props.name}")
print()

# Try to get which NUMA node the GPU is on
try:
    import subprocess
    numa_r = subprocess.run(
        ["cat", "/sys/bus/pci/devices/0000:01:00.0/numa_node"],
        capture_output=True, text=True
    )
    if numa_r.returncode == 0:
        print(f"  GPU NUMA node: {numa_r.stdout.strip()}")
        print(f"  For best performance: numactl --cpunodebind={numa_r.stdout.strip()} --membind={numa_r.stdout.strip()} python ...")
except Exception:
    pass

# ── Synthetic Dataset ──────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    """
    Synthetic image dataset that simulates realistic per-sample CPU work:
    - A small decode delay (reads from a pre-allocated buffer to simulate disk)
    - Optional CPU transforms
    """
    def __init__(self, size, img_size):
        self.size     = size
        self.img_size = img_size
        # Pre-allocate a 100MB buffer to simulate reading from mmap'd dataset file
        # Workers access different offsets to simulate concurrent file reads
        buf_size = 100 * 1024 * 1024 // 4   # 100MB of float32
        self.buffer = torch.rand(buf_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate reading from a mmap'd dataset file:
        # Access a slice of the buffer (simulates disk read latency indirectly
        # through cache effects — workers competing for L3 cache)
        offset = (idx * self.img_size * self.img_size * 3) % (len(self.buffer) - self.img_size**2 * 3)
        _ = self.buffer[offset:offset + self.img_size * 3].sum()  # force cache load

        # Generate image tensor (in real code: PIL.Image.open(path))
        img   = torch.rand(3, self.img_size, self.img_size)
        label = idx % 1000
        return img, label


# ── DataLoader setup ──────────────────────────────────────────────────────────
n_samples = args.batch * args.iters * 2
dataset   = ImageDataset(n_samples, args.img_size)

dataloader = DataLoader(
    dataset,
    batch_size=args.batch,
    num_workers=args.workers,
    pin_memory=args.pin and (device == "cuda"),
    prefetch_factor=2 if args.workers > 0 else None,
    persistent_workers=True if args.workers > 0 else False,
    shuffle=False,
)

# ── Benchmark: measure batch delivery throughput ──────────────────────────────
print(f"Benchmarking batch delivery ({args.iters} iterations)...")
print(f"This measures how fast the DataLoader delivers batches to the main thread.\n")

# Each iteration: DataLoader delivers one batch to this process.
# The main process moves it to GPU immediately to measure the full pipeline.

batch_times_ms = []
transfer_times_ms = []
total_start = time.perf_counter()

for i, (images, labels) in enumerate(dataloader):
    if i >= args.iters:
        break

    t0 = time.perf_counter()

    # H2D transfer (this is what pin_memory + NUMA binding accelerates)
    if device == "cuda":
        images_gpu = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        torch.cuda.synchronize()   # Wait for transfer to complete

    transfer_ms = (time.perf_counter() - t0) * 1000
    transfer_times_ms.append(transfer_ms)

    # Simulate minimal GPU work (just to keep GPU busy, not measure it here)
    if device == "cuda":
        with torch.no_grad():
            _ = images_gpu.mean()   # trivial op

    batch_times_ms.append(transfer_ms)

    if i % 20 == 0:
        avg_ms = sum(batch_times_ms[-20:]) / min(20, len(batch_times_ms))
        print(f"  iter {i:4d}  batch_delivery+transfer={avg_ms:.1f}ms  "
              f"shape={list(images.shape)}")

total_elapsed = time.perf_counter() - total_start
import statistics

print(f"\n{'='*60}")
print(f"  RESULTS — {args.label}")
print(f"{'='*60}")
print(f"  Iterations      : {args.iters}")
print(f"  Total time      : {total_elapsed:.1f}s")
print(f"  Batches/sec     : {args.iters / total_elapsed:.1f}")
print(f"  Samples/sec     : {args.iters * args.batch / total_elapsed:.0f}")
print(f"  Transfer P50    : {statistics.median(transfer_times_ms):.2f}ms")
print(f"  Transfer P99    : {sorted(transfer_times_ms)[int(0.99*len(transfer_times_ms))]:.2f}ms")
print(f"  CPU cores used  : {sorted(affinity) if affinity else 'all'}")

# Save results for comparison
with open(f"results_{args.label}.txt", "w") as f:
    f.write(f"Label           : {args.label}\n")
    f.write(f"CPU affinity    : {sorted(affinity) if affinity else 'all'}\n")
    f.write(f"Workers         : {args.workers}\n")
    f.write(f"pin_memory      : {args.pin}\n")
    f.write(f"Batches/sec     : {args.iters / total_elapsed:.1f}\n")
    f.write(f"Samples/sec     : {args.iters * args.batch / total_elapsed:.0f}\n")
    f.write(f"Transfer P50 ms : {statistics.median(transfer_times_ms):.2f}\n")
    f.write(f"Transfer P99 ms : {sorted(transfer_times_ms)[int(0.99*len(transfer_times_ms))]:.2f}\n")

print(f"\n  Saved: results_{args.label}.txt")
print(f"""
  COMPARE RUNS:
    python dataloader_worker.py --label default
    taskset -c 0-7 python dataloader_worker.py --label taskset
    numactl --cpunodebind=0 --membind=0 python dataloader_worker.py --label numa
    grep "Batches/sec" results_*.txt
  """)
