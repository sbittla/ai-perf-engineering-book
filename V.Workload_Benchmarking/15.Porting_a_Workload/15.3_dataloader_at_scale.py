#!/usr/bin/env python3
"""
V.Workload_Benchmarking/15.Porting_a_Workload/15.3_dataloader_at_scale.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 15: Porting a Workload — Section 3: DataLoader at Scale
=======================================================================
Covers book section 15.3:
  • Why the DataLoader becomes the bottleneck after GPU optimisation
  • num_workers sweep: finding the optimal worker count
  • pin_memory: why it speeds up CPU→GPU batch transfers
  • CPU affinity: binding workers to cores to reduce NUMA effects
  • idle_pct: measuring GPU starvation from a slow DataLoader
  • prefetch_factor: overlapping data loading with GPU compute

Run:  python V.Workload_Benchmarking/15.Porting_a_Workload/15.3_dataloader_at_scale.py
All sections must print ✓.
"""

import os
import statistics
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Exercise 15.3 — DataLoader at Scale")
print("=" * 60)

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_CPUS   = os.cpu_count() or 1
print(f"  Device: {DEVICE}  |  CPUs: {N_CPUS}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Why the DataLoader becomes the bottleneck
# ─────────────────────────────────────────────────────────────
print("── Section 1: The DataLoader Bottleneck ──")
print("""
  After you port a model to GPU and apply FP16, you often find that
  the GPU is still underutilised. The culprit: the DataLoader.

  GPU STARVATION PATTERN:
    Time →  [load batch][GPU compute]  [load batch][GPU compute]  ...
                  ↑ GPU idle here!

  The GPU waits for the CPU to prepare the next batch. In a well-tuned
  pipeline, data loading overlaps with GPU compute:
    Time →  [load B2]      [load B3]      [load B4] ...
               [GPU: B1]     [GPU: B2]     [GPU: B3] ...

  THREE LEVERS TO TUNE:
    1. num_workers     — parallel CPU workers for data loading
    2. pin_memory      — avoids CPU-side copy before DMA transfer
    3. prefetch_factor — how many batches each worker pre-fetches

  MEASURING STARVATION:
    idle_pct = time_waiting_for_data / total_time × 100
    If idle_pct > 10%, your DataLoader is the bottleneck.

  RULE OF THUMB FOR num_workers:
    Start with min(4, N_CPUS // 2). Never exceed N_CPUS.
    If you have NVMe storage, higher is better. If spinning disk, lower.
""")
print("  ✓ Section 1 passed — understand GPU starvation from DataLoader")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Synthetic dataset and baseline DataLoader
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Measuring DataLoader Throughput ──")
print("""
  We use a synthetic in-memory dataset to isolate DataLoader overhead
  from disk I/O. Real workloads add disk latency on top of this.

  TODO 1: Implement measure_loader_throughput(loader, n_batches=50)
  that iterates the loader for n_batches, times each iteration,
  and returns (mean_iter_ms, batches_per_sec).
""")

D_MODEL  = 256
N_ITEMS  = 4096


class SyntheticDataset(Dataset):
    """In-memory random tensors — isolates DataLoader from disk I/O."""
    def __init__(self, n: int, d: int):
        self.data   = torch.randn(n, d)
        self.labels = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = SyntheticDataset(N_ITEMS, D_MODEL)


def measure_loader_throughput(loader: DataLoader, n_batches: int = 50) -> tuple:
    """
    TODO 1: Measure mean iteration time and batches/sec.
    Iterate the loader up to n_batches times. Time each iteration.
    mean_iter_ms = mean of iteration times in ms
    batches_per_sec = 1000 / mean_iter_ms
    Return (mean_iter_ms, batches_per_sec).
    """
    times = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            t0 = time.perf_counter()
            batch = next(it)
            # simulate minimal GPU transfer (pin_memory effect measurable here)
            if DEVICE == "cuda" and loader.pin_memory:
                _ = batch[0].to(DEVICE, non_blocking=True)
                torch.cuda.synchronize()
            else:
                _ = batch[0].to(DEVICE)
            times.append((time.perf_counter() - t0) * 1000)
        except StopIteration:
            it = iter(loader)

    mean_iter_ms = statistics.mean(times)
    batches_per_sec = 1000.0 / mean_iter_ms
    return mean_iter_ms, batches_per_sec


# Baseline: num_workers=0 (main process, sequential loading)
loader_0 = DataLoader(dataset, batch_size=64, num_workers=0,
                      pin_memory=(DEVICE == "cuda"), shuffle=False)
mean_0, bps_0 = measure_loader_throughput(loader_0, n_batches=30)
print(f"  num_workers=0  : {mean_0:.3f} ms/batch  {bps_0:.0f} batches/s")

assert mean_0 > 0, "Iteration time should be positive"
assert bps_0 > 0, "Batches per second should be positive"
print("  ✓ Section 2 passed — DataLoader throughput measurement works")


# ─────────────────────────────────────────────────────────────
# SECTION 3: num_workers sweep
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: num_workers Sweep ──")
print("""
  Each worker is a separate process that loads data in parallel with
  the main training loop. More workers → more CPU parallelism, but
  also more overhead from process spawning and inter-process queuing.

  The optimal num_workers is hardware-dependent. You must sweep it.

  TODO 2: Implement sweep_num_workers(dataset, batch_size, worker_counts)
  that benchmarks each worker count and returns a list of
  (n_workers, mean_ms, bps) tuples, sorted by worker count.
""")


def sweep_num_workers(dataset: Dataset, batch_size: int,
                      worker_counts: list) -> list:
    """
    TODO 2: Sweep num_workers and return [(n_workers, mean_ms, bps), ...].
    For each count in worker_counts, create a DataLoader and call
    measure_loader_throughput(loader, n_batches=30).
    """
    results = []
    for nw in worker_counts:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=nw,
            pin_memory=(DEVICE == "cuda"),
            shuffle=False,
            persistent_workers=(nw > 0),
        )
        mean_ms, bps = measure_loader_throughput(loader, n_batches=30)
        results.append((nw, mean_ms, bps))
        # close loader explicitly to free workers
        del loader
    return results


# Sweep up to min(4, N_CPUS) workers
max_workers = min(4, N_CPUS)
worker_counts = sorted(set([0, 1, 2, max_workers]))

print(f"  Sweeping num_workers in {worker_counts} (CPU count: {N_CPUS})")
sweep_results = sweep_num_workers(dataset, batch_size=64, worker_counts=worker_counts)

print(f"\n  {'workers':>8}  {'ms/batch':>10}  {'batches/s':>12}  {'speedup':>10}")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*10}")
base_bps = sweep_results[0][2]
for nw, mean_ms, bps in sweep_results:
    speedup = bps / base_bps
    print(f"  {nw:>8}  {mean_ms:>10.3f}  {bps:>12.0f}  {speedup:>9.2f}×")

# Find optimal
best = max(sweep_results, key=lambda r: r[2])
print(f"\n  Best: num_workers={best[0]} → {best[2]:.0f} batches/s")

assert len(sweep_results) == len(worker_counts), "Should have one result per worker count"
assert all(bps > 0 for _, _, bps in sweep_results), "All bps should be positive"
print("  ✓ Section 3 passed — num_workers sweep complete")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Measuring GPU idle time (starvation)
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Measuring GPU Idle Time (Starvation) ──")
print("""
  Throughput of the DataLoader alone is not enough. You must also
  measure how much time the GPU spends idle waiting for the next batch.

  IDLE PCT FORMULA:
    For each batch i:
      data_load_time[i] = time from end of GPU compute[i-1] to end of data load[i]
      gpu_compute_time[i] = time from start of GPU compute[i] to end

    idle_pct = sum(max(0, data_load[i] - gpu_compute[i])) / total_time × 100

  SIMPLIFIED (wall-clock method):
    total_wall = time to process N_BATCHES
    gpu_only   = N_BATCHES × gpu_ms_per_batch   (from CUDA events)
    idle_time  = total_wall - gpu_only
    idle_pct   = idle_time / total_wall × 100

  TODO 3: Implement measure_idle_pct(model, loader, n_batches=20)
  that runs a model over n_batches from the loader and returns idle_pct.
""")

model_for_idle = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL),
).to(DEVICE)
model_for_idle.eval()


def measure_idle_pct(model: nn.Module, loader: DataLoader,
                     n_batches: int = 20) -> float:
    """
    TODO 3: Return idle_pct for the given model and loader.
    Wall-clock method:
      1. Record total wall time for n_batches (load + compute)
      2. Measure GPU-only time per batch using CUDA events (or perf_counter)
      3. idle_pct = max(0, total_wall - total_gpu) / total_wall * 100
    """
    it = iter(loader)

    # Warmup: 3 batches
    for _ in range(min(3, n_batches)):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, _ = next(it)
        with torch.no_grad():
            model(xb.to(DEVICE))
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Measure total wall time
    t_wall_start = time.perf_counter()
    total_gpu_ms = 0.0
    it = iter(loader)
    for _ in range(n_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            it = iter(loader)
            xb, _ = next(it)

        xb_dev = xb.to(DEVICE)
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                model(xb_dev)
            e.record()
            torch.cuda.synchronize()
            total_gpu_ms += s.elapsed_time(e)
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                model(xb_dev)
            total_gpu_ms += (time.perf_counter() - t0) * 1000

    total_wall_ms = (time.perf_counter() - t_wall_start) * 1000
    idle_ms  = max(0.0, total_wall_ms - total_gpu_ms)
    idle_pct = idle_ms / total_wall_ms * 100.0
    return idle_pct


# Compare idle_pct for num_workers=0 vs best_workers
loader_0w = DataLoader(dataset, batch_size=64, num_workers=0,
                       pin_memory=(DEVICE == "cuda"), shuffle=False)
idle_0 = measure_idle_pct(model_for_idle, loader_0w, n_batches=20)

loader_best = DataLoader(dataset, batch_size=64, num_workers=best[0],
                         pin_memory=(DEVICE == "cuda"), shuffle=False,
                         persistent_workers=(best[0] > 0))
idle_best = measure_idle_pct(model_for_idle, loader_best, n_batches=20)

print(f"\n  num_workers=0        idle_pct: {idle_0:.1f}%")
print(f"  num_workers={best[0]} (best)  idle_pct: {idle_best:.1f}%")

if idle_best < idle_0:
    print(f"  More workers reduced GPU starvation by {idle_0 - idle_best:.1f}pp")
else:
    print(f"  (In-memory dataset: DataLoader is not the bottleneck)")

if idle_best > 10:
    print(f"  WARNING: {idle_best:.1f}% idle — DataLoader is still the bottleneck")
    print(f"  → Try: more workers, pin_memory=True, prefetch_factor=4")
else:
    print(f"  OK: < 10% idle — DataLoader is not starving the GPU")

assert 0 <= idle_0 <= 100, "Idle pct should be in [0, 100]"
assert 0 <= idle_best <= 100, "Idle pct should be in [0, 100]"
print("  ✓ Section 4 passed — GPU idle pct measured")


# ─────────────────────────────────────────────────────────────
# SECTION 5: CPU affinity and DataLoader tuning checklist
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: CPU Affinity and Tuning Checklist ──")
print("""
  CPU affinity binds DataLoader worker processes to specific CPU cores.
  This matters on NUMA systems (multi-socket servers) where a worker
  on socket 1 reading data into socket 0 memory wastes ~30% extra bandwidth.

  CHECKING AFFINITY:
    import os
    cores = os.sched_getaffinity(0)   # set of cores available to this process
    print(f"Process can run on cores: {sorted(cores)}")

  SETTING AFFINITY PER WORKER (via worker_init_fn):
    def worker_init(worker_id):
        # Bind worker to its own core (if cores >= n_workers)
        core = worker_id % os.cpu_count()
        os.sched_setaffinity(0, {core})

  DATALOADER TUNING CHECKLIST:
    □ num_workers: sweep from 0 to min(8, N_CPUS) — pick the knee
    □ pin_memory=True: always True when DEVICE == "cuda"
    □ persistent_workers=True: avoid worker process respawning per epoch
    □ prefetch_factor=2 (default): increase to 4 if idle_pct > 10%
    □ worker_init_fn: set CPU affinity on NUMA systems
    □ drop_last=True: avoids variable-size last batch overhead
    □ Use in-memory dataset (e.g. mmap) for frequently accessed data
""")

# Check available CPU affinity
try:
    available_cores = sorted(os.sched_getaffinity(0))
    print(f"  Available CPU cores for this process: {available_cores}")
    n_affinity_cores = len(available_cores)
except AttributeError:
    available_cores = list(range(N_CPUS))
    n_affinity_cores = N_CPUS
    print(f"  os.sched_getaffinity not available on this OS — assuming {N_CPUS} cores")

print(f"  Cores available: {n_affinity_cores}  (NUMA binding matters when N_CPUS > 8)")

# Demonstrate worker_init_fn pattern
def affinity_worker_init(worker_id: int):
    """Bind each DataLoader worker to a distinct core."""
    if hasattr(os, "sched_setaffinity"):
        core = available_cores[worker_id % len(available_cores)]
        os.sched_setaffinity(0, {core})

# Build an affinity-pinned loader
if best[0] > 0:
    loader_affinity = DataLoader(
        dataset, batch_size=64, num_workers=best[0],
        pin_memory=(DEVICE == "cuda"), shuffle=False,
        persistent_workers=True, prefetch_factor=2,
        worker_init_fn=affinity_worker_init,
    )
    mean_aff, bps_aff = measure_loader_throughput(loader_affinity, n_batches=30)
    print(f"\n  Affinity-bound loader (workers={best[0]}, prefetch=2):")
    print(f"    {mean_aff:.3f} ms/batch  {bps_aff:.0f} batches/s")
    del loader_affinity
else:
    print(f"\n  (num_workers=0: affinity has no effect)")

print(f"""
  FINAL DATALOADER TUNING RECIPE:
    loader = DataLoader(
        dataset,
        batch_size    = batch_size,
        num_workers   = {best[0]},         # from sweep
        pin_memory    = {DEVICE == "cuda"},
        persistent_workers = {best[0] > 0},
        prefetch_factor    = 2,      # increase to 4 if idle_pct > 10%
        drop_last          = True,   # avoid variable last-batch overhead
        worker_init_fn     = affinity_worker_init,  # NUMA systems only
    )
""")
print("  ✓ Section 5 passed — CPU affinity and DataLoader checklist covered")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 15.3 complete!")
print("  You can now measure DataLoader throughput, sweep num_workers,")
print("  measure GPU idle time, and apply the DataLoader tuning checklist.")
print("  Part V — Workload Benchmarking complete!")
print("=" * 60)
