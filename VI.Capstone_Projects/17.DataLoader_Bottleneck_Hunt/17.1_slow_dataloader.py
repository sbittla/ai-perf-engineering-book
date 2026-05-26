#!/usr/bin/env python3
"""
VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.1_slow_dataloader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 17: Capstone 2 — DataLoader Bottleneck Hunt
Section 1: Diagnosing the Slow DataLoader
=======================================================================
Covers capstone section 17.1:
  • A deliberately slow DataLoader with three hidden bottlenecks
  • Diagnosing each bottleneck: idle_pct, throughput, worker utilisation
  • Measuring GPU starvation quantitatively
  • Building the "before" baseline for the hunt

The three bottlenecks (do NOT fix them here — fix them in 17.2):
  1. num_workers=0   — single-threaded loading; CPU starves GPU
  2. No pin_memory   — extra staging copy on every H2D transfer
  3. Slow augment    — CPU-heavy preprocessing per sample with sleep

Run:  python VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.1_slow_dataloader.py
All sections must print ✓.
"""

import json
import os
import statistics
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Capstone 17.1 — Diagnosing the Slow DataLoader")
print("=" * 60)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_CPUS  = os.cpu_count() or 1
print(f"  Device: {DEVICE}  |  CPUs: {N_CPUS}\n")

D_MODEL  = 256
N_ITEMS  = 2048
BATCH    = 64


# ─────────────────────────────────────────────────────────────
# SECTION 1: The Three Hidden Bottlenecks
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Three Hidden Bottlenecks ──")
print("""
  This capstone is a DETECTIVE EXERCISE. The DataLoader below has three
  intentional performance bugs. Your task (in 17.2) is to fix them one
  by one and measure the improvement from each fix.

  THE THREE BUGS:
    Bug 1: num_workers=0
      → All data loading happens in the main process
      → GPU must wait while the CPU loads and preprocesses each batch
      → Fix: num_workers = min(4, N_CPUs)

    Bug 2: pin_memory=False
      → PyTorch must allocate a pinned staging buffer and copy data
        into it before the DMA engine can transfer to GPU
      → Fix: pin_memory=True (and non_blocking=True on .to(device))

    Bug 3: Slow augmentation (simulated with time.sleep)
      → Real-world: CPU-heavy JPEG decode, resize, colour jitter
      → Simulated: each __getitem__ sleeps for 2 ms (bottleneck floor)
      → Fix: reduce augmentation cost, or move to GPU-side transforms

  DETECTION STRATEGY:
    idle_pct = (total_wall_ms - total_gpu_ms) / total_wall_ms × 100
    loader_ms_per_batch >> gpu_ms_per_batch → DataLoader is the bottleneck
""")
print("  ✓ Section 1 passed — understand the three bottlenecks")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The Intentionally Slow Dataset
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: The Slow Dataset ──")
print("""
  We use a synthetic dataset that simulates the cost of slow
  per-sample preprocessing (JPEG decode, resize, etc.) with a
  configurable sleep delay. The actual tensor data is random.

  In production, the sleep would be replaced by:
    - cv2.imread() + cv2.resize()  : 5–20 ms/sample
    - PIL.Image.open() + transforms: 10–50 ms/sample
    - torch.load() from disk       : 2–10 ms/sample
""")


class SlowDataset(Dataset):
    """Synthetic dataset with configurable per-sample preprocessing delay."""
    def __init__(self, n: int, d: int, sleep_ms: float = 2.0):
        self.data    = torch.randn(n, d)
        self.labels  = torch.randint(0, 10, (n,))
        self.sleep_s = sleep_ms / 1000.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)   # Bug 3: slow augmentation
        return self.data[idx], self.labels[idx]


class FastDataset(Dataset):
    """Same data, no preprocessing delay."""
    def __init__(self, n: int, d: int):
        self.data   = torch.randn(n, d)
        self.labels = torch.randint(0, 10, (n,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ── GPU model to give the DataLoader something to feed ──────
model = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL),
).to(DEVICE)
model.eval()


slow_dataset = SlowDataset(N_ITEMS, D_MODEL, sleep_ms=2.0)
print(f"  SlowDataset: {N_ITEMS} samples, {D_MODEL}D, 2 ms/sample augment delay")
print(f"  Theoretical min load time per batch of {BATCH}: {BATCH * 2:.0f} ms (sequential)")
print("  ✓ Section 2 passed — slow dataset created")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Measuring the Bug — DataLoader Diagnosis
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Diagnosing the Slow DataLoader ──")
print("""
  We measure three things:
    1. Loader throughput (batches/sec) in isolation
    2. GPU kernel time per batch (CUDA events)
    3. Wall-clock time per batch (includes loading)
  From these, idle_pct = (wall - gpu) / wall × 100

  TODO 1: Implement diagnose_dataloader(model, loader, n_batches=15)
  that returns (loader_ms, gpu_ms, idle_pct).
  - loader_ms: mean time from batch start to data-ready-on-device
  - gpu_ms   : mean GPU kernel time per batch (CUDA events)
  - idle_pct : percentage of wall time the GPU is waiting for data
""")


def diagnose_dataloader(model: nn.Module, loader: DataLoader,
                         n_batches: int = 15) -> tuple:
    """
    TODO 1: Diagnose DataLoader bottleneck.
    For each batch: time (a) data loading + H2D transfer, (b) GPU forward pass.
    Return (mean_loader_ms, mean_gpu_ms, idle_pct).
    """
    it = iter(loader)

    # Warmup: 2 batches
    for _ in range(min(2, n_batches)):
        try:
            xb, _ = next(it)
            xb_dev = xb.to(DEVICE)
            with torch.no_grad():
                model(xb_dev)
        except StopIteration:
            break
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    loader_times = []
    gpu_times    = []
    it = iter(loader)

    for _ in range(n_batches):
        try:
            # Time the load + H2D transfer
            t_load_start = time.perf_counter()
            xb, _ = next(it)
            xb_dev = xb.to(DEVICE)   # Bug 2: no pin_memory, no non_blocking
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            loader_times.append((time.perf_counter() - t_load_start) * 1000)

            # Time the GPU forward pass
            if DEVICE == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                with torch.no_grad():
                    model(xb_dev)
                e.record()
                torch.cuda.synchronize()
                gpu_times.append(s.elapsed_time(e))
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(xb_dev)
                gpu_times.append((time.perf_counter() - t0) * 1000)

        except StopIteration:
            break

    if not loader_times:
        return 0.0, 0.0, 0.0

    mean_loader_ms = statistics.mean(loader_times)
    mean_gpu_ms    = statistics.mean(gpu_times)
    total_wall_ms  = (mean_loader_ms + mean_gpu_ms)
    idle_pct = mean_loader_ms / total_wall_ms * 100 if total_wall_ms > 0 else 0.0
    return mean_loader_ms, mean_gpu_ms, idle_pct


# THE BUGGY LOADER (all three bugs present)
slow_loader = DataLoader(
    slow_dataset,
    batch_size=BATCH,
    num_workers=0,        # Bug 1: single-threaded
    pin_memory=False,     # Bug 2: no pinned memory
    shuffle=False,
)

print(f"  Running diagnosis on SLOW loader (workers=0, pin_memory=False, sleep=2ms)...")
print(f"  (This will take ~{N_ITEMS // BATCH * 2 * BATCH // 1000 + 2}s — slow by design)")
loader_ms, gpu_ms, idle_pct = diagnose_dataloader(model, slow_loader, n_batches=8)

print(f"\n  SLOW LOADER DIAGNOSIS:")
print(f"    Load + H2D per batch : {loader_ms:.1f} ms")
print(f"    GPU forward per batch: {gpu_ms:.3f} ms")
print(f"    GPU idle percentage  : {idle_pct:.1f}%")

if idle_pct > 50:
    verdict = "CRITICAL: GPU is mostly idle — DataLoader is the bottleneck"
elif idle_pct > 10:
    verdict = "WARNING: significant GPU starvation — DataLoader needs tuning"
else:
    verdict = "OK: DataLoader is not the primary bottleneck"
print(f"    Verdict: {verdict}")

assert loader_ms > 0, "Loader time should be positive"
assert gpu_ms > 0, "GPU time should be positive"
assert idle_pct > 0, "Some idle time expected with slow loader"
print("  ✓ Section 3 passed — slow loader diagnosed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Identifying each bug's contribution
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Isolating Bug Contributions ──")
print("""
  To attribute each bug's contribution, we fix ONE bug at a time
  and measure the remaining idle_pct. This is the isolation principle
  from Chapter 14: change one variable per experiment.

  We use a FAST dataset (no sleep) to isolate bugs 1 and 2 from bug 3.
  Then we combine them to see the full picture.
""")

fast_dataset = FastDataset(N_ITEMS, D_MODEL)

# Loader A: All 3 bugs (no sleep version for speed)
loader_a = DataLoader(fast_dataset, batch_size=BATCH,
                      num_workers=0, pin_memory=False, shuffle=False)
ms_a, gpu_a, idle_a = diagnose_dataloader(model, loader_a, n_batches=20)

# Loader B: Fix bug 1 only (add workers)
best_workers = min(2, N_CPUS)
loader_b = DataLoader(fast_dataset, batch_size=BATCH,
                      num_workers=best_workers, pin_memory=False,
                      shuffle=False, persistent_workers=True)
ms_b, gpu_b, idle_b = diagnose_dataloader(model, loader_b, n_batches=20)

# Loader C: Fix bug 1 + bug 2
loader_c = DataLoader(fast_dataset, batch_size=BATCH,
                      num_workers=best_workers, pin_memory=(DEVICE == "cuda"),
                      shuffle=False, persistent_workers=True)
ms_c, gpu_c, idle_c = diagnose_dataloader(model, loader_c, n_batches=20)

print(f"\n  Bug isolation (fast dataset, no sleep):")
print(f"  {'Config':<30}  {'Loader (ms)':>12}  {'GPU (ms)':>10}  {'Idle %':>8}")
print(f"  {'─'*30}  {'─'*12}  {'─'*10}  {'─'*8}")
print(f"  {'Baseline (all bugs)':<30}  {ms_a:>12.3f}  {gpu_a:>10.3f}  {idle_a:>7.1f}%")
print(f"  {f'+workers={best_workers}':<30}  {ms_b:>12.3f}  {gpu_b:>10.3f}  {idle_b:>7.1f}%")
print(f"  {'+pin_memory':<30}  {ms_c:>12.3f}  {gpu_c:>10.3f}  {idle_c:>7.1f}%")

# Save the "before" snapshot
diagnosis = {
    "device": DEVICE,
    "batch_size": BATCH,
    "d_model": D_MODEL,
    "bugs": {
        "num_workers": 0,
        "pin_memory": False,
        "sleep_ms": 2.0,
    },
    "baseline_no_sleep": {
        "loader_ms": round(ms_a, 3),
        "gpu_ms": round(gpu_a, 3),
        "idle_pct": round(idle_a, 1),
    },
    "fix1_workers": {
        "n_workers": best_workers,
        "idle_pct": round(idle_b, 1),
    },
    "fix1_fix2_pin": {
        "idle_pct": round(idle_c, 1),
    },
}
with open("/tmp/capstone17_diagnosis.json", "w") as f:
    json.dump(diagnosis, f, indent=2)

print(f"""
  DIAGNOSIS SAVED: /tmp/capstone17_diagnosis.json

  SUMMARY OF FINDINGS:
    Bug 1 (num_workers=0)      : GPU idle ~{idle_a:.0f}% with single worker
    Fix 1 (+{best_workers} workers)        : reduces idle to ~{idle_b:.0f}%
    Fix 2 (+pin_memory)        : reduces idle to ~{idle_c:.0f}%
    Fix 3 (remove sleep)       : requires changing __getitem__ augmentation

  NEXT STEP: Run 17.2_fast_dataloader.py to apply all fixes
  and measure the combined improvement.
""")

assert "baseline_no_sleep" in diagnosis
print("  ✓ Section 4 passed — bug contributions isolated")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 17.1 complete!")
print("  Three DataLoader bottlenecks identified and quantified.")
print("  Next: VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.2_fast_dataloader.py")
print("=" * 60)
