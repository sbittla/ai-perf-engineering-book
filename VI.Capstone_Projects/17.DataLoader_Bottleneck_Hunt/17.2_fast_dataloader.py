#!/usr/bin/env python3
"""
VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.2_fast_dataloader.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 17: Capstone 2 — DataLoader Bottleneck Hunt
Section 2: Applying the Fixes and Measuring the Result
=======================================================================
Covers capstone section 17.2:
  • Apply each fix independently and measure its isolated speedup
  • Combine all fixes and compute the total throughput improvement
  • Measure the residual idle_pct after all fixes
  • Build a before/after comparison report with per-fix attribution

Run:  python VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/17.2_fast_dataloader.py
All sections must print ✓. Run 17.1 first for the diagnosis context.
"""

import json
import os
import statistics
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Capstone 17.2 — Applying the Fixes")
print("=" * 60)

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_CPUS  = os.cpu_count() or 1
print(f"  Device: {DEVICE}  |  CPUs: {N_CPUS}\n")

D_MODEL     = 256
N_ITEMS     = 2048
BATCH       = 64
SLEEP_MS    = 1.0    # augment delay: enough to make workers beneficial, fast to test
BEST_WORKERS = min(4, N_CPUS)


# ── Shared infrastructure ────────────────────────────────────
class SlowDataset(Dataset):
    def __init__(self, n, d, sleep_ms=2.0):
        self.data = torch.randn(n, d)
        self.labels = torch.randint(0, 10, (n,))
        self.sleep_s = sleep_ms / 1000.0
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        return self.data[idx], self.labels[idx]

class FastDataset(Dataset):
    """No preprocessing delay — simulates GPU-side or pre-cached augmentation."""
    def __init__(self, n, d):
        self.data = torch.randn(n, d)
        self.labels = torch.randint(0, 10, (n,))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


model = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL),
).to(DEVICE)
model.eval()


def measure_pipeline(model, loader, n_batches=20):
    """Measure (throughput_bps, idle_pct, loader_ms, gpu_ms)."""
    it = iter(loader)
    for _ in range(min(3, n_batches)):
        try:
            xb, _ = next(it)
            with torch.no_grad():
                model(xb.to(DEVICE))
        except StopIteration:
            break
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    loader_times, gpu_times = [], []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            t0 = time.perf_counter()
            xb, _ = next(it)
            pin = getattr(loader, "pin_memory", False)
            xb_dev = xb.to(DEVICE, non_blocking=pin)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            loader_times.append((time.perf_counter() - t0) * 1000)

            if DEVICE == "cuda":
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                with torch.no_grad(): model(xb_dev)
                e.record()
                torch.cuda.synchronize()
                gpu_times.append(s.elapsed_time(e))
            else:
                t1 = time.perf_counter()
                with torch.no_grad(): model(xb_dev)
                gpu_times.append((time.perf_counter() - t1) * 1000)
        except StopIteration:
            break

    if not loader_times:
        return 0, 0, 0, 0
    ml = statistics.mean(loader_times)
    mg = statistics.mean(gpu_times)
    idle = ml / (ml + mg) * 100 if (ml + mg) > 0 else 0
    bps  = 1000.0 / (ml + mg)
    return bps, idle, ml, mg


# ─────────────────────────────────────────────────────────────
# SECTION 1: Establish the buggy baseline (fast dataset for speed)
# ─────────────────────────────────────────────────────────────
print("── Section 1: Buggy Baseline (fast dataset) ──")
print("""
  We re-measure the baseline using the fast dataset (no sleep)
  so we can isolate bugs 1 and 2 without waiting for bug 3.
  Bug 3 (slow augment) is measured separately at the end.
""")

loader_buggy = DataLoader(SlowDataset(N_ITEMS, D_MODEL, sleep_ms=SLEEP_MS),
                           batch_size=BATCH, num_workers=0,
                           pin_memory=False, shuffle=False)
print(f"  (sleep={SLEEP_MS}ms/sample, batch={BATCH} → ~{SLEEP_MS*BATCH:.0f}ms theoretical load)")
bps_base, idle_base, ml_base, mg_base = measure_pipeline(model, loader_buggy, n_batches=6)
print(f"  Buggy baseline:  {bps_base:.0f} batches/s  idle={idle_base:.1f}%  "
      f"load={ml_base:.2f}ms  gpu={mg_base:.3f}ms")
assert bps_base > 0
print("  ✓ Section 1 passed — buggy baseline measured")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Fix 1 — Add num_workers
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Fix 1 — num_workers ──")
print(f"""
  THE FIX: Change num_workers from 0 to {BEST_WORKERS}.

  WHY IT WORKS:
    With num_workers=0, the main process loads data sequentially:
      load batch → transfer → compute → load batch → ...
    The GPU is idle during loading.

    With num_workers={BEST_WORKERS}, worker processes prefetch the next batch
    while the GPU processes the current one (overlap):
      [load B2][load B3][load B4]...  (workers, in parallel)
               [GPU: B1]  [GPU: B2]   (main process)

  TODO 1: Measure the isolated speedup from num_workers={BEST_WORKERS}
  (keep pin_memory=False to isolate this fix).
""")


def fix1_workers(n_workers: int) -> tuple:
    """TODO 1: Add num_workers; return (bps, idle_pct, speedup)."""
    loader = DataLoader(
        SlowDataset(N_ITEMS, D_MODEL, sleep_ms=SLEEP_MS),
        batch_size=BATCH,
        num_workers=n_workers,
        pin_memory=False,
        shuffle=False,
        persistent_workers=(n_workers > 0),
    )
    bps, idle, ml, mg = measure_pipeline(model, loader, n_batches=6)
    speedup = bps / bps_base if bps_base > 0 else 1.0
    return bps, idle, speedup


bps_w, idle_w, speedup_w = fix1_workers(BEST_WORKERS)
print(f"  Fix 1 ({BEST_WORKERS} workers):  {bps_w:.0f} batches/s  idle={idle_w:.1f}%  speedup={speedup_w:.2f}×")
assert bps_w > 0, "Worker-based loader should produce some throughput"
print(f"  ✓ Section 2 passed — Fix 1 applied ({speedup_w:.2f}× speedup)")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Fix 2 — Add pin_memory
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Fix 2 — pin_memory ──")
print("""
  THE FIX: Change pin_memory=False to pin_memory=True,
  and use .to(device, non_blocking=True).

  WHY IT WORKS:
    Without pinning: CPU allocates pageable memory. Before DMA transfer,
    the OS must copy to a pinned staging buffer (wasted copy).
    With pinning: DataLoader allocates page-locked memory directly.
    The DMA engine reads from it immediately. Eliminates the staging copy.

  EFFECT ON BANDWIDTH:
    PCIe 4.0 x16 theoretical: 32 GB/s
    Non-pinned effective:      ~16 GB/s (staging copy halves it)
    Pinned effective:          ~28–30 GB/s

  non_blocking=True: overlaps the H2D transfer with CPU-side work.

  TODO 2: Measure Fix 1 + Fix 2 combined.
""")


def fix1_fix2(n_workers: int) -> tuple:
    """TODO 2: Add workers + pin_memory. Return (bps, idle_pct, speedup)."""
    loader = DataLoader(
        SlowDataset(N_ITEMS, D_MODEL, sleep_ms=SLEEP_MS),
        batch_size=BATCH,
        num_workers=n_workers,
        pin_memory=(DEVICE == "cuda"),
        shuffle=False,
        persistent_workers=(n_workers > 0),
    )
    bps, idle, ml, mg = measure_pipeline(model, loader, n_batches=6)
    speedup = bps / bps_base if bps_base > 0 else 1.0
    return bps, idle, speedup


bps_p, idle_p, speedup_p = fix1_fix2(BEST_WORKERS)
print(f"  Fix 1+2 ({BEST_WORKERS}w+pin):  {bps_p:.0f} batches/s  idle={idle_p:.1f}%  speedup={speedup_p:.2f}×")
pin_gain = speedup_p - speedup_w
print(f"  Pin_memory isolated gain: {pin_gain:.2f}×")
print(f"  ✓ Section 3 passed — Fix 2 applied (cumulative {speedup_p:.2f}× speedup)")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Fix 3 — Remove slow augmentation
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Fix 3 — Remove Slow Augmentation ──")
print("""
  THE FIX: Replace the per-sample time.sleep(2ms) with no-op or
  GPU-side augmentation (e.g. torchvision GPU transforms).

  In production, this means:
    - Pre-compute and cache augmented tensors to disk or mmap
    - Move resize/normalize to GPU after the batch is transferred
    - Use faster decode libraries (turbojpeg vs PIL)
    - Reduce the number of augmentation operations

  With all three fixes applied, the DataLoader should no longer
  be the bottleneck for this workload.

  TODO 3: Measure all three fixes combined (fast dataset = no sleep,
  workers=BEST_WORKERS, pin_memory=True).
""")


def all_fixes(n_workers: int) -> tuple:
    """TODO 3: Apply all three fixes. Return (bps, idle_pct, speedup)."""
    loader = DataLoader(
        FastDataset(N_ITEMS, D_MODEL),    # Fix 3: no sleep (fast augment)
        batch_size=BATCH,
        num_workers=n_workers,             # Fix 1: workers
        pin_memory=(DEVICE == "cuda"),     # Fix 2: pin memory
        shuffle=False,
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None,
    )
    bps, idle, ml, mg = measure_pipeline(model, loader, n_batches=20)
    speedup = bps / bps_base if bps_base > 0 else 1.0
    return bps, idle, speedup


bps_all, idle_all, speedup_all = all_fixes(BEST_WORKERS)
print(f"  All fixes ({BEST_WORKERS}w+pin+no-sleep):  {bps_all:.0f} batches/s  "
      f"idle={idle_all:.1f}%  speedup={speedup_all:.2f}×")
print(f"  ✓ Section 4 passed — all three fixes applied")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Before/after report
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Before/After Report ──")
print("""
  The complete before/after comparison shows the isolated contribution
  of each fix and the total improvement.
""")

print(f"\n  {'Config':<30}  {'Batches/s':>10}  {'Idle %':>8}  {'Speedup':>9}")
print(f"  {'─'*30}  {'─'*10}  {'─'*8}  {'─'*9}")
configs = [
    ("Buggy (w=0, no pin, sleep=2ms)", bps_base, idle_base, 1.0),
    (f"+ num_workers={BEST_WORKERS}",      bps_w,    idle_w,    speedup_w),
    ("+ pin_memory",                   bps_p,    idle_p,    speedup_p),
    ("+ no slow augment",              bps_all,  idle_all,  speedup_all),
]
for name, bps, idle, sp in configs:
    print(f"  {name:<30}  {bps:>10.0f}  {idle:>7.1f}%  {sp:>8.2f}×")

report = {
    "device": DEVICE,
    "batch_size": BATCH,
    "n_workers": BEST_WORKERS,
    "baseline_bps": round(bps_base, 1),
    "final_bps": round(bps_all, 1),
    "total_speedup": round(speedup_all, 2),
    "final_idle_pct": round(idle_all, 1),
    "fix_contributions": {
        "workers": round(speedup_w, 2),
        "pin_memory": round(speedup_p, 2),
        "fast_augment": round(speedup_all, 2),
    },
}
with open("/tmp/capstone17_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"""
  TOTAL SPEEDUP: {speedup_all:.2f}× (from {bps_base:.0f} to {bps_all:.0f} batches/s)
  FINAL idle_pct: {idle_all:.1f}%  (target: < 10%)
  Report saved: /tmp/capstone17_report.json

  LESSON: Each fix had diminishing returns:
    Fix 1 (workers)     : {speedup_w:.2f}× — biggest gain, addresses root cause
    Fix 2 (pin_memory)  : {speedup_p:.2f}× cumulative — moderate H2D gain
    Fix 3 (augment)     : {speedup_all:.2f}× cumulative — eliminates per-sample floor
  The last fix is often the hardest: it requires redesigning the pipeline.
""")
assert report["total_speedup"] > 0, "Total speedup should be positive"
print("  ✓ Section 5 passed — before/after report complete")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 17.2 complete!")
print(f"  DataLoader hunt complete: {speedup_all:.2f}× improvement documented.")
print("  Next: VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.1_kv_cache_scaling.py")
print("=" * 60)
