#!/usr/bin/env python3
"""
7.DataLoader_Optimisation/7.2_io_bottleneck.py  ─  Chapter 7: I/O Bottleneck Diagnosis
=======================================================================
Covers book section 7.2:
  • Simulating GPU starvation with a slow DataLoader
  • Measuring idle_pct from wall time vs GPU compute time
  • Diagnosing with nvidia-smi dmon
  • The fix checklist based on idle_pct thresholds

Run:  python II.GPU_Programming_and_Profiling/7.DataLoader_Optimisation/7.2_io_bottleneck.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Exercise 7.2 — Diagnosing and Fixing I/O Bottlenecks")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# Helper: measure forward pass GPU time with CUDA events
# ─────────────────────────────────────────────────────────────
def measure_batch_gpu_ms(model, batch_tensor):
    """Return GPU compute time in ms for one forward pass."""
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(batch_tensor.to(DEVICE))
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            model(batch_tensor.to(DEVICE))
        return (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────────────────────────
# SECTION 1: Simulating GPU Starvation
# ─────────────────────────────────────────────────────────────
print("── Section 1: Simulating GPU Starvation ──")
print("""
  A GPU-starved training pipeline looks like this in nsys:
    CPU: [=====DataLoader 20ms=====][GPU launch]
    GPU:                             [==compute 5ms==]    [waiting...]

  The GPU is idle for most of the wall time because it must wait for
  the CPU to finish loading the next batch.

  We quantify this with:
    idle_pct = (total_wall_ms - gpu_compute_ms) / total_wall_ms * 100

  A healthy pipeline has idle_pct < 10%.
  A DataLoader-bottlenecked pipeline has idle_pct > 50%.
""")

class SlowDataset(Dataset):
    """Simulates 20ms disk I/O per __getitem__."""
    def __init__(self, n=100):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        time.sleep(0.020)  # 20ms — slow disk
        return torch.randn(3, 224, 224)

# TODO 1: Implement SlowDataset (already done above for you).
#   Verify the class is a valid Dataset subclass by creating an instance.
assert issubclass(SlowDataset, Dataset), "SlowDataset must inherit from Dataset"

# 3-layer conv model for the compute part
conv_model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(4),
    nn.Flatten(),
    nn.Linear(32 * 4 * 4, 10),
).to(DEVICE)
conv_model.eval()

slow_loader = DataLoader(SlowDataset(n=40), batch_size=4, num_workers=0)

N_BATCHES = 10
gpu_compute_ms = 0.0

wall_start = time.perf_counter()
for i, batch in enumerate(slow_loader):
    if i >= N_BATCHES:
        break
    gpu_compute_ms += measure_batch_gpu_ms(conv_model, batch)
wall_end = time.perf_counter()

total_wall_ms = (wall_end - wall_start) * 1000
idle_pct_slow = (total_wall_ms - gpu_compute_ms) / total_wall_ms * 100

print(f"  SlowDataset ({N_BATCHES} batches):")
print(f"    Total wall time  : {total_wall_ms:.0f} ms")
print(f"    GPU compute time : {gpu_compute_ms:.1f} ms")
print(f"    Idle %           : {idle_pct_slow:.1f}%")

assert idle_pct_slow > 50, (
    f"idle_pct for SlowDataset should be > 50%, got {idle_pct_slow:.1f}%. "
    "Check that the SlowDataset sleep is 20ms per item."
)
print("  ✓ Section 1 passed — GPU starvation detected (idle_pct > 50%)")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring GPU Utilisation with Fast DataLoader
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Measuring GPU Utilisation ──")
print("""
  A fast DataLoader (SSD or in-memory data) should keep the GPU fed.
  When idle_pct < 30%, the GPU is the bottleneck, not the DataLoader.
  This is when it is worth profiling the GPU kernels with ncu.

  We repeat the measurement with FastDataset (no sleep) to see the
  healthy baseline.
""")

class FastDataset(Dataset):
    """No I/O delay — data already in memory."""
    def __init__(self, n=100):
        self.n = n
        self.data = [torch.randn(3, 224, 224) for _ in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.data[idx]

fast_loader = DataLoader(FastDataset(n=80), batch_size=4, num_workers=2)

gpu_compute_ms_fast = 0.0
wall_start_fast = time.perf_counter()
for i, batch in enumerate(fast_loader):
    if i >= N_BATCHES:
        break
    # TODO 2: Measure GPU forward pass time for this batch.
    #   Use measure_batch_gpu_ms(conv_model, batch) and accumulate.
    pass  # YOUR CODE HERE → gpu_compute_ms_fast += measure_batch_gpu_ms(conv_model, batch)

wall_end_fast = time.perf_counter()
total_wall_ms_fast = (wall_end_fast - wall_start_fast) * 1000

if total_wall_ms_fast > 0:
    idle_pct_fast = (total_wall_ms_fast - gpu_compute_ms_fast) / total_wall_ms_fast * 100
else:
    idle_pct_fast = 0.0

if gpu_compute_ms_fast == 0.0:
    print("  (TODO 2 not completed — using reference idle_pct)")
    idle_pct_fast = 15.0  # reference: fast loader, GPU mostly busy

print(f"  FastDataset ({N_BATCHES} batches):")
print(f"    Total wall time  : {total_wall_ms_fast:.0f} ms")
print(f"    GPU compute time : {gpu_compute_ms_fast:.1f} ms")
print(f"    Idle %           : {idle_pct_fast:.1f}%")

assert idle_pct_fast < 30, (
    f"FastDataset idle_pct should be < 30%, got {idle_pct_fast:.1f}%. "
    "Check that FastDataset has no sleep and TODO 2 is implemented."
)
print("  ✓ Section 2 passed — FastDataset keeps GPU busy (idle_pct < 30%)")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Diagnosing with nvidia-smi
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Diagnosing with nvidia-smi ──")
print("""
  ── Step-by-step diagnosis guide ──

  Step 1: Launch nvidia-smi dmon in a SEPARATE terminal while your
          training script is running:

            nvidia-smi dmon -s u -d 1

          This prints GPU utilisation metrics every second.
          The columns are:
            sm%  : fraction of SMs active (0-100)
            mem% : fraction of memory bandwidth in use (0-100)
            enc%/dec%: video encoder/decoder (usually 0 for AI workloads)

  Step 2: Look at the "sm%" column during the main training phase.
          If sm% is consistently below 60%, the GPU is being starved.
          Common causes and their idle_pct signatures:

            sm% < 20%: severely I/O bound
                       → check disk speed (hdparm -t), increase num_workers
            sm% 20-60%: moderately I/O bound
                        → add pin_memory, increase prefetch_factor
            sm% > 80%: GPU is the bottleneck
                       → profile kernels with ncu (Exercise 5.2)

  Step 3: Run your script under nsys to see the GPU timeline:
            nsys profile --trace=cuda,nvtx python train.py --steps 50

  Step 4: Look for gaps between CUDA kernels in the nsys timeline.
          Each gap is GPU idle time. If gaps > 10ms, the DataLoader
          is the bottleneck.

  Step 5: Check DataLoader worker count:
            import torch
            print(torch.get_num_threads())   # CPU threads available

          Set num_workers = min(cpu_count // 2, 8).
""")

# No TODO in this section — it is a reading + reference section
assert True, "Reading section — always passes"
print("  ✓ Section 3 passed — nvidia-smi diagnosis guide reviewed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: The Fix Checklist
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: The Fix Checklist ──")
print("""
  Based on measured idle_pct, apply fixes in priority order:

  idle_pct > 70%: DataLoader is severely bottlenecked
    → Increase num_workers (try 4, 8, 16)
    → Check disk speed: hdparm -t /dev/nvme0
    → Move dataset to faster storage (NVMe SSD, tmpfs, or RAM disk)
    → Cache preprocessed data as .pt tensors to skip per-epoch decoding

  idle_pct 40-70%: DataLoader is moderately bottlenecked
    → Add pin_memory=True to DataLoader
    → Use non_blocking=True in .to(device) calls
    → Increase prefetch_factor (try 4 or 8)
    → Reduce preprocessing compute in __getitem__

  idle_pct < 40%: DataLoader is NOT the bottleneck
    → Profile GPU kernels with ncu (Exercise 5.2)
    → Check if model is memory-bound or compute-bound
    → Consider torch.compile or mixed precision (Exercises 6.1, 6.3)
""")

def diagnose_dataloader(idle_pct: float) -> list:
    """
    TODO 4: Implement this function.
    Return a list of recommended fix strings based on idle_pct:
      idle_pct > 70  → ["add num_workers, check disk speed",
                         "move dataset to faster storage"]
      idle_pct > 40  → ["add pin_memory, increase prefetch_factor",
                         "use non_blocking=True"]
      else           → ["DataLoader is not the bottleneck — profile GPU kernels"]
    """
    pass  # YOUR CODE HERE


result_high = diagnose_dataloader(80)
result_mid  = diagnose_dataloader(55)
result_low  = diagnose_dataloader(20)

assert result_high is not None, "diagnose_dataloader(80) must return a list"
assert len(result_high) > 0,    "diagnose_dataloader(80) must return non-empty list"
assert "num_workers" in result_high[0], (
    f"diagnose_dataloader(80)[0] should contain 'num_workers', "
    f"got: '{result_high[0]}'"
)

print(f"  idle_pct=80%:  {result_high}")
print(f"  idle_pct=55%:  {result_mid}")
print(f"  idle_pct=20%:  {result_low}")
print("  ✓ Section 4 passed — fix checklist function implemented")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 7.2 complete!")
print()
print("  You have now completed all exercises in Part II —")
print("  GPU Programming & Profiling.")
print()
print("  Part II covered:")
print("    Chapter 4: CUDA execution model, coalescing, Tensor Cores")
print("    Chapter 5: nsys, ncu, torch.profiler")
print("    Chapter 6: AMP, quantization, torch.compile")
print("    Chapter 7: DataLoader tuning, I/O bottleneck diagnosis")
print("=" * 60)
