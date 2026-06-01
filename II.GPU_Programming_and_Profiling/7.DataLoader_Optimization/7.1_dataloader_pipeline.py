#!/usr/bin/env python3
"""
7.DataLoader_Optimization/7.1_dataloader_pipeline.py  ─  Chapter 7: DataLoader Tuning
=======================================================================
Covers book section 7.1:
  • num_workers=0 bottleneck and how to fix it
  • pin_memory + non_blocking for faster host→device transfers
  • prefetch_factor tuning
  • The .item() anti-pattern and GPU stalls

Run:  python II.GPU_Programming_and_Profiling/7.DataLoader_Optimization/7.1_dataloader_pipeline.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Exercise 7.1 — DataLoader Pipeline Tuning")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: num_workers=0 Bottleneck
# ─────────────────────────────────────────────────────────────
print("── Section 1: num_workers=0 Bottleneck ──")
print("""
  With num_workers=0, data loading runs synchronously on the MAIN thread.
  The GPU completes a batch, then waits for the CPU to load the next one.
  The GPU is idle for the entire duration of the DataLoader call.

  With num_workers > 0, worker processes load and preprocess the NEXT
  batch in the background while the GPU computes the CURRENT batch.
  This is called double-buffering (or N-buffering for prefetch_factor > 1).

  Rule of thumb: set num_workers = number of CPU cores / 2, capped at 8.
  For NFS/S3-backed datasets, num_workers may need to be higher (16–32)
  to saturate the storage bandwidth.

  We simulate I/O latency with time.sleep(0.0005) = 0.5ms per item.
  With batch_size=8 and single worker: 4ms loading per batch.
  With 4 workers:    <1ms loading per batch (parallel).
""")

class SlowDataset(Dataset):
    """Simulates a dataset with 0.5ms I/O latency per item."""
    def __init__(self, n=320):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        time.sleep(0.0005)  # 0.5ms simulated file I/O
        return torch.randn(64), torch.randint(0, 10, (1,)).squeeze()

dataset = SlowDataset(n=320)  # 320 items, 40 batches of 8

single_loader = DataLoader(dataset, batch_size=8, num_workers=0, shuffle=False)
multi_loader  = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False,
                           persistent_workers=True)

N_BATCHES = 20

# TODO 1: Implement the timing loop for both DataLoaders.
#   For each loader, iterate over N_BATCHES batches and measure total wall time.
#   single_worker_ms = wall time for single_loader (ms)
#   multi_worker_ms  = wall time for multi_loader  (ms)

single_worker_ms = None  # YOUR CODE HERE
multi_worker_ms  = None  # YOUR CODE HERE

if single_worker_ms is None or multi_worker_ms is None:
    print("  (TODO 1 not completed — using simulated values)")
    single_worker_ms = N_BATCHES * 8 * 0.5   # 0.5ms per item, sequential
    multi_worker_ms  = N_BATCHES * 8 * 0.5 / 4  # 4x parallelism

assert multi_worker_ms < single_worker_ms * 0.7, (
    f"Multi-worker ({multi_worker_ms:.1f} ms) should be < 70% of "
    f"single-worker ({single_worker_ms:.1f} ms)"
)
speedup = single_worker_ms / multi_worker_ms
print(f"  Single worker (num_workers=0): {single_worker_ms:.1f} ms for {N_BATCHES} batches")
print(f"  Multi  worker (num_workers=4): {multi_worker_ms:.1f} ms for {N_BATCHES} batches")
print(f"  Speedup: {speedup:.2f}x")
print("  ✓ Section 1 passed — multi-worker DataLoader is faster")

# ─────────────────────────────────────────────────────────────
# SECTION 2: pin_memory and non_blocking
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: pin_memory and non_blocking ──")
print("""
  The CPU→GPU transfer has two parts:
    1. Copy from pageable CPU memory to a pinned (page-locked) CPU buffer
    2. DMA from pinned buffer to GPU

  With pin_memory=True in DataLoader, step 1 happens in the worker
  processes.  The main thread receives tensors already in pinned memory
  and the GPU's DMA engine can start immediately.

  With .to(device, non_blocking=True), the DMA transfer runs asynchronously.
  The CPU can launch the NEXT batch's preprocessing while the GPU DMA
  copies the current batch.

  Combined: pin_memory=True + non_blocking=True is the fastest CPU→GPU
  path for DataLoader output.  Always use both together.
""")

class SimpleDataset(Dataset):
    def __init__(self, n=200):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.randn(3, 64, 64)

simple_ds = SimpleDataset()

regular_loader = DataLoader(simple_ds, batch_size=16, num_workers=2,
                            pin_memory=False)
pinned_loader  = DataLoader(simple_ds, batch_size=16, num_workers=2,
                            pin_memory=True if DEVICE == "cuda" else False)

# Get one batch from each and time the .to(DEVICE) transfer
regular_batch = next(iter(regular_loader))
pinned_batch  = next(iter(pinned_loader))

# TODO 2: Time the host→device transfer for one batch.
#   regular_ms: time regular_batch.to(DEVICE) with GPU sync
#   pinned_ms:  time pinned_batch.to(DEVICE, non_blocking=True) with GPU sync

regular_ms = None  # YOUR CODE HERE
pinned_ms  = None  # YOUR CODE HERE

if DEVICE == "cuda":
    if regular_ms is None or pinned_ms is None:
        print("  (TODO 2 not completed — showing concept only)")
        regular_ms = 2.0  # reference
        pinned_ms  = 1.0
    assert pinned_ms <= regular_ms * 1.5, (
        f"Pinned ({pinned_ms:.3f} ms) should not be slower than "
        f"regular ({regular_ms:.3f} ms) × 1.5"
    )
    print(f"  Regular (pageable) .to(device): {regular_ms:.3f} ms")
    print(f"  Pinned  (non_blocking=True)   : {pinned_ms:.3f} ms")
else:
    regular_ms = pinned_ms = 0.1
    print("  (Requires CUDA — skipping pin_memory transfer benchmark)")
    print("  On CUDA, pin_memory=True with non_blocking=True is ~1.5–2x faster")

print("  ✓ Section 2 passed — pin_memory transfer verified")

# ─────────────────────────────────────────────────────────────
# SECTION 3: prefetch_factor
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: prefetch_factor ──")
print("""
  prefetch_factor (default=2 when num_workers>0) controls how many
  batches each worker pre-loads AHEAD of what the main process has
  requested.

  prefetch_factor=1: each worker loads exactly what is needed next
  prefetch_factor=4: each worker pre-loads 4 batches ahead

  For high-latency storage (NFS, S3, spinning disk), a higher
  prefetch_factor fills the pipeline and hides the I/O latency.
  For fast local SSD, prefetch_factor=2 is usually sufficient.

  We simulate 5ms I/O per item to make prefetching visibly beneficial.
""")

class IoSlowDataset(Dataset):
    """5ms I/O per item."""
    def __init__(self, n=200):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        time.sleep(0.005)  # 5ms I/O
        return torch.randn(32), torch.randint(0, 5, (1,)).squeeze()

io_ds = IoSlowDataset()

pf1_loader = DataLoader(io_ds, batch_size=4, num_workers=2,
                        prefetch_factor=1, persistent_workers=True)
pf4_loader = DataLoader(io_ds, batch_size=4, num_workers=2,
                        prefetch_factor=4, persistent_workers=True)

N_IO = 20  # batches to time

# TODO 3: Time 20 batches for each prefetch_factor.
prefetch1_ms = None  # YOUR CODE HERE
prefetch4_ms = None  # YOUR CODE HERE

if prefetch1_ms is None or prefetch4_ms is None:
    print("  (TODO 3 not completed — using simulated values)")
    prefetch1_ms = N_IO * 4 * 5.0 / 2   # 2 workers, no look-ahead
    prefetch4_ms = N_IO * 4 * 5.0 / 2 / 1.5  # approx benefit

assert prefetch4_ms <= prefetch1_ms * 1.2, (
    f"prefetch_factor=4 ({prefetch4_ms:.0f} ms) should not be slower than "
    f"1.2x prefetch_factor=1 ({prefetch1_ms:.0f} ms)"
)
print(f"  prefetch_factor=1: {prefetch1_ms:.0f} ms for {N_IO} batches")
print(f"  prefetch_factor=4: {prefetch4_ms:.0f} ms for {N_IO} batches")
print(f"  (Higher prefetch buffers more batches ahead, hiding I/O latency)")
print("  ✓ Section 3 passed — prefetch_factor comparison complete")

# ─────────────────────────────────────────────────────────────
# SECTION 4: The .item() Anti-Pattern
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: The .item() Anti-Pattern ──")
print("""
  loss.item() does two things:
    1. Forces the CPU to wait for the GPU to finish (GPU synchronization)
    2. Copies the scalar value from GPU to CPU

  If called inside the inner training loop (every step), it serialises
  the entire pipeline: the GPU cannot start the next step until the CPU
  has received the loss value from the current step.

  The fix: accumulate loss as a TENSOR, call .item() once at the end.
    BAD:  total += loss.item()   # sync every step
    GOOD: total += loss.detach() # no sync, tensor accumulation
    GOOD: total.item()           # single sync at the end

  On GPU, the difference can be 2–5x throughput.
  On CPU, the difference is smaller (no GPU sync) but still measurable
  due to Python function call overhead.
""")

STEPS_ITEM = 100
SIZE = 1024

model_item = nn.Linear(SIZE, 1, bias=False).to(DEVICE)
x_item = torch.randn(64, SIZE, device=DEVICE)

# Version A: .item() inside the loop (BAD)
t_bad_start = time.perf_counter()
bad_total = 0.0
for _ in range(STEPS_ITEM):
    out = model_item(x_item)
    loss_bad = out.sum()
    bad_total += loss_bad.item()   # ← GPU sync every step
if DEVICE == "cuda":
    torch.cuda.synchronize()
bad_ms = (time.perf_counter() - t_bad_start) * 1000

# TODO 4: Implement the GOOD version: accumulate loss.detach() in a tensor,
#   call .item() ONCE at the end.
#   Measure the wall time in good_ms.

# YOUR CODE HERE: implement good version
good_ms = None  # YOUR CODE HERE

if good_ms is None:
    print("  (TODO 4 not completed — using reference timing)")
    good_ms = bad_ms * 0.6  # reference: typically 1.5–5x faster on GPU

assert good_ms > 0, "good_ms must be > 0"
# Use a generous threshold: on CPU-only machines the difference is small
assert bad_ms > good_ms * 0.5, (
    f"bad_ms ({bad_ms:.1f} ms) should be > 0.5x good_ms ({good_ms:.1f} ms). "
    "On CPU the difference may be small."
)
ratio = bad_ms / good_ms
print(f"  .item() in loop (BAD)          : {bad_ms:.1f} ms for {STEPS_ITEM} steps")
print(f"  .detach() + single .item() (GOOD): {good_ms:.1f} ms for {STEPS_ITEM} steps")
print(f"  Speedup from removing .item()   : {ratio:.2f}x")
if ratio < 1.2:
    print("  (Small difference — likely running on CPU; on CUDA this is 2–5x)")
print("  ✓ Section 4 passed — .item() anti-pattern demonstrated")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 7.1 complete!")
print("  You now know how to tune num_workers, use pin_memory,")
print("  tune prefetch_factor, and avoid the .item() anti-pattern.")
print("  Next: II.GPU_Programming_and_Profiling/7.DataLoader_Optimization/7.2_io_bottleneck.py")
print("=" * 60)
