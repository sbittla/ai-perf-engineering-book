#!/usr/bin/env python3
"""
lock_contention.py  ─  Phase 3 / Module 6: Simulate & Detect Lock Contention
=============================================================================

HOW TO RUN
    python lock_contention.py                    # run all experiments
    python lock_contention.py --threads 8        # custom thread count

DIAGNOSE WITH LINUX TOOLS WHILE THIS RUNS
    # Count futex (lock) syscalls:
    strace -c -e trace=futex python lock_contention.py

    # See scheduler context switches:
    vmstat 1 10

    # CPU flamegraph (look for pthread_mutex_lock wide bars):
    py-spy record -o lock_flame.svg -- python lock_contention.py


"""

import argparse, threading, time, queue, os
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--threads", type=int, default=4)
args = parser.parse_args()

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 – Python threading: GIL contention
# ─────────────────────────────────────────────────────────────────────────────
sep("EXP 1 – Python GIL Contention")
print(f"""
  The Python GIL (Global Interpreter Lock) allows only ONE thread to
  execute Python bytecode at a time.  Multiple threads appear to run
  in parallel but actually take turns.

  For CPU-bound Python code:
    1 thread vs {args.threads} threads → same or SLOWER throughput
    Reason: threads compete for GIL → context switch overhead

  For I/O-bound Python code:
    Threads release the GIL during I/O → true concurrency

  Implication for DataLoader:
    Workers do file I/O (release GIL) + augmentation (hold GIL)
    Heavy augmentation → GIL contention → workers bottleneck each other
    Fix: move augmentation to C extensions (torchvision) or GPU
""")

COUNTER = 0
LOCK    = threading.Lock()
N_ITERS = 200_000

def increment_with_lock(n):
    global COUNTER
    for _ in range(n):
        with LOCK:           # acquire GIL + mutex
            COUNTER += 1

def increment_no_lock(n):
    # No lock: demonstrates race condition (incorrect result, but faster)
    global COUNTER
    for _ in range(n):
        COUNTER += 1

# Single-threaded baseline
COUNTER = 0
t0 = time.perf_counter()
increment_with_lock(N_ITERS)
t_single = time.perf_counter() - t0
print(f"  Single thread (locked)   : {t_single*1000:.1f} ms  counter={COUNTER}")

# Multi-threaded with lock (contention)
COUNTER = 0
threads = [threading.Thread(target=increment_with_lock,
                            args=(N_ITERS // args.threads,))
           for _ in range(args.threads)]
t0 = time.perf_counter()
for t in threads: t.start()
for t in threads: t.join()
t_multi = time.perf_counter() - t0
print(f"  {args.threads} threads  (locked)   : {t_multi*1000:.1f} ms  counter={COUNTER}  "
      f"({t_single/t_multi:.2f}× vs single)")

print(f"""
  Multi-threaded with lock is {'slower' if t_multi > t_single else 'faster'} due to:
    - Lock acquisition overhead (kernel futex syscall each time)
    - Thread context switches (OS schedules threads on/off CPU)
    - False sharing: threads on different cores invalidate each other's cache

  DIAGNOSE: strace -c python lock_contention.py
  Look for: futex   calls in the thousands
""")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 – DataLoader worker GIL contention simulation
# ─────────────────────────────────────────────────────────────────────────────
sep("EXP 2 – DataLoader Worker Throughput: CPU vs GPU Augmentation")
print("""
  DataLoader workers use multiprocessing (bypass GIL), but the main
  thread still competes for CPU resources during collation.

  CPU augmentation (transforms.v1):
    Worker: load image → CPU augment (holds CPU) → push to queue
    Main thread must wait for workers to finish each batch

  GPU augmentation (transforms.v2 on CUDA):
    Worker: load image → push raw tensor to queue (fast)
    Main thread: GPU augment on batch (overlapped with next worker load)

  We measure batch delivery throughput in both modes.
""")
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T1
import torchvision.transforms.v2 as T2

class FakeDataset(Dataset):
    def __init__(self, size, augment_fn):
        self.size = size
        self.augment_fn = augment_fn
    def __len__(self): return self.size
    def __getitem__(self, i):
        img = torch.rand(3, 256, 256)
        return self.augment_fn(img), i % 1000

# CPU augmentation (heavy)
cpu_aug = T1.Compose([
    T1.ToPILImage(),
    T1.RandomResizedCrop(224),
    T1.RandomHorizontalFlip(),
    T1.ColorJitter(0.3, 0.3, 0.3),
    T1.ToTensor(),
    T1.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
# Minimal CPU transform (just resize to tensor)
minimal_aug = T1.Compose([T1.Resize(224), T1.ToTensor()])
# Fake: skip PIL conversion for speed comparison
fast_aug = lambda x: x

BATCH   = 16
ITERS   = 20
WORKERS = 2

for label, aug_fn in [("CPU-heavy aug", cpu_aug),
                       ("minimal CPU aug", minimal_aug),
                       ("raw tensor (no aug)", fast_aug)]:
    ds = FakeDataset(BATCH * ITERS * 2, aug_fn)
    try:
        dl = DataLoader(ds, batch_size=BATCH, num_workers=WORKERS,
                        pin_memory=True, prefetch_factor=2)
        t0 = time.perf_counter()
        for i, (imgs, _) in enumerate(dl):
            if i >= ITERS: break
        elapsed = time.perf_counter() - t0
        tput = BATCH * ITERS / elapsed
        print(f"  {label:<24}: {tput:.0f} samples/sec  ({elapsed:.2f}s for {ITERS} batches)")
    except Exception as e:
        print(f"  {label:<24}: skipped ({e})")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 – PyTorch allocator lock contention (multiple CUDA streams)
# ─────────────────────────────────────────────────────────────────────────────
sep("EXP 3 – CUDA Allocator Contention (Multiple Streams)")
print("""
  PyTorch's caching memory allocator uses a mutex to protect its internal
  free-block lists.  When many CUDA streams allocate/free simultaneously,
  they compete for this lock.

  Signs of allocator contention:
    • strace shows many futex() syscalls from PyTorch C++ threads
    • nvtx timeline shows "cudaMalloc" spikes between kernels
    • Setting PYTORCH_NO_CUDA_MEMORY_CACHING=1 makes it much worse

  Fix: increase block size or use persistent buffers to reduce allocations.
""")
if torch.cuda.is_available():
    DEVICE = "cuda"
    N_STREAMS = 4
    streams   = [torch.cuda.Stream() for _ in range(N_STREAMS)]
    SIZE      = 10 * 1024 * 1024   # 10M floats = 40MB per allocation

    # Without pre-allocation: allocator called frequently
    t0 = time.perf_counter()
    for _ in range(20):
        tensors = []
        for s in streams:
            with torch.cuda.stream(s):
                tensors.append(torch.rand(SIZE, device=DEVICE))   # allocates
        torch.cuda.synchronize()
        del tensors   # triggers free → allocator lock again
    t_alloc = time.perf_counter() - t0

    # With pre-allocation: buffers reused, no allocator calls
    bufs = [torch.empty(SIZE, device=DEVICE) for _ in streams]
    t0   = time.perf_counter()
    for _ in range(20):
        for s, buf in zip(streams, bufs):
            with torch.cuda.stream(s):
                buf.normal_()   # reuse existing allocation
        torch.cuda.synchronize()
    t_prealloc = time.perf_counter() - t0

    print(f"  Repeated alloc/free  : {t_alloc*1000:.1f} ms")
    print(f"  Pre-allocated buffers: {t_prealloc*1000:.1f} ms  ({t_alloc/t_prealloc:.2f}× faster)")
    print(f"\n  DETECT: PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python lock_contention.py")
    del bufs
else:
    print("  (Skipped – CUDA not available)")

print(f"""
  SUMMARY – How to diagnose lock contention:
    strace -c python script.py              # count syscall types (futex = locks)
    strace -e trace=futex -c python ...     # only futex calls
    perf stat -e context-switches python .. # OS context switch count
    py-spy record ... | grep -i lock        # flamegraph for lock frames
""")
