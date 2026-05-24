#!/usr/bin/env python3
"""
cache_miss_analysis.py  ─  Phase 3 / Module 6: CPU Cache Miss Patterns
=======================================================================

HOW TO RUN
    python cache_miss_analysis.py
    python cache_miss_analysis.py --exp sequential   # just one pattern


"""

import argparse, time, random
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
    choices=["all","sequential","random","stride","matrix","dataloader"])
args = parser.parse_args()

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

def wall_bw(fn, size_bytes, warmup=3, iters=10):
    for _ in range(warmup): fn()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    elapsed = (time.perf_counter() - t0) / iters
    return size_bytes / elapsed / 1e9  # GB/s

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 – Sequential vs Random Access
# ─────────────────────────────────────────────────────────────────────────────
def exp_sequential():
    sep("EXP 1 – Sequential vs Random Access Pattern")
    print("""
  Sequential: data[0], data[1], data[2], …
    CPU prefetcher detects the pattern and pre-loads cache lines → fast

  Random: data[rand()], data[rand()], …
    CPU can't predict → every access may miss cache → slow

  Ratio between them = "miss penalty" for your access pattern.
    """)
    n = 64 * 1024 * 1024    # 64M ints = 256MB (bigger than L3)
    arr = np.arange(n, dtype=np.int32)

    # Sequential scan
    bw_seq = wall_bw(lambda: arr.sum(), n * 4)

    # Random access (gather)
    idx_random = np.random.randint(0, n, size=n // 16, dtype=np.int64)
    bw_rand = wall_bw(lambda: arr[idx_random].sum(), len(idx_random) * 4)

    print(f"  Array size         : {n*4/1e6:.0f} MB  (> L3 cache)")
    print(f"  Sequential scan    : {bw_seq:.1f} GB/s  ← prefetcher active")
    print(f"  Random gather      : {bw_rand:.1f} GB/s  ← every access misses")
    print(f"  Miss penalty ratio : {bw_seq/bw_rand:.1f}×")
    print(f"\n  For DataLoader: reading images by sequential index = fast")
    print(f"  Shuffled dataset + large dataset = many cache misses per epoch")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 – Stride access pattern
# ─────────────────────────────────────────────────────────────────────────────
def exp_stride():
    sep("EXP 2 – Stride Access and Cache Line Waste")
    print("""
  CPU loads data in 64-byte CACHE LINES.
  With stride-1: every byte in the cache line is used → efficient.
  With stride-64 (floats): 1 out of 16 floats per cache line is used.
    Effective bandwidth drops 16× even though you read the same data.

  Relevance: transposed matrix multiply, wrong tensor layout (NHWC vs NCHW)
    """)
    n = 32 * 1024 * 1024    # 128MB
    arr = np.random.rand(n).astype(np.float32)

    print(f"  {'Stride':>8}  {'BW (GB/s)':>12}  {'Cache eff%':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}")

    baseline_bw = None
    for stride in [1, 2, 4, 8, 16, 32, 64]:
        idx = np.arange(0, n, stride, dtype=np.int64)
        bw  = wall_bw(lambda: arr[idx].sum(), len(idx) * 4)
        if baseline_bw is None: baseline_bw = bw
        eff = bw / baseline_bw * 100
        print(f"  {stride:>8}  {bw:>12.1f}  {eff:>11.0f}%")

    print(f"\n  Stride-16 (float) = one float per cache line = {baseline_bw/16:.1f} GB/s effective")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 – Matrix transposition cache miss comparison
# ─────────────────────────────────────────────────────────────────────────────
def exp_matrix():
    sep("EXP 3 – Matrix Layout: Row-Major vs Column-Major Access")
    print("""
  C/NumPy/PyTorch store matrices in ROW-MAJOR order (C contiguous):
    data[row][col] = data[row * ncols + col]
    Reading across a row = sequential → cache friendly
    Reading down a column = stride = ncols → cache unfriendly

  For matrix multiply A @ B:
    A accessed row-by-row  → sequential → fast
    B accessed col-by-col  → stride     → slow
  Solution: transpose B before the loop (or use cuBLAS which handles this).
    """)
    N = 4096   # NxN matrix
    A = np.random.rand(N, N).astype(np.float32)

    # Row access (cache-friendly)
    bw_row = wall_bw(lambda: A.sum(axis=1), N * N * 4)

    # Column access (cache-unfriendly: stride=N floats between elements)
    bw_col = wall_bw(lambda: A.sum(axis=0), N * N * 4)

    # Transposed (make column access sequential)
    At = np.ascontiguousarray(A.T)
    bw_transp = wall_bw(lambda: At.sum(axis=1), N * N * 4)

    print(f"  {N}×{N} float32 matrix ({N*N*4/1e6:.0f} MB)")
    print(f"  Row-major sum (axis=1)       : {bw_row:.1f} GB/s  ← cache friendly")
    print(f"  Column-major sum (axis=0)    : {bw_col:.1f} GB/s  ← cache unfriendly")
    print(f"  Transposed then row sum      : {bw_transp:.1f} GB/s  ← transposed = friendly")
    del A, At

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4 – DataLoader cache competition between workers
# ─────────────────────────────────────────────────────────────────────────────
def exp_dataloader():
    sep("EXP 4 – DataLoader Worker Cache Competition")
    print("""
  With 4 DataLoader workers on a 4-core CPU:
    Each worker loads a different image from a different location.
    All 4 workers compete for the shared L3 cache.
    Each worker's hot data evicts the others' data → cache thrashing.

  With 1 worker (sequential):
    Data loaded in order, prefetcher helps, no competition.
    But: GPU waits idle while loading.

  OPTIMAL: num_workers ≈ 2–4 (balance: enough parallelism, not too much thrash)
  PLUS: Move augmentation to GPU so workers only do I/O, not CPU compute.
    """)
    import torch
    from torch.utils.data import Dataset, DataLoader

    class CachePressureDataset(Dataset):
        """Simulates loading images from a large randomly-accessed array."""
        def __init__(self, n_images, image_size):
            # 100MB buffer: bigger than L3 → every worker access is a miss
            self.buf = torch.rand(25 * 1024 * 1024)   # 100MB
            self.n   = n_images
            self.img_size = image_size

        def __len__(self): return self.n

        def __getitem__(self, i):
            # Simulate random image read (cache miss pattern)
            offset = (i * self.img_size * self.img_size * 3) % (len(self.buf) - 1000)
            _ = self.buf[offset:offset+1000].sum()   # force cache miss
            return torch.rand(3, self.img_size, self.img_size), i % 1000

    ds = CachePressureDataset(1000, 64)

    print(f"  {'Workers':>8}  {'Batches/s':>12}  {'Samples/s':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*12}")

    for nw in [0, 1, 2, 4]:
        try:
            dl = DataLoader(ds, batch_size=16, num_workers=nw,
                            prefetch_factor=2 if nw > 0 else None)
            t0 = time.perf_counter()
            for i, _ in enumerate(dl):
                if i >= 20: break
            elapsed = time.perf_counter() - t0
            bps = 20 / elapsed
            sps = 20 * 16 / elapsed
            print(f"  {nw:>8}  {bps:>12.1f}  {sps:>12.1f}")
        except Exception as e:
            print(f"  {nw:>8}  error: {e}")

    print(f"""
  perf command to measure during DataLoader run:
    sudo perf stat -e cache-misses,cache-references \\
        python cache_miss_analysis.py --exp dataloader
  """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  cache_miss_analysis.py  ─  Phase 3 Module 6")
    print(f"{'='*60}")
    print("""
  Measuring cache effects requires large arrays (> L3 size ~ 8-32 MB).
  Results vary by CPU model.  Run with perf for hardware counter truth.
  """)
    d = {
        "sequential": exp_sequential,
        "stride":     exp_stride,
        "matrix":     exp_matrix,
        "dataloader": exp_dataloader,
    }
    if args.exp == "all":
        for fn in d.values(): fn()
    else:
        d[args.exp]()
