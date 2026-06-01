#!/usr/bin/env python3
"""
I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py  ─  Chapter 2: GPU Timing & Profiling Basics
=======================================================================
Covers book section 2.5:
  • Why time.time() is wrong for GPU code (async execution)
  • CUDA events — the correct timing method
  • Warmup runs — mandatory before any benchmark
  • A reusable benchmark() helper
  • torch.profiler — finding slow operators
  • GPU memory tracking (allocated / reserved / peak)
  • NVTX markers for Nsight Systems

Run:  python I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py
      nsys profile --trace=cuda,nvtx python I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py   (for Section 6)
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

print("=" * 60)
print("  Exercise 05 — GPU Timing & Profiling Basics")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: The wrong way vs the right way to time GPU code
# ─────────────────────────────────────────────────────────────
print("── Section 1: Timing GPU Operations Correctly ──")
print("""
  GPU operations are ASYNCHRONOUS.  The CPU submits work to a queue
  and returns immediately — the GPU may not have started yet.

  time.time() after a GPU op measures CPU submission latency, NOT
  GPU execution time.  Use CUDA Events: they record timestamps
  directly on the GPU timeline, bypassing CPU overhead entirely.
""")

if DEVICE == "cuda":
    N = 1024
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float16)

    # WRONG: CPU timer with no sync
    t0 = time.time()
    C  = torch.mm(A, B)
    t_wrong = (time.time() - t0) * 1000
    print(f"  WRONG (no sync)  : {t_wrong:.3f} ms  ← GPU not done yet")

    # BETTER but still wrong — includes Python overhead before and after
    t0 = time.time()
    C  = torch.mm(A, B)
    torch.cuda.synchronize()
    t_cpu_sync = (time.time() - t0) * 1000
    print(f"  Better (w/ sync) : {t_cpu_sync:.3f} ms  ← includes Python overhead")

    # CORRECT: CUDA events record timestamps on the GPU timeline
    # TODO 1: Create start and end CUDA events with enable_timing=True
    start_evt = None  # YOUR CODE HERE  → torch.cuda.Event(enable_timing=True)
    end_evt   = None  # YOUR CODE HERE  → torch.cuda.Event(enable_timing=True)

    # TODO 2: Record start_evt, run torch.mm(A, B), record end_evt
    pass  # YOUR CODE HERE  → start_evt.record()  → torch.mm  → end_evt.record()

    # TODO 3: Synchronize then measure elapsed time in ms
    elapsed_correct = None  # YOUR CODE HERE  → torch.cuda.synchronize(), start_evt.elapsed_time(end_evt)

    assert start_evt is not None,       "create start event"
    assert end_evt   is not None,       "create end event"
    assert elapsed_correct is not None, "measure elapsed time"
    assert elapsed_correct > 0,         "elapsed should be > 0"
    print(f"  CORRECT (events) : {elapsed_correct:.3f} ms  ← true GPU execution time")
else:
    elapsed_correct = 1.0
    print("  (CUDA not available — showing CPU timing only)")

print("  ✓ Section 1 passed — always use CUDA events for GPU timing")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Warmup — why the first run is always slow
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Warmup Runs ──")
print("""
  First GPU call triggers: CUDA context init, kernel JIT compilation,
  cuBLAS autotuning.  These happen ONCE but inflate the first measurement.
  Always discard the first N runs before benchmarking (minimum: 5).
  With torch.compile, discard at least 3 runs for compilation to complete.
""")

model = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512)).to(DEVICE)
model.eval()
x = torch.randn(32, 512, device=DEVICE)

raw_times = []
for i in range(10):
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(x)
        e.record()
        torch.cuda.synchronize()
        raw_times.append(s.elapsed_time(e))
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        raw_times.append((time.perf_counter() - t0) * 1000)

print(f"  {'Run':>4}  {'Time (ms)':>10}")
for i, t in enumerate(raw_times):
    marker = " ← WARMUP (discard)" if i < 3 else ""
    print(f"  {i:>4}  {t:>10.3f}{marker}")

first_run_ms = raw_times[0]
steady_ms    = sum(raw_times[5:]) / len(raw_times[5:])
print(f"\n  First run : {first_run_ms:.3f} ms   Steady state: {steady_ms:.3f} ms")
assert first_run_ms >= steady_ms * 0.5, "first run should be >= steady state"
print("  ✓ Section 2 passed — always warm up before benchmarking")

# ─────────────────────────────────────────────────────────────
# SECTION 3: A reusable benchmark() function
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Reusable benchmark() Helper ──")
print("""
  A correct benchmark function:
    1. Runs fn() `warmup` times (discards results)
    2. Synchronizes after warmup (so queued GPU work doesn't bleed in)
    3. Times `iters` runs with CUDA events (GPU) or perf_counter (CPU)
    4. Returns average time in milliseconds
""")

def benchmark(fn, warmup: int = 5, iters: int = 20, label: str = "") -> float:
    """
    TODO 4: Complete this benchmark function.
    See the description above for the exact implementation.
    The implementation is shown in the book (Section 2.5) and reproduced
    in the chapter summary code snippet — implement it here from scratch.
    """
    # 1. Warmup
    for _ in range(warmup):
        fn()

    # 2. Sync after warmup
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 3. Timed runs
    if torch.cuda.is_available():
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        avg_ms = s.elapsed_time(e) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        avg_ms = (time.perf_counter() - t0) / iters * 1000

    if label:
        print(f"  [{label}] {avg_ms:.3f} ms / iter")
    return avg_ms

def matmul_fn():
    a = torch.randn(256, 256, device=DEVICE)
    b = torch.randn(256, 256, device=DEVICE)
    return a @ b

result = benchmark(matmul_fn, warmup=5, iters=20, label="256×256 matmul")
assert result is not None and result > 0, "benchmark should return a positive time"
print(f"  Verified: benchmark() returned {result:.3f} ms")
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: torch.profiler — find slow ops
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: torch.profiler ──")
print("""
  torch.profiler wraps your code and records CPU + CUDA time for every op.
  Output: a table sorted by the metric you choose.

  Use sort_by='cuda_time_total' on GPU, 'cpu_time_total' on CPU.
  The operator at the top of the table is your first optimization target.
""")

model2 = nn.Sequential(
    nn.Embedding(1000, 128),
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(DEVICE)
model2.eval()
ids = torch.randint(0, 1000, (8, 16), device=DEVICE)

# TODO 5: Build the activities list
#   Always include ProfilerActivity.CPU
#   If DEVICE == "cuda", also include ProfilerActivity.CUDA
activities = None  # YOUR CODE HERE

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        model2(ids)

sort_by = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_by, row_limit=8))
assert activities is not None, "set the activities list"
print("  ✓ Section 4 passed — identify slow ops before optimizing")

# ─────────────────────────────────────────────────────────────
# SECTION 5: GPU Memory tracking
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: GPU Memory Tracking ──")
print("""
  Three memory quantities:
    allocated : tensors PyTorch actively holds right now
    reserved  : total memory the allocator holds (may be > allocated)
    peak      : maximum allocated since last reset_peak_memory_stats()

  OOM errors often occur when fragmentation causes reserved > available
  even though allocated is low.  torch.cuda.empty_cache() returns reserved
  memory to the OS pool.
""")

if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / 1e6   # MB

    # TODO 6: Allocate a 100 MB float32 tensor on DEVICE
    #   100 MB = 100*1e6 bytes / 4 bytes/float32 = 25,000,000 elements
    big_tensor = None  # YOUR CODE HERE  → torch.randn(25_000_000, device=DEVICE)

    mem_after = torch.cuda.memory_allocated() / 1e6

    # TODO 7: Delete big_tensor to free the memory
    pass  # YOUR CODE HERE  → del big_tensor

    torch.cuda.empty_cache()   # return reserved memory to OS
    mem_freed = torch.cuda.memory_allocated() / 1e6
    peak_mem  = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Before allocation : {mem_before:.1f} MB")
    print(f"  After  allocation : {mem_after:.1f} MB")
    print(f"  After  del+empty  : {mem_freed:.1f} MB")
    print(f"  Peak allocated    : {peak_mem:.1f} MB")

    assert mem_after > mem_before + 50, \
        f"big_tensor should add ~100 MB, got {mem_after-mem_before:.1f} MB"
    print("  ✓ Section 5 passed — track memory to avoid OOM")
else:
    print("  (CUDA not available — Section 5 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 6: NVTX markers for Nsight Systems
# ─────────────────────────────────────────────────────────────
print("\n── Section 6: NVTX Markers ──")
print("""
  NVTX (NVIDIA Tools Extension) adds named annotations to the Nsight
  Systems timeline.  Each range_push/range_pop pair appears as a
  colored band labelled with your string.

  To see the markers in Nsight Systems:
    nsys profile --trace=cuda,nvtx python I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py

  Pattern:
    torch.cuda.nvtx.range_push("phase_name")
    ... do work ...
    torch.cuda.nvtx.range_pop()
""")

if DEVICE == "cuda":
    # TODO 8: Add an NVTX range around each phase below.
    # Replace each `pass` with the correct range_push / range_pop calls.

    # Phase: tokenise
    pass  # YOUR CODE HERE  → torch.cuda.nvtx.range_push("tokenise")
    ids2 = torch.randint(0, 1000, (4, 32), device=DEVICE)
    pass  # YOUR CODE HERE  → torch.cuda.nvtx.range_pop()

    # Phase: forward
    pass  # YOUR CODE HERE  → torch.cuda.nvtx.range_push("forward")
    with torch.no_grad():
        out = model2(ids2)
    pass  # YOUR CODE HERE  → torch.cuda.nvtx.range_pop()

    # Phase: loss  (already filled in — study the pattern)
    torch.cuda.nvtx.range_push("loss")
    target  = torch.randint(0, 10, (4, 32), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    loss    = loss_fn(out.reshape(-1, 10), target.reshape(-1))
    torch.cuda.nvtx.range_pop()

    print(f"  loss = {loss.item():.4f}")
    print("  ✓ Section 6 passed — run under nsys to see NVTX ranges in timeline")
else:
    print("  (CUDA not available — NVTX Section 6 skipped)")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 05 complete!")
print("  You now know how to time, profile, and track memory for GPU code.")
print("  This is the foundation for every profiling chapter that follows.")
print("  Next: I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py")
print("=" * 60)
