#!/usr/bin/env python3
"""
exercise_05_performance_basics.py  ─  PyTorch Basics: GPU Timing & Profiling
=============================================================================
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

# ─────────────────────────────────────────────────────────
# SECTION 1: The wrong way vs the right way to time GPU code
# ─────────────────────────────────────────────────────────
print("── Section 1: Timing GPU Operations Correctly ──")
print("""
  GPU operations are ASYNCHRONOUS.  The CPU submits work to a queue
  and immediately returns — the GPU may not have started yet.

  time.time() after a GPU op returns the CPU submission time,
  NOT the GPU completion time.  Always use CUDA Events.
""")

if DEVICE == "cuda":
    N = 1024
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float16)

    # WRONG: CPU timer (result is misleading)
    t0 = time.time()
    C  = torch.mm(A, B)
    t_wrong = (time.time() - t0) * 1000
    print(f"  WRONG (no sync) : {t_wrong:.3f} ms  ← GPU probably not done yet")

    # STILL WRONG: even with synchronize after timing start
    t0 = time.time()
    C  = torch.mm(A, B)
    torch.cuda.synchronize()
    t_cpu_sync = (time.time() - t0) * 1000
    print(f"  Better (w/sync) : {t_cpu_sync:.3f} ms  ← includes Python overhead")

    # CORRECT: CUDA events record timestamps on the GPU timeline
    # TODO 1: Create start and end CUDA events with enable_timing=True
    start_evt = None  # YOUR CODE HERE
    end_evt   = torch.cuda.Event(enable_timing=True)

    # TODO 2: Record start_evt, run torch.mm(A, B), record end_evt
    pass  # YOUR CODE HERE
    C = torch.mm(A, B)
    end_evt.record()

    # TODO 3: Synchronize then get elapsed time
    pass  # YOUR CODE HERE
    elapsed_correct = start_evt.elapsed_time(end_evt)

    assert start_evt is not None,       "create start event"
    assert end_evt   is not None,       "create end event"
    assert elapsed_correct is not None, "measure elapsed time"
    assert elapsed_correct > 0,         "elapsed should be > 0"
    print(f"  CORRECT (events): {elapsed_correct:.3f} ms  ← true GPU time")
else:
    elapsed_correct = 1.0
    print("  (CUDA not available — showing CPU timing only)")

print("  ✓ Section 1 — always use CUDA events for GPU timing")

# ─────────────────────────────────────────────────────────
# SECTION 2: Warmup — why the first run is always slow
# ─────────────────────────────────────────────────────────
print("\n── Section 2: Warmup Runs ──")
print("""
  First GPU call triggers: CUDA context init, kernel JIT compilation,
  cuBLAS autotuning.  These happen ONCE but inflate the first measurement.
  Always discard the first N runs before benchmarking.
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
        with torch.no_grad(): model(x)
        e.record()
        torch.cuda.synchronize()
        raw_times.append(s.elapsed_time(e))
    else:
        t0 = time.perf_counter()
        with torch.no_grad(): model(x)
        raw_times.append((time.perf_counter() - t0) * 1000)

print(f"  {'Run':>4}  {'Time (ms)':>10}")
for i, t in enumerate(raw_times):
    marker = " ← WARMUP (discard)" if i < 3 else ""
    print(f"  {i:>4}  {t:>10.3f}{marker}")

first_run_ms = raw_times[0]
steady_ms    = sum(raw_times[5:]) / len(raw_times[5:])
print(f"\n  First run: {first_run_ms:.3f}ms  Steady state: {steady_ms:.3f}ms")
assert first_run_ms >= steady_ms * 0.5, "first run should be >= steady state"
print("  ✓ Section 2 — always warm up before benchmarking")

# ─────────────────────────────────────────────────────────
# SECTION 3: Writing a reusable benchmark function
# ─────────────────────────────────────────────────────────
print("\n── Section 3: Reusable benchmark() Helper ──")

def benchmark(fn, warmup=5, iters=20, label=""):
    """
    TODO 4: Complete this function.
    It should:
      1. Run fn() `warmup` times (discard results)
      2. If CUDA available, synchronize after warmup
      3. Time `iters` runs using CUDA events (CUDA) or perf_counter (CPU)
      4. Return average time in milliseconds
    """
    # 1. Warmup runs
    for _ in range(warmup):
        fn()

    # 2. Sync after warmup so queued GPU work doesn't bleed into timing
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

# Test it
def matmul_fn():
    a = torch.randn(256, 256, device=DEVICE)
    b = torch.randn(256, 256, device=DEVICE)
    return a @ b

result = benchmark(matmul_fn, warmup=5, iters=20, label="256×256 matmul")
assert result is not None and result > 0, "benchmark should return a positive time"
print(f"  256×256 matmul: {result:.3f} ms")
print("  ✓ Section 3 — reusable benchmark function")

# ─────────────────────────────────────────────────────────
# SECTION 4: torch.profiler — find slow ops
# ─────────────────────────────────────────────────────────
print("\n── Section 4: torch.profiler ──")
print("  torch.profiler records every op with CPU + CUDA time.\n")

model2 = nn.Sequential(
    nn.Embedding(1000, 128),
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 10)
).to(DEVICE)
model2.eval()

ids = torch.randint(0, 1000, (8, 16), device=DEVICE)

# TODO 5: Profile model2(ids) with CPU + CUDA activities
activities = None  # YOUR CODE HERE
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        model2(ids)

sort_by = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
print(prof.key_averages().table(sort_by=sort_by, row_limit=8))

print("  ✓ Section 4 — use profiler to find slow ops before optimising")

# ─────────────────────────────────────────────────────────
# SECTION 5: GPU Memory tracking
# ─────────────────────────────────────────────────────────
print("\n── Section 5: GPU Memory Tracking ──")
print("""
  Three memory values to know:
    allocated : tensors PyTorch actively uses right now
    reserved  : total GPU memory the allocator holds (may be larger)
    peak      : maximum allocated since last reset_peak_memory_stats()
""")

if DEVICE == "cuda":
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated() / 1e6   # MB

    # TODO 6: Allocate a 100MB float32 tensor on DEVICE
    #   100MB = 100*1e6 bytes / 4 bytes_per_float = 25M floats
    big_tensor = None  # YOUR CODE HERE

    mem_after = torch.cuda.memory_allocated() / 1e6

    # TODO 7: Delete big_tensor and empty the cache
    pass  # YOUR CODE HERE
    torch.cuda.empty_cache()

    mem_freed = torch.cuda.memory_allocated() / 1e6
    peak_mem  = torch.cuda.max_memory_allocated() / 1e6

    print(f"  Before allocation : {mem_before:.1f} MB")
    print(f"  After  allocation : {mem_after:.1f} MB")
    print(f"  After  del+empty  : {mem_freed:.1f} MB")
    print(f"  Peak allocated    : {peak_mem:.1f} MB")

    assert mem_after > mem_before + 50, f"big_tensor should add ~100MB, got {mem_after-mem_before:.1f}MB"
    print("  ✓ Section 5 — track memory to avoid OOM")
else:
    print("  (CUDA not available — skipped)")

# ─────────────────────────────────────────────────────────
# SECTION 6: nvtx markers for Nsight Systems
# ─────────────────────────────────────────────────────────
print("\n── Section 6: NVTX Markers ──")
print("""
  NVTX (NVIDIA Tools Extension) lets you add named annotations to the
  Nsight Systems timeline.  When you open an nsys report, your marker
  names appear as coloured bands — easy to identify your code phases.

  Usage:
    torch.cuda.nvtx.range_push("data_loading")
    ...
    torch.cuda.nvtx.range_pop()
""")

if DEVICE == "cuda":
    # TODO 8: Add NVTX markers around the three sections below.
    # Names: "tokenize", "forward", "loss"

    pass  # YOUR CODE HERE
    torch.cuda.nvtx.range_push("tokenize")
    ids2 = torch.randint(0, 1000, (4, 32), device=DEVICE)
    torch.cuda.nvtx.range_pop()

    # Section: forward
    torch.cuda.nvtx.range_push("forward")
    with torch.no_grad():
        out = model2(ids2)
    torch.cuda.nvtx.range_pop()

    # Section: loss
    torch.cuda.nvtx.range_push("loss")
    target = torch.randint(0, 10, (4, 32), device=DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(out.reshape(-1, 10), target.reshape(-1))
    torch.cuda.nvtx.range_pop()

    print(f"  loss = {loss.item():.4f}")
    print("  ✓ Section 6 — add NVTX markers, then profile with:")
    print("    nsys profile --trace=cuda,nvtx python exercise_05_performance_basics.py")
else:
    print("  (CUDA not available — NVTX skipped)")

print("\n" + "=" * 60)
print("  ALL SECTIONS COMPLETE — Exercise 05 done!")
print("  You now know how to time, profile, and measure GPU code.")
print("  This is the foundation for everything in the course.")
print("=" * 60)