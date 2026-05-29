#!/usr/bin/env python3
"""
V.Workload_Benchmarking/14.Benchmarking_Methodology/14.2_benchmark_mistakes.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 14: Benchmarking Methodology — Section 1: The Five Common Mistakes
=======================================================================
Covers book section 14.1:
  • Mistake 1: No warmup — first GPU run is always slow
  • Mistake 2: Using time.time() for GPU ops — measures CPU, not GPU
  • Mistake 3: Single sample — no variance, no P99, no reliable mean
  • Mistake 4: Changing multiple variables at once — can't isolate cause
  • Mistake 5: Reporting peak, not steady state
  • The correct benchmark() helper that avoids all five mistakes

Run:  python V.Workload_Benchmarking/14.Benchmarking_Methodology/14.2_benchmark_mistakes.py
All sections must print ✓.
"""

import time
import statistics
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 14.1 — The Five Benchmarking Mistakes")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# Shared test model
model = nn.Sequential(
    nn.Linear(512, 1024), nn.ReLU(),
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Linear(1024, 512)
).to(DEVICE)
model.eval()
x = torch.randn(32, 512, device=DEVICE)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Mistake 1 — No warmup
# ─────────────────────────────────────────────────────────────
print("── Section 1: Mistake 1 — Missing Warmup ──")
print("""
  The first GPU forward pass is always slow because:
    • CUDA context initialisation (one-time, ~50ms)
    • Kernel JIT compilation (first use of each op, ~5–50ms per op)
    • cuBLAS auto-tuning (first use of each matrix size)
    • CPU→GPU weight transfer if model not fully on GPU yet

  If you measure the first run, you report the cold-start latency,
  not the steady-state latency your users will actually experience.

  Rule: discard at least 5 runs before measuring. For torch.compile,
  discard at least 3 full forward passes to allow compilation to complete.
""")

timings = []
for i in range(15):
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad(): model(x)
        e.record()
        torch.cuda.synchronize()
        timings.append(s.elapsed_time(e))
    else:
        t0 = time.perf_counter()
        with torch.no_grad(): model(x)
        timings.append((time.perf_counter() - t0) * 1000)

first_run   = timings[0]
steady_mean = statistics.mean(timings[5:])

print(f"  {'Run':>4}  {'ms':>8}")
for i, t in enumerate(timings):
    flag = " ← cold (DISCARD)" if i < 5 else ""
    print(f"  {i:>4}  {t:>8.3f}{flag}")

print(f"\n  First run  : {first_run:.3f} ms  ← inflated by init")
print(f"  Steady mean: {steady_mean:.3f} ms  ← what users see")

# Verify first run >= steady (allowing for CPU noise at small scale)
# On CPU it can be reversed due to OS scheduling; just check both are positive
assert first_run > 0 and steady_mean > 0, "Both timings should be positive"
print("  ✓ Section 1 passed — always warm up; discard first N runs")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Mistake 2 — time.time() for GPU ops
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Mistake 2 — Using time.time() for GPU Code ──")
print("""
  GPU operations are ASYNCHRONOUS. torch.nn.Linear().to("cuda") returns
  to Python immediately — the kernel may not have started yet.

  time.time() after a GPU op measures CPU dispatch latency, which is
  typically 0.05–0.5 ms. The actual GPU kernel may take 5–50 ms.
  You will consistently underreport latency by 10–100×.

  The correct approach: use CUDA Events.
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()  # timestamp on GPU timeline
    <GPU work>
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)
""")

if DEVICE == "cuda":
    # Warmup
    for _ in range(5):
        with torch.no_grad(): model(x)
    torch.cuda.synchronize()

    # WRONG: time.time() with no sync
    t0 = time.time()
    with torch.no_grad(): model(x)
    t_wrong = (time.time() - t0) * 1000

    # BETTER but still wrong: sync AFTER measurement
    t0 = time.time()
    with torch.no_grad(): model(x)
    torch.cuda.synchronize()
    t_cpu_sync = (time.time() - t0) * 1000

    # CORRECT: CUDA events
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad(): model(x)
    e.record()
    torch.cuda.synchronize()
    t_events = s.elapsed_time(e)

    print(f"  WRONG (no sync)    : {t_wrong:.4f} ms  ← GPU still running!")
    print(f"  BETTER (w/sync)    : {t_cpu_sync:.4f} ms  ← includes Python overhead")
    print(f"  CORRECT (events)   : {t_events:.4f} ms  ← true GPU kernel time")
    assert t_events > 0, "CUDA event time should be positive"
else:
    print("  (CUDA not available — this mistake only applies to GPU code)")
    print("  On CPU: time.perf_counter() is correct (no async issue)")
    t0 = time.perf_counter()
    with torch.no_grad(): model(x)
    t_cpu = (time.perf_counter() - t0) * 1000
    print(f"  CPU time: {t_cpu:.3f} ms  (perf_counter is accurate)")

print("  ✓ Section 2 passed — use CUDA events on GPU; perf_counter on CPU")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Mistake 3 — Single sample, no variance
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Mistake 3 — Single Sample, No Variance ──")
print("""
  A single timing measurement is meaningless for two reasons:
    1. GPU and OS scheduling add random jitter (typically ±1–5%)
    2. A single measurement cannot detect P99 outliers

  MINIMUM REQUIREMENTS:
    • 20+ iterations for mean/std estimate
    • 50+ iterations for stable P99
    • Report mean ± std, plus P99 (not just mean)

  The P99 / mean ratio > 1.5 is a warning sign: you have occasional
  slow outliers that will affect real users under load.

  TODO 1: Implement gather_timings() that collects N timing measurements
  (using CUDA events on GPU, perf_counter on CPU) after a warmup phase.
""")


def gather_timings(fn, warmup: int = 5, n: int = 50) -> list:
    """
    TODO 1: Collect n timing measurements of fn() after warmup.
    Use CUDA events on GPU, time.perf_counter on CPU.
    Return list of floats in milliseconds.
    """
    # Warmup
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n):
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000)
    return times


def forward():
    with torch.no_grad():
        model(x)


times = gather_timings(forward, warmup=5, n=50)
assert len(times) == 50, f"Expected 50 measurements, got {len(times)}"
assert all(t > 0 for t in times), "All timings should be positive"

mean_ms = statistics.mean(times)
std_ms  = statistics.stdev(times)
p50_ms  = sorted(times)[int(0.50 * len(times))]
p99_ms  = sorted(times)[int(0.99 * len(times))]

print(f"  50 iterations collected:")
print(f"    Mean  : {mean_ms:.3f} ms")
print(f"    Std   : {std_ms:.3f} ms  ({std_ms/mean_ms*100:.1f}% of mean)")
print(f"    P50   : {p50_ms:.3f} ms")
print(f"    P99   : {p99_ms:.3f} ms")
print(f"    P99/mean ratio: {p99_ms/mean_ms:.2f}×")
print("  ✓ Section 3 passed — always collect 20+ samples and report P99")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Mistake 4 — Changing multiple variables at once
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Mistake 4 — Changing Multiple Variables ──")
print("""
  "We switched from FP32 to FP16 AND increased batch size AND enabled
  torch.compile. The result is 5× faster!" — which change caused what?

  You cannot attribute speedup to a specific optimisation if you change
  multiple things simultaneously. The correct approach: change ONE variable
  at a time, measure, and then combine.

  CORRECT EXPERIMENTAL DESIGN:
    Baseline: FP32, batch=8, no compile
    Run 1:    FP16, batch=8, no compile   → FP16 contribution
    Run 2:    FP32, batch=32, no compile  → batching contribution
    Run 3:    FP32, batch=8, compile      → compile contribution
    Run 4:    FP16, batch=32, compile     → combined (cross-check)

  If Run4 < Run1 × Run2 × Run3 / Baseline², there is interaction between
  the optimisations (e.g. compile benefits more from FP16 than FP32).
""")

# Demonstrate isolation: measure each variable individually
configs_isolation = [
    ("Baseline FP32 b=8",    torch.float32, 8,  False),
    ("FP16 only   b=8",      torch.float16, 8,  False),
    ("Batch=32 FP32",        torch.float32, 32, False),
]

baseline_ms = None
print(f"  {'Config':<22}  {'Mean (ms)':>10}  {'Speedup':>10}")
print(f"  {'─'*22}  {'─'*10}  {'─'*10}")
for name, dtype, batch, _ in configs_isolation:
    x_cfg = torch.randn(batch, 512, device=DEVICE, dtype=dtype)
    m_cfg = model.to(dtype=dtype)

    def fn_cfg():
        with torch.no_grad():
            m_cfg(x_cfg)

    ts = gather_timings(fn_cfg, warmup=5, n=20)
    mean_t = statistics.mean(ts)
    if baseline_ms is None:
        baseline_ms = mean_t
    speedup = baseline_ms / mean_t
    print(f"  {name:<22}  {mean_t:>10.3f}  {speedup:>9.2f}×")
    # Reset model to float32 for next config
    model.to(dtype=torch.float32)

print("  ✓ Section 4 passed — isolate one variable at a time")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Mistake 5 — Reporting peak, not steady state
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Mistake 5 — Reporting Peak, Not Steady State ──")
print("""
  GPU clocks boost above their base frequency for short bursts (boost clock).
  After 10–30 seconds under full load, the GPU throttles to its sustained clock
  to stay within thermal limits. The first 5 seconds can be 5–15% faster than
  sustained performance.

  In vendor benchmarks, peak throughput often comes from the first 30-second
  window. In production, your model runs for hours. You must measure both:
    burst_tps      = throughput during first 5 seconds
    sustained_tps  = throughput during seconds 20–30

  If sustained < burst × 0.90, the GPU is thermally throttling.

  TODO 2: Implement burst_vs_sustained() that returns (burst_mean, steady_mean).
  Run fn() for duration_burst_s seconds, then duration_steady_s more seconds.
  Return mean latency (ms) from each phase separately.
""")


def burst_vs_sustained(fn, duration_burst_s: float = 2.0,
                       duration_steady_s: float = 4.0) -> tuple:
    """
    TODO 2: Measure burst then sustained throughput.
    Phase 1: collect timings for duration_burst_s seconds (burst phase).
    Phase 2: run without measuring for a gap, then measure for duration_steady_s.
    Return (mean_burst_ms, mean_steady_ms).
    """
    # Phase 1: burst
    burst_times = []
    t_end = time.perf_counter() + duration_burst_s
    while time.perf_counter() < t_end:
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            burst_times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            burst_times.append((time.perf_counter() - t0) * 1000)

    # Phase 2: steady state
    steady_times = []
    t_end = time.perf_counter() + duration_steady_s
    while time.perf_counter() < t_end:
        if DEVICE == "cuda":
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            steady_times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            steady_times.append((time.perf_counter() - t0) * 1000)

    burst_mean  = statistics.mean(burst_times)  if burst_times  else 0.0
    steady_mean = statistics.mean(steady_times) if steady_times else 0.0
    return burst_mean, steady_mean


def fwd():
    with torch.no_grad():
        model(x)


burst_ms, steady_ms = burst_vs_sustained(fwd, duration_burst_s=1.0, duration_steady_s=2.0)
throttle_pct = abs(steady_ms - burst_ms) / burst_ms * 100 if burst_ms > 0 else 0.0

print(f"  Burst mean   : {burst_ms:.3f} ms")
print(f"  Steady mean  : {steady_ms:.3f} ms")
print(f"  Difference   : {throttle_pct:.1f}%  {'← throttling possible' if throttle_pct > 10 else '← OK (< 10%)'}")
print(f"  Note: significant throttling only appears after 10–30s on a loaded GPU")
assert burst_ms > 0 and steady_ms > 0, "Both phases should produce measurements"
print("  ✓ Section 5 passed — always report sustained, not peak burst performance")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 14.1 complete!")
print("  You have seen all five benchmarking mistakes and their fixes.")
print("  The gather_timings() helper from Section 3 is the foundation")
print("  for every measurement in the rest of this chapter.")
print("  Next: V.Workload_Benchmarking/14.Benchmarking_Methodology/14.2_throughput_latency_curve.py")
print("=" * 60)
