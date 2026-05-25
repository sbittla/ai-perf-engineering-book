"""
Exercise 14.1 — The Science of Honest Measurement

Chapter 14 (Background): Why Benchmarks Lie and How to Fix Them
Book: AI Systems Performance Engineering

Run:
    python 14.1_measurement_basics.py

What this exercise does:
  1. Explains the three fundamental measurement problems: dynamic systems,
     warmup effects, and the observer effect.
  2. Demonstrates cache-warmup effects on a live CPU workload.
  3. Shows why mean is misleading — computes P50/P90/P95/P99 on a bimodal distribution.
  4. Uses CV (Coefficient of Variation) to diagnose benchmark stability.
  5. Applies Amdahl's Law: calculates real end-to-end speedup vs. GPU-only speedup.
  6. Compares four benchmark types: microbenchmark, synthetic, trace replay, parametrised.
  7. Runs a rigorous CUDA-event timing harness (if GPU available) with P50/P99/CV output.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then explore the chapters in Part V for hands-on work.
"""

import sys
import math
import time
import random
import statistics

try:
    import torch
    TORCH = True
except ImportError:
    TORCH = False

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Why Measurement Is Hard
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: Why Measurement Is Hard")
print("=" * 60)

print("""
Three fundamental problems with measuring AI system performance:

Problem 1: Dynamic Systems
  Modern hardware adapts to workload in real time.
  CPU: frequency scaling (Intel Turbo Boost / AMD Precision Boost)
  GPU: clock boost (base 1410 MHz → boost 1980 MHz on RTX 4090)
  DRAM: power management states (C-states, P-states)

  Consequence: your first measurement is NOT representative.
  The CPU may be at base clock; your second run will be faster.

Problem 2: Warmup Effects
  cuDNN auto-tuning: the first forward pass benchmarks 5-20 kernel
  variants for your specific tensor shapes and selects the fastest.
  This takes 200-2000 ms. Run 1 is slow; all subsequent runs are fast.

  Python JIT: PyTorch 2.0+ torch.compile() traces and compiles your
  model on first call. Run 1 compiles; runs 2+ execute compiled code.

  L2/L3 cache warmup: data repeatedly accessed moves from DRAM
  into cache. Run 1 is cache-cold; later runs are cache-warm.

Problem 3: Observer Effect
  Profiling changes timing. Adding torch.profiler() adds 10-30%
  overhead from event recording. CUDA event timing is more accurate
  than CPU wall-clock for GPU kernels.

Ground Rule: always discard the first N runs as warmup,
then report P50 and P99 over at least 50 stable samples.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Demonstrating Warmup Effects
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: Warmup Effect Demo (CPU-based)")
print("=" * 60)

def expensive_fn(n=500):
    total = 0.0
    for i in range(n):
        total += math.sqrt(i) * math.log(i + 1)
    return total

# Simulate cache-cold first run vs warmed-up subsequent runs
timings = []
for i in range(20):
    t0 = time.perf_counter()
    expensive_fn(n=50000)
    t1 = time.perf_counter()
    timings.append((t1 - t0) * 1000)

print(f"  {'Run':>4}  {'Time (ms)':>12}  {'Note'}")
print(f"  {'-'*4}  {'-'*12}  {'-'*30}")
for i, t in enumerate(timings):
    note = ""
    if i == 0:
        note = "← first run (cache cold)"
    elif i < 3:
        note = "← warming up"
    elif i == len(timings) - 1:
        note = "← stable"
    print(f"  {i+1:>4}  {t:>12.3f}  {note}")

steady = timings[5:]
print()
print(f"  First run:   {timings[0]:.3f} ms")
print(f"  Steady P50:  {statistics.median(steady):.3f} ms")
print(f"  Difference:  {(timings[0] / statistics.median(steady) - 1) * 100:.1f}% slower on run 1")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Mean vs Percentiles
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: Mean vs Percentiles — Why the Mean Lies")
print("=" * 60)

print("""
Scenario: a model serving system processes requests with these latencies (ms):
  [10, 10, 10, 10, 10, 10, 10, 10, 10, 1000]
  (9 fast requests, 1 slow due to a garbage collection pause)

  Mean   = (9 × 10 + 1000) / 10 = 109 ms  ← MISLEADING
  P50    = 10 ms                            ← typical user experience
  P90    = 10 ms                            ← 90% of users see this
  P99    = 1000 ms                          ← 1 in 100 users sees this

The mean obscures the tail. In a system serving 1,000 requests/second,
P99 means 10 users per second are getting 100× worse service than average.
""")

random.seed(42)
samples = [random.gauss(50, 5) for _ in range(990)] + [random.gauss(800, 50) for _ in range(10)]
random.shuffle(samples)

samples_sorted = sorted(samples)
n = len(samples_sorted)

def percentile(data, p):
    idx = int(math.ceil(p / 100 * len(data))) - 1
    return data[max(0, min(idx, len(data) - 1))]

mean_val = sum(samples) / len(samples)
p50 = percentile(samples_sorted, 50)
p90 = percentile(samples_sorted, 90)
p95 = percentile(samples_sorted, 95)
p99 = percentile(samples_sorted, 99)

print("Simulated distribution: 990 fast (50±5 ms) + 10 slow (800±50 ms):")
print(f"  Mean : {mean_val:>8.1f} ms  ← inflated by tail")
print(f"  P50  : {p50:>8.1f} ms  ← typical user")
print(f"  P90  : {p90:>8.1f} ms")
print(f"  P95  : {p95:>8.1f} ms")
print(f"  P99  : {p99:>8.1f} ms  ← worst 1%")
print()
print("Rule: always report at least P50 and P99. Never report only the mean.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Coefficient of Variation — Is My Benchmark Stable?
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: Coefficient of Variation (CV)")
print("=" * 60)

print("""
CV = std_dev / mean × 100%

CV tells you how noisy your benchmark is relative to its own magnitude.
It is unit-free, so you can compare stability across different workloads.

  CV < 2%   — excellent, your result is reliable
  CV 2-5%   — acceptable for most benchmarks
  CV 5-10%  — something is variable (thermal throttle? background load?)
  CV > 10%  — do not report this result, find and fix the noise source
""")

scenarios = [
    ("Stable (thermal steady state)",   [50.1, 50.2, 50.0, 50.3, 50.1, 50.2, 49.9, 50.1]),
    ("Mild variability (mixed load)",   [50, 52, 48, 55, 47, 53, 51, 49, 54, 46]),
    ("Noisy (thermal throttling)",      [50, 60, 45, 70, 40, 65, 55, 35, 75, 55]),
    ("Bimodal (warmup contamination)",  [200, 51, 52, 49, 50, 48, 53, 51, 50, 52]),
]

print(f"  {'Scenario':<40} {'Mean':>8} {'StdDev':>8} {'CV %':>8} {'Verdict'}")
print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8} {'-'*15}")
for label, data in scenarios:
    m = statistics.mean(data)
    s = statistics.stdev(data)
    cv = s / m * 100
    verdict = "OK" if cv < 5 else ("check" if cv < 10 else "NOISY")
    if label.startswith("Bimodal"):
        verdict = "warmup!"
    print(f"  {label:<40} {m:>8.1f} {s:>8.1f} {cv:>8.1f} {verdict:>15}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Amdahl's Law — Calculating Real Speedup
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: Amdahl's Law — What Speedup Is Actually Possible?")
print("=" * 60)

print("""
Amdahl's Law (1967): if you speed up a fraction P of your workload by S×,
the overall end-to-end speedup is:

  Speedup = 1 / ((1 - P) + P/S)

Where:
  P = fraction of time spent on the part you are optimising
  S = speedup of that specific part
  (1 - P) = the part you are NOT changing (serial fraction)

Example: if the GPU kernel runs 10× faster but only occupies 80% of runtime,
  overall speedup = 1 / (0.2 + 0.8/10) = 1 / 0.28 = 3.57×  (not 10×!)
""")

gpu_speedup = 4.0
print(f"GPU kernel speedup assumed: {gpu_speedup:.0f}×")
print()
print(f"  {'GPU fraction (P)':<22} {'Serial fraction (1-P)':<24} {'End-to-end speedup'}")
print(f"  {'-'*22} {'-'*24} {'-'*20}")

for p in [0.20, 0.50, 0.80, 0.90, 0.95, 0.99, 1.00]:
    serial = 1.0 - p
    speedup = 1.0 / (serial + p / gpu_speedup)
    print(f"  {p*100:>6.0f}%{'':<15} {serial*100:>6.0f}%{'':<17} {speedup:>8.2f}×")

print()
print("Key lesson: optimising the GPU kernel matters only if the GPU is the")
print("bottleneck. If DataLoader occupies 50% of wall time, a 10× GPU speedup")
print("delivers only 1.67× end-to-end improvement.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Benchmark Types Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: Benchmark Type Comparison")
print("=" * 60)

benchmark_types = [
    ("Microbenchmark",           "Single kernel or op",
     "Isolates one variable",    "Not representative of full workload"),
    ("End-to-end synthetic",     "Full model, synthetic data",
     "Repeatable, no data prep", "May not match real input distribution"),
    ("Prod trace replay",        "Real request log replayed",
     "Most realistic",           "Hard to reproduce, privacy concerns"),
    ("Parametrised replay",      "Synthetic data, real shape distribution",
     "Reproducible + realistic", "Requires shape analysis upfront"),
]

print(f"  {'Type':<28} {'Scope':<28} {'Strength':<30} {'Weakness'}")
print(f"  {'-'*28} {'-'*28} {'-'*30} {'-'*35}")
for btype, scope, strength, weakness in benchmark_types:
    print(f"  {btype:<28} {scope:<28} {strength:<30} {weakness}")

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: The Rigorous Timing Harness
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 7: A Rigorous Timing Harness")
print("=" * 60)

print("""
Every benchmark result you report must include:
  1. Measurement method (CUDA events / CPU wall clock / perf counters)
  2. Warmup iterations (how many runs discarded before recording)
  3. Sample count (how many stable measurements taken)
  4. P50 and P99 (not just mean)
  5. CV% (to confirm the benchmark was stable)

Below is the harness pattern used throughout this book:
""")

if TORCH and torch.cuda.is_available():
    print("Running GPU timing harness demo...")

    def benchmark_fn():
        a = torch.randn(1024, 1024, device="cuda")
        b = torch.randn(1024, 1024, device="cuda")
        return torch.mm(a, b)

    WARMUP = 10
    RUNS = 50

    for _ in range(WARMUP):
        benchmark_fn()
        torch.cuda.synchronize()

    latencies = []
    for _ in range(RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        benchmark_fn()
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))

    lat_sorted = sorted(latencies)
    mean_lat = statistics.mean(latencies)
    std_lat = statistics.stdev(latencies)
    cv_lat = std_lat / mean_lat * 100
    p50_lat = percentile(lat_sorted, 50)
    p99_lat = percentile(lat_sorted, 99)

    print(f"  Operation   : torch.mm(1024, 1024)")
    print(f"  Warmup runs : {WARMUP}")
    print(f"  Sample runs : {RUNS}")
    print(f"  Measurement : CUDA events (device-side, not CPU wall clock)")
    print(f"  Mean        : {mean_lat:.3f} ms")
    print(f"  P50         : {p50_lat:.3f} ms")
    print(f"  P99         : {p99_lat:.3f} ms")
    print(f"  CV          : {cv_lat:.2f}%  {'← stable' if cv_lat < 5 else '← check variability'}")
else:
    print("  [No CUDA GPU available — install PyTorch with CUDA to run GPU harness]")
    print()
    print("  Template (copy this pattern for all your benchmarks):")
    print("    WARMUP, RUNS = 10, 50")
    print("    for _ in range(WARMUP):")
    print("        fn(); torch.cuda.synchronize()")
    print("    latencies = []")
    print("    for _ in range(RUNS):")
    print("        start = torch.cuda.Event(enable_timing=True)")
    print("        end   = torch.cuda.Event(enable_timing=True)")
    print("        start.record(); fn(); end.record()")
    print("        torch.cuda.synchronize()")
    print("        latencies.append(start.elapsed_time(end))")

print()
print("Explore next: V.Workload_Benchmarking/14.Benchmarking_Methodology/")
print("              V.Workload_Benchmarking/15.Porting_a_Workload/")
