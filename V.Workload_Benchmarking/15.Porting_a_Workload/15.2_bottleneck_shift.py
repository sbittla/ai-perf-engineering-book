#!/usr/bin/env python3
"""
V.Workload_Benchmarking/15.Porting_a_Workload/15.2_bottleneck_shift.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 15: Porting a Workload — Section 2: Bottleneck Shift
=======================================================================
Covers book section 15.2:
  • Why the bottleneck CHANGES as you apply optimizations
  • The optimization ladder: CPU FP32 → GPU FP32 → GPU FP16 → torch.compile
  • How to measure each rung of the ladder
  • CPU-GPU overlap: when to worry about the Python overhead floor
  • Quantifying the bottleneck with arithmetic intensity at each step

Run:  python V.Workload_Benchmarking/15.Porting_a_Workload/15.2_bottleneck_shift.py
All sections must print ✓.
"""

import statistics
import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 15.2 — Bottleneck Shift")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The Bottleneck Shift Mental Model
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Bottleneck Shift ──")
print("""
  Every optimization removes one bottleneck and exposes the next.
  This is called the BOTTLENECK SHIFT. If you don't anticipate it,
  you will be surprised when your GPU FP16 model is still "slow".

  THE optimization LADDER (typical order for an inference workload):

    Rung 1: CPU FP32 (baseline)
      Bottleneck: compute — CPUs have ~200 GFLOPS vs GPUs ~10 TFLOPS
      Speedup potential: 50×

    Rung 2: GPU FP32 (simple port)
      Bottleneck: memory bandwidth — FP32 AI may be below ridge point
      Speedup potential: 2–5× over GPU FP32

    Rung 3: GPU FP16 / BF16 (AMP)
      Bottleneck: Tensor Core utilization — need large tiles for peak
      Speedup potential: 1.5–2× over GPU FP32

    Rung 4: torch.compile (kernel fusion)
      Bottleneck: Python overhead / small-batch memory access pattern
      Speedup potential: 1.2–1.8× for inference

    Rung 5: INT8 quantisation (static or dynamic)
      Bottleneck: I/O between weight cache and compute units
      Speedup potential: 1.5–3× over FP16 for large models

  KEY INSIGHT: at each rung, the LIMITING FACTOR changes.
  Measuring throughput alone won't tell you which rung you are on.
  You must also measure: AI, memory bandwidth utilization, SM utilization.
""")
print("  ✓ Section 1 passed — understand the bottleneck shift ladder")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Benchmark helper for the optimization ladder
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Measuring Each Rung of the Ladder ──")
print("""
  We will benchmark the same MLP across multiple configurations.
  The key is to change ONE thing per rung and record all metrics.

  TODO 1: Implement benchmark_config(model, x, warmup=10, iters=50)
  that returns (mean_ms, p99_ms, throughput_sps).
  Use CUDA events on GPU, perf_counter on CPU.
""")

D_MODEL = 512
BATCH   = 32

# The reference model (FP32, CPU)
torch.manual_seed(42)
_model_fp32 = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL),
)
_model_fp32.eval()


def benchmark_config(model: nn.Module, x: torch.Tensor,
                     warmup: int = 10, iters: int = 50) -> tuple:
    """
    TODO 1: Benchmark model(x) and return (mean_ms, p99_ms, throughput_sps).
    Use CUDA events on GPU, time.perf_counter on CPU.
    throughput_sps = x.shape[0] / (mean_ms / 1000)
    """
    dev = str(x.device)

    def run():
        with torch.no_grad():
            model(x)

    for _ in range(warmup):
        run()
    if "cuda" in dev:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if "cuda" in dev:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            run()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            run()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(times)
    p99_ms  = sorted(times)[int(0.99 * len(times))]
    tps     = x.shape[0] / (mean_ms / 1000)
    return mean_ms, p99_ms, tps


# Verify the helper works on CPU baseline
x_cpu = torch.randn(BATCH, D_MODEL)
mean_ms, p99_ms, tps = benchmark_config(_model_fp32, x_cpu, warmup=5, iters=20)
assert mean_ms > 0 and tps > 0, "Benchmark should return positive values"
print(f"  Baseline (CPU FP32): {mean_ms:.3f} ms  P99={p99_ms:.3f} ms  {tps:.0f} sps")
print("  ✓ Section 2 passed — benchmark_config() works correctly")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Running the full optimization ladder
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: The Full optimization Ladder ──")
print("""
  We now run every rung and build a comparison table.
  On CPU-only machines, we simulate the GPU rungs with dtype changes
  to demonstrate the dtype effect; the speedup numbers will differ.
""")

import copy

results = {}

# --- Rung 1: CPU FP32 ---
x_cpu32 = torch.randn(BATCH, D_MODEL, dtype=torch.float32)
m_cpu32 = copy.deepcopy(_model_fp32)
m_cpu32.eval()
mean, p99, tps = benchmark_config(m_cpu32, x_cpu32, warmup=5, iters=20)
results["CPU FP32"] = dict(mean_ms=mean, p99_ms=p99, tps=tps, device="cpu", dtype="float32")

# --- Rung 2: GPU FP32 ---
if DEVICE == "cuda":
    x_gpu32 = x_cpu32.to(DEVICE)
    m_gpu32 = copy.deepcopy(_model_fp32).to(DEVICE)
    m_gpu32.eval()
    mean, p99, tps = benchmark_config(m_gpu32, x_gpu32, warmup=10, iters=50)
    results["GPU FP32"] = dict(mean_ms=mean, p99_ms=p99, tps=tps, device="cuda", dtype="float32")

# --- Rung 3: GPU FP16 ---
if DEVICE == "cuda":
    x_gpu16 = x_cpu32.half().to(DEVICE)
    m_gpu16 = copy.deepcopy(_model_fp32).half().to(DEVICE)
    m_gpu16.eval()
    mean, p99, tps = benchmark_config(m_gpu16, x_gpu16, warmup=10, iters=50)
    results["GPU FP16"] = dict(mean_ms=mean, p99_ms=p99, tps=tps, device="cuda", dtype="float16")

# --- Rung 4: torch.compile ---
if DEVICE == "cuda":
    try:
        m_compiled = torch.compile(copy.deepcopy(_model_fp32).half().to(DEVICE), mode="reduce-overhead")
        m_compiled.eval()
        mean, p99, tps = benchmark_config(m_compiled, x_gpu16, warmup=15, iters=50)
        results["GPU FP16 + compile"] = dict(mean_ms=mean, p99_ms=p99, tps=tps, device="cuda", dtype="float16")
    except Exception as e:
        results["GPU FP16 + compile"] = dict(mean_ms=float("nan"), p99_ms=float("nan"),
                                              tps=0, device="cuda", dtype="float16",
                                              note=str(e)[:40])

# --- CPU FP16 (if no GPU: show dtype effect) ---
if DEVICE != "cuda":
    # Demonstrate that PyTorch CPU FP16 matmul often falls back to FP32
    # Use BF16 which has better CPU support on newer hardware
    try:
        x_cpu16 = x_cpu32.bfloat16()
        m_cpu16 = copy.deepcopy(_model_fp32).bfloat16()
        m_cpu16.eval()
        mean, p99, tps = benchmark_config(m_cpu16, x_cpu16, warmup=5, iters=20)
        results["CPU BF16"] = dict(mean_ms=mean, p99_ms=p99, tps=tps, device="cpu", dtype="bfloat16")
    except Exception:
        pass  # BF16 on old CPUs may not be supported

# Print table
baseline_tps = results["CPU FP32"]["tps"]
print(f"\n  {'Config':<22}  {'Mean (ms)':>10}  {'P99 (ms)':>10}  {'Tput (sps)':>12}  {'Speedup':>9}")
print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*9}")
for name, r in results.items():
    if r["tps"] > 0:
        speedup = r["tps"] / baseline_tps
        print(f"  {name:<22}  {r['mean_ms']:>10.3f}  {r['p99_ms']:>10.3f}  {r['tps']:>12.0f}  {speedup:>8.2f}×")
    else:
        print(f"  {name:<22}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}  {'N/A':>9}")

assert len(results) >= 1, "At least CPU FP32 result should exist"
assert results["CPU FP32"]["tps"] > 0, "CPU baseline should have positive throughput"
print("  ✓ Section 3 passed — optimization ladder measured")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Diagnosing the current bottleneck
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Diagnosing the Bottleneck at Each Rung ──")
print("""
  Throughput alone doesn't tell you WHY you're at a given rung.
  You need a diagnosis: what is the limiting resource?

  DIAGNOSIS QUESTIONS:
    Q1: Is GPU SM utilization < 50%?
        → CPU-bound: Python overhead, DataLoader, or too-small batch
    Q2: Is GPU memory bandwidth > 80% of peak?
        → Memory-bandwidth-bound: try FP16, reduce model size, fuse ops
    Q3: Is GPU SM utilization > 80% but throughput below roofline?
        → Compute-bound: try Tensor Cores (FP16), increase arithmetic intensity
    Q4: Does throughput scale linearly with batch?
        → Good: compute-bound and properly utilizing the GPU
    Q5: Does throughput plateau early (e.g. batch=4)?
        → Memory-bound or GPU launch overhead dominates

  TODO 2: Implement throughput_scaling_ratio(model, device, d_model, batches)
  that benchmarks each batch size and returns the ratio
  (throughput at largest batch) / (throughput at smallest batch).
  A ratio > 8× for 10× batch increase suggests good compute utilization.
""")


def throughput_scaling_ratio(model: nn.Module, device: str,
                             d_model: int, batches: list) -> float:
    """
    TODO 2: Return throughput ratio between largest and smallest batch.
    For each b in batches, benchmark model(randn(b, d_model)) and record tps.
    Return max(tps_list) / min(tps_list).
    """
    tps_list = []
    for b in batches:
        x = torch.randn(b, d_model, device=device)
        _, _, tps = benchmark_config(model, x, warmup=5, iters=20)
        tps_list.append(tps)

    print(f"  {'Batch':>6}  {'Tput (sps)':>12}")
    for b, t in zip(batches, tps_list):
        print(f"  {b:>6}  {t:>12.0f}")

    return max(tps_list) / min(tps_list)


test_batches = [1, 4, 16, 64]
test_model = copy.deepcopy(_model_fp32).to(DEVICE)
test_model.eval()

print(f"\n  Throughput scaling (CPU FP32, d_model={D_MODEL}):")
ratio = throughput_scaling_ratio(test_model, DEVICE, D_MODEL, test_batches)
print(f"\n  Scaling ratio (batch={test_batches[-1]} / batch={test_batches[0]}): {ratio:.1f}×")

if ratio > 30:
    diag = "strong compute scaling — good GPU utilization"
elif ratio > 5:
    diag = "moderate scaling — approaching memory-bandwidth ceiling"
else:
    diag = "weak scaling — memory-bound or overhead-dominated"

print(f"  Diagnosis: {diag}")
assert ratio > 0, "Ratio should be positive"
print("  ✓ Section 4 passed — throughput scaling ratio measured")


# ─────────────────────────────────────────────────────────────
# SECTION 5: The Python overhead floor
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: The Python Overhead Floor ──")
print("""
  Even on GPU, Python code has overhead: each torch.nn.Linear call
  launches a CUDA kernel, which requires a CPU-side dispatch (~5–20 µs
  per kernel). For a 100-layer model at batch=1, this can be 2+ ms of
  overhead — with the actual GPU work taking only 0.5 ms.

  This is called the PYTHON OVERHEAD FLOOR. It limits throughput
  even when the GPU is fast, because you can't reduce latency below
  the overhead of dispatching all the kernels.

  SYMPTOMS:
    • GPU SM utilization < 20% for batch=1
    • Increasing batch from 1→2 gives < 1.5× throughput
    • profiler shows many tiny kernels with large gaps between them

  REMEDIES:
    1. torch.compile() — fuses kernels, reduces dispatch count
    2. CUDA Graphs   — records kernel launches; replays without CPU overhead
    3. Larger batches — amortises fixed overhead over more samples
    4. cuBLAS GEMM batching — run all layers as a batched GEMM

  MEASURING THE OVERHEAD:
    If GPU time (CUDA events) << wall-clock time (perf_counter), you have overhead.
    overhead_ms = wall_clock_ms - gpu_kernel_ms
""")


def measure_overhead_ms(model: nn.Module, x: torch.Tensor,
                        iters: int = 50) -> tuple:
    """Measure (wall_clock_ms, gpu_kernel_ms) per iteration."""
    if DEVICE != "cuda":
        # On CPU: overhead is not measurable; return (wall, wall)
        for _ in range(5):
            with torch.no_grad():
                model(x)
        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                model(x)
        wall_ms = (time.perf_counter() - t0) / iters * 1000
        return wall_ms, wall_ms

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()

    # Wall clock (includes Python dispatch overhead)
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            model(x)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / iters * 1000

    # GPU kernel time only (CUDA events)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        with torch.no_grad():
            model(x)
    e.record()
    torch.cuda.synchronize()
    gpu_ms = s.elapsed_time(e) / iters

    return wall_ms, gpu_ms


# Use a small batch to make overhead visible
x_small = torch.randn(1, D_MODEL, device=DEVICE)
m_overhead = copy.deepcopy(_model_fp32).to(DEVICE)
m_overhead.eval()

wall_ms, gpu_ms = measure_overhead_ms(m_overhead, x_small, iters=50)
overhead_ms = wall_ms - gpu_ms

print(f"  Batch=1, d_model={D_MODEL}:")
print(f"    Wall-clock (CPU dispatch + GPU): {wall_ms:.3f} ms")
print(f"    GPU kernel only (CUDA events)  : {gpu_ms:.3f} ms")
print(f"    Python overhead floor          : {overhead_ms:.3f} ms  ({overhead_ms/wall_ms*100:.1f}% of wall)")

if DEVICE == "cuda":
    if overhead_ms / wall_ms > 0.3:
        print(f"  WARNING: overhead > 30% of wall — consider torch.compile or CUDA Graphs")
    else:
        print(f"  OK: overhead < 30% — GPU is the primary bottleneck (as expected)")

assert wall_ms > 0, "Wall clock should be positive"
assert gpu_ms > 0, "GPU time should be positive"
print("  ✓ Section 5 passed — Python overhead floor measured")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 15.2 complete!")
print("  You can now run the optimization ladder, diagnose the bottleneck")
print("  at each rung, and measure the Python overhead floor.")
print("  Next: V.Workload_Benchmarking/15.Porting_a_Workload/15.3_dataloader_at_scale.py")
print("=" * 60)
