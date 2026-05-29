#!/usr/bin/env python3
"""
V.Workload_Benchmarking/14.Benchmarking_Methodology/14.3_throughput_latency_curve.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 14: Benchmarking Methodology — Section 2: Throughput-Latency Curves
=======================================================================
Covers book section 14.2:
  • The throughput-latency (T-L) tradeoff: more throughput always costs latency
  • Sweeping batch size and request rate to trace the Pareto frontier
  • The "knee" of the curve: where latency starts degrading steeply
  • Queuing theory: Little's Law and why the knee appears
  • Selecting the optimal operating point from the T-L curve
  • How the Pareto frontier is used in MLPerf submissions

Run:  python V.Workload_Benchmarking/14.Benchmarking_Methodology/14.3_throughput_latency_curve.py
All sections must print ✓.
"""

import math
import statistics
import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 14.2 — Throughput-Latency Curves")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# Shared model
model = nn.Sequential(
    nn.Linear(256, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 256)
).to(DEVICE)
model.eval()


def benchmark(fn, warmup=5, iters=30):
    """Correct benchmark: warmup, then CUDA events or perf_counter."""
    for _ in range(warmup):
        fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
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


# ─────────────────────────────────────────────────────────────
# SECTION 1: Why throughput and latency trade off
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Throughput-Latency Tradeoff ──")
print("""
  SINGLE REQUEST (batch=1):
    Low latency: request processed immediately, ~5 ms.
    Low throughput: GPU processes 1 request at a time = 200 req/s.
    GPU utilisation: ~10% (for decode-style workloads).

  LARGE BATCH (batch=128):
    High throughput: 128 requests in ~15 ms = 8,533 req/s.
    High latency: each request waits for the batch to fill.
    GPU utilisation: ~90%.

  The key insight: you cannot have BOTH minimum latency AND maximum throughput
  simultaneously. The T-L curve shows the achievable trade-off space.

  LITTLE'S LAW:  L = λ × W
    L = average number of requests in the system
    λ = arrival rate (requests/second)
    W = average time in system (latency)

  At low load (λ small): W ≈ service time (latency is just processing time).
  Near capacity (λ → μ): queue builds → W → ∞ (latency explodes).
  The "knee" is where λ/μ ≈ 0.7–0.8 (70–80% utilisation).
""")
print("  ✓ Section 1 passed — throughput and latency are fundamentally coupled")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Sweeping batch size to trace the T-L curve
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Batch Size Sweep — Measuring the T-L Curve ──")
print("""
  The simplest way to trace the T-L curve is to sweep batch size.
  For each batch size B:
    - Latency   = time to complete one forward pass (affects one user waiting)
    - Throughput = B / latency  (samples processed per second)

  TODO 1: Implement tl_sweep() that runs the model across batch_sizes,
  measures latency (mean and P99) and throughput for each, and returns
  a list of dicts with keys: batch, latency_mean_ms, p99_ms, throughput_sps.
""")


def tl_sweep(model, batch_sizes, seq_len=256, warmup=5, iters=30) -> list:
    """
    TODO 1: Sweep batch sizes and return T-L curve data.
    For each B in batch_sizes:
      - Create input x of shape (B, seq_len) on DEVICE
      - Collect iters timing measurements using benchmark()
      - Compute mean_ms, p99_ms, throughput_sps = B / (mean_ms / 1000)
    Return list of dicts: {batch, latency_mean_ms, p99_ms, throughput_sps}
    Skip and note any OOM errors.
    """
    results = []
    for B in batch_sizes:
        try:
            x_b = torch.randn(B, seq_len, device=DEVICE)

            def fn():
                with torch.no_grad():
                    model(x_b)

            times = benchmark(fn, warmup=warmup, iters=iters)
            mean_ms = statistics.mean(times)
            p99_ms  = sorted(times)[int(0.99 * len(times))]
            tps     = B / (mean_ms / 1000)
            results.append({
                "batch":             B,
                "latency_mean_ms":   mean_ms,
                "p99_ms":            p99_ms,
                "throughput_sps":    tps,
            })
        except RuntimeError:
            print(f"  Batch={B}: OOM — skipping")
    return results


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
tl_data = tl_sweep(model, BATCH_SIZES)

assert len(tl_data) > 0, "T-L sweep should return at least one result"

print(f"\n  {'Batch':>6}  {'Latency (ms)':>13}  {'P99 (ms)':>10}  {'Throughput':>12}  {'P99/P50':>8}")
print(f"  {'─'*6}  {'─'*13}  {'─'*10}  {'─'*12}  {'─'*8}")
for r in tl_data:
    ratio = r["p99_ms"] / r["latency_mean_ms"] if r["latency_mean_ms"] > 0 else 0
    print(f"  {r['batch']:>6}  {r['latency_mean_ms']:>13.2f}  {r['p99_ms']:>10.2f}  "
          f"  {r['throughput_sps']:>10.0f}/s  {ratio:>8.2f}")

# Verify: larger batch → higher throughput
if len(tl_data) >= 2:
    assert tl_data[-1]["throughput_sps"] > tl_data[0]["throughput_sps"], \
        "Larger batch should have higher throughput"
print("  ✓ Section 2 passed — T-L curve collected across batch sizes")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Finding the "knee" of the T-L curve
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Finding the Knee — Optimal Operating Point ──")
print("""
  The "knee" of the T-L curve is the batch size where adding more
  throughput starts costing disproportionate latency.

  DEFINITION: knee is where the marginal latency cost per unit of
  additional throughput starts increasing rapidly (second derivative > 0).

  A practical heuristic for the knee:
    Find the smallest batch where P99 > P50 × 2.
    The batch just below that is the knee.

  For latency-sensitive SLOs, operate at or below the knee.
  For throughput-maximising use cases, operate above the knee.

  TODO 2: Implement find_knee() that returns the batch size at the knee,
  defined as the largest batch where p99 / latency_mean < threshold.
""")


def find_knee(tl_data: list, p99_to_mean_threshold: float = 2.0) -> dict:
    """
    TODO 2: Return the last T-L data point where p99_ms / latency_mean_ms < threshold.
    If all points are below threshold, return the last point.
    """
    knee = tl_data[0]
    for r in tl_data:
        ratio = r["p99_ms"] / r["latency_mean_ms"] if r["latency_mean_ms"] > 0 else 0
        if ratio < p99_to_mean_threshold:
            knee = r
    return knee


knee_point = find_knee(tl_data)
print(f"  Knee point: batch={knee_point['batch']}")
print(f"    Latency  : {knee_point['latency_mean_ms']:.2f} ms  (P99: {knee_point['p99_ms']:.2f} ms)")
print(f"    Throughput: {knee_point['throughput_sps']:.0f} samples/sec")

# Pareto frontier: for each latency budget, best throughput
print(f"\n  Pareto frontier (max throughput within latency budget):")
print(f"  {'Budget (ms)':>12}  {'Best batch':>11}  {'Throughput':>12}")
print(f"  {'─'*12}  {'─'*11}  {'─'*12}")
budgets = [5, 10, 20, 50, 100, 200]
for budget in budgets:
    candidates = [r for r in tl_data if r["p99_ms"] <= budget]
    if not candidates:
        continue
    best = max(candidates, key=lambda r: r["throughput_sps"])
    print(f"  {budget:>12}ms  {best['batch']:>11}  {best['throughput_sps']:>10.0f}/s")

assert knee_point is not None, "find_knee should return a result"
print("  ✓ Section 3 passed — knee detected and Pareto frontier computed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Request rate sweep (queuing model)
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Simulating Request Rate Sweep ──")
print("""
  In serving systems, latency depends on the ARRIVAL RATE of requests,
  not just the batch size. As arrival rate approaches system capacity,
  a queue builds and latency explodes.

  We model this with a simple M/D/1 queue (Poisson arrivals, deterministic service):
    Service rate μ = 1 / service_time
    Utilisation ρ = λ / μ  (λ = arrival rate)
    Mean latency W = service_time + (ρ × service_time) / (2 × (1 - ρ))

  This formula captures the exponential latency increase near capacity.

  TODO 3: Implement md1_latency_ms() that returns mean latency from the M/D/1 formula.
  Handle ρ >= 1.0 by returning infinity (system overloaded).
""")


def md1_latency_ms(arrival_rate: float, service_time_ms: float) -> float:
    """
    TODO 3: Return mean latency from M/D/1 queueing model.
    rho = arrival_rate × (service_time_ms / 1000)
    If rho >= 1.0: return float('inf')
    W_ms = service_time_ms + rho * service_time_ms / (2 * (1 - rho))
    Return W_ms.
    """
    rho = arrival_rate * (service_time_ms / 1000)
    if rho >= 1.0:
        return float("inf")
    return service_time_ms + rho * service_time_ms / (2 * (1.0 - rho))


# Verify: at ρ=0 → latency = service_time; at ρ→1 → latency→∞
assert abs(md1_latency_ms(0.0, 10.0) - 10.0) < 0.01, "At ρ=0, latency = service_time"
assert md1_latency_ms(100.0, 10.0) == float("inf"), "At ρ>=1, return inf"

SERVICE_TIME_MS = statistics.mean([r["latency_mean_ms"] for r in tl_data[:3]])  # batch=1 service time
capacity        = 1000 / SERVICE_TIME_MS  # requests per second at capacity

print(f"  Service time (batch=1): {SERVICE_TIME_MS:.2f} ms → capacity = {capacity:.0f} req/s")
print(f"\n  {'Load (req/s)':>14}  {'ρ (util)':>10}  {'Latency (ms)':>14}")
print(f"  {'─'*14}  {'─'*10}  {'─'*14}")

load_levels = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
for load_frac in load_levels:
    arrival = load_frac * capacity
    W = md1_latency_ms(arrival, SERVICE_TIME_MS)
    W_str = f"{W:.2f}" if W != float("inf") else "∞ (overloaded)"
    rho = arrival * (SERVICE_TIME_MS / 1000)
    print(f"  {arrival:>14.1f}  {rho:>10.2f}  {W_str:>14}")

# Verify latency at 80% utilisation > latency at 50%
W_50 = md1_latency_ms(0.50 * capacity, SERVICE_TIME_MS)
W_80 = md1_latency_ms(0.80 * capacity, SERVICE_TIME_MS)
assert W_80 > W_50, f"Higher load → higher latency: {W_50:.2f} vs {W_80:.2f}"
print(f"\n  Latency at 50% load: {W_50:.2f} ms → at 80% load: {W_80:.2f} ms "
      f"({W_80/W_50:.1f}× worse)")
print("  ✓ Section 4 passed — M/D/1 queueing model shows latency explosion near capacity")


# ─────────────────────────────────────────────────────────────
# SECTION 5: MLPerf-style benchmark report
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Benchmark Report Format ──")
print("""
  MLPerf Inference requires a specific report format. Even if you are not
  submitting to MLPerf, this format is a useful standard:

    Scenario: Offline (maximise throughput) or Server (SLO on P99 latency)
    Metric:   Offline: samples/sec;  Server: samples/sec subject to P99 < SLO

  TODO 4: Implement benchmark_report() that prints the summary table below
  and returns the recommended operating point (dict) based on the SLO.
""")


def benchmark_report(tl_data: list, p99_slo_ms: float, scenario: str = "server") -> dict:
    """
    TODO 4: Print benchmark summary and return recommended config.
    scenario='offline': return point with maximum throughput_sps.
    scenario='server':  return point with max throughput_sps where p99_ms <= p99_slo_ms.
    If no 'server' point meets SLO, return the one with smallest p99_ms.
    """
    print(f"\n  Benchmark Report — {scenario.upper()} scenario  (P99 SLO: {p99_slo_ms} ms)")
    print(f"  {'Batch':>6}  {'P50 (ms)':>10}  {'P99 (ms)':>10}  {'Tput (sps)':>12}  {'SLO OK':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}")
    for r in tl_data:
        ok = "✓" if r["p99_ms"] <= p99_slo_ms else "✗"
        print(f"  {r['batch']:>6}  {r['latency_mean_ms']:>10.2f}  {r['p99_ms']:>10.2f}  "
              f"{r['throughput_sps']:>12.0f}  {ok:>8}")

    if scenario == "offline":
        rec = max(tl_data, key=lambda r: r["throughput_sps"])
    else:
        candidates = [r for r in tl_data if r["p99_ms"] <= p99_slo_ms]
        if candidates:
            rec = max(candidates, key=lambda r: r["throughput_sps"])
        else:
            rec = min(tl_data, key=lambda r: r["p99_ms"])
    return rec


P99_SLO = 50.0  # ms
rec = benchmark_report(tl_data, p99_slo_ms=P99_SLO, scenario="server")
print(f"\n  Recommended: batch={rec['batch']}  "
      f"throughput={rec['throughput_sps']:.0f} sps  P99={rec['p99_ms']:.2f} ms")
assert "batch" in rec and "throughput_sps" in rec, "Report must return a valid recommendation"
print("  ✓ Section 5 passed — benchmark report generated with SLO-aware recommendation")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 14.2 complete!")
print("  You can now sweep batch sizes, trace T-L curves, find the")
print("  knee, apply queuing theory, and generate benchmark reports.")
print("  Next: V.Workload_Benchmarking/14.Benchmarking_Methodology/14.4_workload_characterization.py")
print("=" * 60)
