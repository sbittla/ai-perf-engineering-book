#!/usr/bin/env python3
"""
27.Observability/27.1_observability_metrics.py  —  Chapter 27: Production Observability
=======================================================================
Covers book sections 27.1 - 27.5:
  27.1  Metrics / logs / traces
  27.2  GPU telemetry with DCGM
  27.3  Prometheus + Grafana (PromQL)
  27.4  OpenTelemetry tracing
  27.5  SLOs, error-budget burn, capacity planning

Difficulty: ***--  (3/5 - data + arithmetic)
Est. time:  40-50 minutes
Expected ranges:
  - Healthy GPU SM activity band: 40-80%
  - p99 latency computed from a histogram should exceed p50
Troubleshooting:
  - No cluster needed; sample DCGM/latency data is embedded.
  - In production, dcgm-exporter publishes these on :9400/metrics for Prometheus.
Challenge extension:
  - Add a multi-window, multi-burn-rate alert (fast + slow burn) and show which
    window fires first for a sudden vs gradual SLO violation.

Run:  python VIII.Advanced_Performance/27.Observability/27.1_observability_metrics.py
"""
print("=" * 70)
print("  Exercise 27.1 - Production Observability")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Parse DCGM-style GPU metrics
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: DCGM GPU Telemetry --")
print("""
  DCGM / dcgm-exporter publishes SM activity, Tensor-Core activity, HBM
  bandwidth, power, temperature. SM activity is the first thing to check:
  a low value means the GPU is starved (DataLoader / batching), not compute-bound.
""")

# sample fleet snapshot: gpu -> sm_active %
sm_active = {"gpu0": 72, "gpu1": 68, "gpu2": 18, "gpu3": 74}
def classify(sm):
    if sm < 30:  return "STARVED (investigate input pipeline / batching)"
    if sm > 90:  return "SATURATED (scale out)"
    return "healthy"
for g, sm in sm_active.items():
    print(f"  {g}: SM active {sm:3d}%  -> {classify(sm)}")
starved = [g for g, sm in sm_active.items() if sm < 30]
assert starved == ["gpu2"], "should flag the one starved GPU"
print("  [check] flagged the starved GPU from SM-activity telemetry")


# ─────────────────────────────────────────────────────────────
# SECTION 2: p99 latency from a histogram (PromQL-style)
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: p99 Latency from a Histogram --")
print("""
  Prometheus stores latency as cumulative histogram buckets; p99 is interpolated
  from them (histogram_quantile). Here we compute it directly from bucket counts.
""")

# (upper_bound_seconds, cumulative_count)
buckets = [(0.1, 500), (0.25, 850), (0.5, 960), (1.0, 990), (2.0, 1000)]
total = buckets[-1][1]

def quantile_from_hist(buckets, q):
    target = q * buckets[-1][1]
    prev_b, prev_c = 0.0, 0
    for ub, c in buckets:
        if c >= target:
            # linear interpolation within the bucket
            frac = (target - prev_c) / (c - prev_c) if c > prev_c else 0
            return prev_b + frac * (ub - prev_b)
        prev_b, prev_c = ub, c
    return buckets[-1][0]

p50 = quantile_from_hist(buckets, 0.50)
p99 = quantile_from_hist(buckets, 0.99)
print(f"  p50 = {p50*1000:6.0f} ms     p99 = {p99*1000:6.0f} ms")
assert p99 > p50, "p99 must exceed p50"
print("  [check] p99 computed from histogram buckets")


# ─────────────────────────────────────────────────────────────
# SECTION 3: SLO error-budget burn + capacity planning
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: SLO Burn and Capacity --")
print("""
  SLO: p99 TTFT < 500 ms for 99% of requests. Alert on how fast we burn the
  1% error budget, not on raw spikes. Capacity: stay left of the latency knee.
""")

def error_budget_burn(bad, total, slo=0.99):
    budget = (1 - slo) * total          # allowed failures
    return (bad / budget) if budget else float("inf")

bad = sum(c2 - c1 for (b1, c1), (b2, c2) in zip([(0,0)]+buckets, buckets) if b2 > 0.5)
burn = error_budget_burn(40, 1000)      # 40 requests over the 500 ms SLO
print(f"  40 slow / 1000 reqs  ->  error-budget burn = {burn:.1f}x")
assert burn > 1.0, "burning faster than budget should alert"

# capacity: knee throughput vs target load
knee_tok_s = 6500
target_load_tok_s = 18000
replicas = -(-target_load_tok_s // knee_tok_s)   # ceil division
print(f"  target {target_load_tok_s} tok/s at knee {knee_tok_s} tok/s "
      f"-> need {replicas} replicas")
assert replicas == 3
print("  [check] error-budget burn and replica count computed")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 27.1 complete")
print("  You triaged GPU telemetry, computed p99 from a histogram, and turned")
print("  an SLO into a burn alert and a capacity (replica) decision.")
print("=" * 70)
