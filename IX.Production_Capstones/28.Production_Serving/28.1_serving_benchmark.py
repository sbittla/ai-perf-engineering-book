#!/usr/bin/env python3
"""
28.Production_Serving/28.1_serving_benchmark.py  —  Chapter 28: Capstone 5
=======================================================================
Production LLM Serving Benchmark (vLLM): throughput-latency knee + cost/token.
Covers sections 28.1 - 28.3.

Difficulty: ****-  (4/5 - production benchmarking)
Est. time:  60 minutes (90 with a live vLLM server)
Expected ranges (reference, Llama-3.1-8B on one H100):
  - Knee throughput: ~6,000-7,000 output tok/s
  - Cost at knee (~$3/hr): ~$0.12-0.15 per 1M tokens
Troubleshooting:
  - Live server: `vllm serve <model> --gpu-memory-utilization 0.90`, then point
    BASE_URL at it. Without a server, the reference sweep runs.
  - If cost looks too high, you are likely measuring below the knee (low load).
Challenge extension:
  - Add a p99-TTFT SLO of 500 ms and report the max throughput that still meets
    it (the true operating point), not just the raw knee.

Run:  python IX.Production_Capstones/28.Production_Serving/28.1_serving_benchmark.py
"""
print("=" * 70)
print("  Exercise 28.1 - Capstone 5: Production LLM Serving Benchmark")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Throughput-latency sweep -> find the knee
# ─────────────────────────────────────────────────────────────
print("\n-- Section 1: Throughput-Latency Knee --")
print("""
  Sweep offered load and record throughput + p99 TTFT. Throughput climbs until
  the GPU saturates, then latency rises sharply (the knee). Operate just left
  of it: max throughput while the latency SLO still holds.
""")

# reference vLLM sweep: req/s -> (throughput tok/s, p99 TTFT ms)
sweep = {5:(1800, 90), 10:(3600, 130), 20:(6500, 280), 30:(6900, 650), 40:(7000, 1400)}
SLO_TTFT_MS = 500

def knee(sweep, slo_ms):
    ok = {r: v for r, (v, t) in sweep.items() if t <= slo_ms}
    return max(ok, key=lambda r: ok[r]) if ok else None

best = knee(sweep, SLO_TTFT_MS)
print(f"  {'req/s':>6} {'tok/s':>8} {'p99 TTFT':>10}")
for r, (tp, t) in sweep.items():
    mark = "  <- knee (meets SLO)" if r == best else ("  (SLO breached)" if t > SLO_TTFT_MS else "")
    print(f"  {r:>6} {tp:>8} {t:>8} ms{mark}")
assert best == 20, "knee under 500ms SLO is at 20 req/s"
print(f"  [check] operating point = {best} req/s, {sweep[best][0]} tok/s")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Cost per million tokens
# ─────────────────────────────────────────────────────────────
print("\n-- Section 2: Cost Per Million Tokens --")
print("""
  cost/token = GPU $/hr / tokens-per-hour. Running at the knee amortizes the
  fixed GPU rent over far more tokens than running at low load.
""")

def cost_per_m_tokens(usd_per_hr, tok_s):
    return usd_per_hr / (tok_s * 3600) * 1_000_000

at_knee = cost_per_m_tokens(3.0, sweep[best][0])
at_low  = cost_per_m_tokens(3.0, sweep[5][0])
print(f"  at knee  ({sweep[best][0]} tok/s): ${at_knee:5.3f} / 1M tokens")
print(f"  at low   ({sweep[5][0]} tok/s): ${at_low:5.3f} / 1M tokens "
      f"({at_low/at_knee:.1f}x worse)")
assert at_low > at_knee * 2, "low load should be much costlier per token"
print("  [check] running at the knee cuts cost/token several-fold")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Cost across GPU prices
# ─────────────────────────────────────────────────────────────
print("\n-- Section 3: Cost Across GPU Prices --")
prices = {"L40S": 1.0, "A100": 2.0, "H100": 3.0}
for gpu, price in prices.items():
    c = cost_per_m_tokens(price, sweep[best][0])
    print(f"  {gpu:<5} (${price:.1f}/hr): ${c:5.3f} / 1M tokens")
assert cost_per_m_tokens(1.0, 6500) < cost_per_m_tokens(3.0, 6500)
print("  [check] cost/token scales with hourly price at equal throughput")


print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise 28.1 complete")
print("  You found the SLO-respecting throughput knee and reported cost per")
print("  million tokens across operating points and GPU prices.")
print("=" * 70)
