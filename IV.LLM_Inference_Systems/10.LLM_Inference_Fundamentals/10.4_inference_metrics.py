#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.4_inference_metrics.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 10: LLM Inference Fundamentals — Section 3: Inference Metrics
=======================================================================
Covers book section 10.3:
  • TTFT, TPS, end-to-end latency: what each measures and why it matters
  • P50 and P99 percentiles: why averages mislead for latency SLOs
  • Throughput vs latency tradeoff: why increasing batch increases throughput
    but also increases TTFT
  • Building a simulated request trace to compute all key metrics

Run:  python IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_inference_metrics.py
All sections must print ✓.
"""

import math
import random
import statistics
import time

print("=" * 60)
print("  Exercise 10.3 — LLM Inference Metrics")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: The LLM latency vocabulary
# ─────────────────────────────────────────────────────────────
print("── Section 1: Inference Metrics Vocabulary ──")
print("""
  TTFT  (Time To First Token)
    Wall-clock time from request arrival to first token delivered.
    Equals prefill time for a single request.
    SLO example: < 500ms for interactive chat.

  TPS  (Tokens Per Second)  —  also called generation speed
    Number of new tokens produced per second during decode.
    Dictates how long a response takes to stream to the user.
    SLO example: > 30 tok/s to feel faster than human reading speed.

  E2E Latency  (End-to-End latency)
    TTFT + decode time for all requested tokens.
    = TTFT + (n_tokens / TPS) × 1000 ms

  Throughput
    System-level metric: total tokens/s across ALL concurrent requests.
    High throughput ≠ low latency — they often trade against each other.

  P50 / P99 percentiles
    P50 (median): 50% of requests complete within this time.
    P99: 99% of requests complete within this time.
    Averages hide tail latency; SLOs are typically written as P99.
""")
print("  ✓ Section 1 passed — commit these definitions to memory")


# ─────────────────────────────────────────────────────────────
# SECTION 2: P50 and P99 — why averages are not enough
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Percentile Latency — P50 vs P99 ──")
print("""
  In production, a small fraction of requests have very long prompts
  or request many tokens. These long-tail requests inflate average latency
  and mask the experience of the majority.

  Example SLO: "P99 TTFT < 2000ms"
  This means: 99% of requests get their first token within 2 seconds.
  1% of requests may exceed this — which is acceptable for rare edge cases.

  TODO 1: Implement percentile(values, pct) below.
  pct=0.50 → P50, pct=0.99 → P99.
""")


def percentile(values: list, pct: float) -> float:
    """
    TODO 1: Return the pct-th percentile of values.
    Sort the list, then index at int(pct * len(values)).
    Clamp the index to len(values) - 1.
    """
    # YOUR CODE HERE
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = min(int(pct * len(sorted_v)), len(sorted_v) - 1)
    return sorted_v[idx]


# Test with known distribution (0–99 so index 50 → value 50, index 99 → value 99)
test_values = list(range(100))       # 0 to 99
assert percentile(test_values, 0.50) == 50, f"P50 should be 50, got {percentile(test_values, 0.50)}"
assert percentile(test_values, 0.99) == 99, f"P99 should be 99, got {percentile(test_values, 0.99)}"

# Simulate realistic TTFT distribution (most fast, a few slow)
random.seed(42)
ttft_samples = (
    [random.uniform(50, 150)  for _ in range(900)] +  # 90% fast requests
    [random.uniform(500, 800) for _ in range(90)]  +  # 9% medium
    [random.uniform(1500, 3000) for _ in range(10)]    # 1% very slow
)
random.shuffle(ttft_samples)

avg_ttft = statistics.mean(ttft_samples)
p50_ttft = percentile(ttft_samples, 0.50)
p99_ttft = percentile(ttft_samples, 0.99)

print(f"  Simulated TTFT over {len(ttft_samples)} requests:")
print(f"    Mean : {avg_ttft:>8.1f} ms  ← average (misleading!)")
print(f"    P50  : {p50_ttft:>8.1f} ms  ← most users see this")
print(f"    P99  : {p99_ttft:>8.1f} ms  ← SLO target")
print(f"    P99 / Mean ratio: {p99_ttft/avg_ttft:.1f}×  (P99 much worse than avg)")

assert p99_ttft > avg_ttft, "P99 should exceed average due to long-tail outliers"
print("  ✓ Section 2 passed — always report P99, not just mean latency")


# ─────────────────────────────────────────────────────────────
# SECTION 3: E2E latency calculation
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: End-to-End Latency Calculation ──")
print("""
  E2E latency for a single request:
    e2e_ms = ttft_ms + (n_tokens_generated / tps) * 1000

  For a chat interface: user waits e2e_ms before they see the full response.
  For a streaming interface: user waits ttft_ms before seeing ANY text.

  TODO 2: Implement e2e_latency_ms() below.
""")


def e2e_latency_ms(ttft_ms: float, n_tokens: int, tps: float) -> float:
    """
    TODO 2: Return total end-to-end latency in ms.
    e2e_ms = ttft_ms + (n_tokens / tps) * 1000
    """
    # YOUR CODE HERE
    return ttft_ms + (n_tokens / tps) * 1000


# Test with round numbers
assert abs(e2e_latency_ms(200, 100, 50) - 2200) < 1, \
    "e2e_ms: 200ms TTFT + 100tok/50tok/s*1000 = 2200ms"

scenarios = [
    ("Fast GPU, short response",  100, 50,  30),
    ("Fast GPU, long response",   100, 50, 300),
    ("Slow server, short",        500, 20,  30),
    ("Slow server, long",         500, 20, 300),
]

print(f"  {'Scenario':<28}  {'TTFT':>6}  {'TPS':>5}  {'Tokens':>7}  {'E2E (ms)':>10}")
print(f"  {'─'*28}  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*10}")
for name, ttft_ms, tps, n_tok in scenarios:
    e2e = e2e_latency_ms(ttft_ms, n_tok, tps)
    print(f"  {name:<28}  {ttft_ms:>6}  {tps:>5}  {n_tok:>7}  {e2e:>10.0f}")

print("  ✓ Section 3 passed — e2e latency = TTFT + decode time")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Throughput vs latency tradeoff
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Throughput vs Latency Tradeoff ──")
print("""
  Increasing batch size improves THROUGHPUT (tokens/sec across all requests)
  because the GPU is better utilised. But it also increases TTFT for each
  individual request, because requests must wait for others in the batch.

  This is the fundamental tradeoff in LLM serving:
    Small batch → low latency, low GPU utilisation
    Large batch → high GPU utilisation, high latency

  The optimal operating point depends on your SLO:
    • Latency-sensitive API: keep batch small (TTFT < 200ms)
    • Throughput-maximising offline: use large batch (maximise tok/s)

  Here we simulate a simple queueing model.
""")


def simulate_serving(
    n_requests: int,
    batch_size: int,
    prefill_ms_per_request: float,
    decode_ms_per_token: float,
    n_tokens_per_request: int,
    seed: int = 42
) -> dict:
    """
    Simulate static batching: group requests into batches, process each batch
    in sequence. Returns dict with ttft_list, e2e_list, throughput_tps.
    """
    random.seed(seed)
    ttft_list = []
    e2e_list  = []
    t_clock   = 0.0
    total_tokens = 0

    for batch_start in range(0, n_requests, batch_size):
        batch = list(range(batch_start, min(batch_start + batch_size, n_requests)))
        # Prefill: all requests processed in parallel (takes max of their prefills)
        batch_prefill_ms = prefill_ms_per_request * len(batch)
        t_after_prefill  = t_clock + batch_prefill_ms

        # Each request records its TTFT = time from batch_start to after prefill
        for _ in batch:
            ttft_list.append(batch_prefill_ms)

        # Decode: all requests generate n_tokens
        decode_ms = decode_ms_per_token * n_tokens_per_request
        t_clock   = t_after_prefill + decode_ms
        total_tokens += len(batch) * n_tokens_per_request

        for _ in batch:
            e2e_list.append(batch_prefill_ms + decode_ms)

    elapsed_s   = t_clock / 1000
    throughput  = total_tokens / elapsed_s if elapsed_s > 0 else 0

    return {
        "ttft_list":  ttft_list,
        "e2e_list":   e2e_list,
        "throughput": throughput,
    }


N_REQUESTS   = 64
PREFILL_MS   = 50    # ms per request in batch
DECODE_MS_PT = 2     # ms per token
N_TOK        = 100   # tokens per request

print(f"  Requests={N_REQUESTS}, prefill={PREFILL_MS}ms/req, decode={DECODE_MS_PT}ms/tok, {N_TOK} tokens")
print(f"\n  {'Batch':>6}  {'Throughput':>12}  {'P50 TTFT':>10}  {'P99 TTFT':>10}  {'P50 E2E':>10}")
print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

batch_sizes    = [1, 4, 8, 16, 32]
throughputs    = []
p99_ttfts      = []

for bs in batch_sizes:
    result = simulate_serving(N_REQUESTS, bs, PREFILL_MS, DECODE_MS_PT, N_TOK)
    tp   = result["throughput"]
    p50t = percentile(result["ttft_list"], 0.50)
    p99t = percentile(result["ttft_list"], 0.99)
    p50e = percentile(result["e2e_list"],  0.50)
    throughputs.append(tp)
    p99_ttfts.append(p99t)
    print(f"  {bs:>6}  {tp:>10.0f}/s  {p50t:>10.0f}  {p99t:>10.0f}  {p50e:>10.0f}")

# Verify: larger batch → higher throughput and higher TTFT
assert throughputs[-1] > throughputs[0], \
    "throughput should increase with batch size"
assert p99_ttfts[-1] > p99_ttfts[0], \
    "P99 TTFT should increase with batch size"
print(f"\n  Throughput: {throughputs[0]:.0f} → {throughputs[-1]:.0f} tok/s  (batch 1 → {batch_sizes[-1]})")
print(f"  P99 TTFT:  {p99_ttfts[0]:.0f} → {p99_ttfts[-1]:.0f} ms  (latency penalty)")
print("  ✓ Section 4 passed — larger batch trades latency for throughput")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Setting and measuring SLOs
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: SLO Compliance Check ──")
print("""
  An SLO (Service Level Objective) is a target for a specific percentile.
  Example: "P99 TTFT < 500ms AND P99 E2E < 5000ms"

  TODO 3: Implement check_slo() that returns True if all requests satisfy
  the given SLO, False otherwise.
""")


def check_slo(values: list, pct: float, threshold_ms: float) -> bool:
    """
    TODO 3: Return True if the pct-th percentile of values is <= threshold_ms.
    """
    # YOUR CODE HERE
    return percentile(values, pct) <= threshold_ms


# Test SLO checker
assert check_slo([100, 200, 300], 0.99, 300) is True,  "P99=300 should pass SLO=300"
assert check_slo([100, 200, 300], 0.99, 299) is False, "P99=300 should fail SLO=299"

TTFT_SLO = 600   # ms
E2E_SLO  = 3000  # ms
print(f"  SLO targets: P99 TTFT < {TTFT_SLO}ms, P99 E2E < {E2E_SLO}ms")
print(f"\n  {'Batch':>6}  {'P99 TTFT':>10}  {'TTFT OK':>9}  {'P99 E2E':>10}  {'E2E OK':>8}")
print(f"  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*10}  {'─'*8}")
for bs in batch_sizes:
    result = simulate_serving(N_REQUESTS, bs, PREFILL_MS, DECODE_MS_PT, N_TOK)
    p99t = percentile(result["ttft_list"], 0.99)
    p99e = percentile(result["e2e_list"],  0.99)
    tok  = "✓" if check_slo(result["ttft_list"], 0.99, TTFT_SLO) else "✗"
    eok  = "✓" if check_slo(result["e2e_list"],  0.99, E2E_SLO)  else "✗"
    print(f"  {bs:>6}  {p99t:>10.0f}  {tok:>9}  {p99e:>10.0f}  {eok:>8}")

print("  ✓ Section 5 passed — SLO compliance verified per batch size")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 10.3 complete!")
print("  You can compute TTFT, TPS, E2E latency, P50/P99,")
print("  and reason about the throughput-latency tradeoff.")
print("  Next: IV.LLM_Inference_Systems/11.Batching_Strategies/11.1_static_batching.py")
print("=" * 60)
