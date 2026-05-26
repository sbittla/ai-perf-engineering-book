#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/11.Batching_Strategies/11.1_static_batching.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 11: Batching Strategies — Section 1: Static Batching
=======================================================================
Covers book section 11.1:
  • How static batching (HuggingFace generate) works
  • Padding waste: GPU computes dummy tokens for finished requests
  • How padding fraction grows with output length variance
  • Measuring GPU idle time and wasted token-steps
  • Concrete arithmetic showing why padding kills throughput

Run:  python IV.LLM_Inference_Systems/11.Batching_Strategies/11.1_static_batching.py
All sections must print ✓.
"""

import random
import statistics

print("=" * 60)
print("  Exercise 11.1 — Static Batching and Padding Waste")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: Static batching mechanics
# ─────────────────────────────────────────────────────────────
print("── Section 1: How Static Batching Works ──")
print("""
  Static batching groups requests into a fixed-size batch.
  All requests in the batch are processed together for exactly
  max(output_lengths) decode steps. Shorter requests are padded
  with dummy tokens once they finish — the GPU still computes
  on those positions (wasted work).

  TIMELINE (batch of 4, output lengths [10, 50, 100, 200]):
    Steps 0–10   : all 4 sequences active
    Steps 11–50  : seq 0 done, padding → 3 real + 1 padded
    Steps 51–100 : seqs 0,1 done           → 2 real + 2 padded
    Steps 101–200: seqs 0,1,2 done         → 1 real + 3 padded

  Total token-steps computed  : 4 × 200 = 800
  Real (non-padding) token-steps: 10+50+100+200 = 360
  PADDING FRACTION: 1 - 360/800 = 55%  ← GPU wastes 55% of decode work

  This gets worse with high variance in output lengths.
""")
print("  ✓ Section 1 passed — static batching pads all sequences to max length")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring padding waste
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Padding Fraction Calculation ──")
print("""
  For a batch with output lengths L = [l_0, l_1, ..., l_{B-1}]:
    Total token-steps  = B × max(L)
    Real  token-steps  = sum(L)
    Padding fraction   = 1 - sum(L) / (B × max(L))

  TODO 1: Implement padding_fraction() below.
""")


def padding_fraction(output_lengths: list) -> float:
    """
    TODO 1: Return the fraction of token-steps wasted on padding.
    = 1 - sum(output_lengths) / (len(output_lengths) * max(output_lengths))
    Return 0.0 if output_lengths is empty or max is 0.
    """
    # YOUR CODE HERE
    if not output_lengths or max(output_lengths) == 0:
        return 0.0
    B   = len(output_lengths)
    total_real    = sum(output_lengths)
    total_computed = B * max(output_lengths)
    return 1.0 - total_real / total_computed


# Verify with the example in the explanation
ex_batch = [10, 50, 100, 200]
pf = padding_fraction(ex_batch)
assert abs(pf - (1 - 360/800)) < 0.01, f"Expected ~0.55, got {pf:.3f}"
print(f"  Example batch {ex_batch}: padding fraction = {pf:.1%}")

# Explore how variance affects padding
print(f"\n  {'Distribution':<30}  {'Avg len':>8}  {'Max len':>8}  {'Padding':>10}")
print(f"  {'─'*30}  {'─'*8}  {'─'*8}  {'─'*10}")

scenarios = [
    ("All equal (no variance)",     [100] * 16),
    ("Low variance  (95–105)",      [random.randint(95,  105) for _ in range(16)]),
    ("Medium variance (50–150)",    [random.randint(50,  150) for _ in range(16)]),
    ("High variance  (10–500)",     [random.randint(10,  500) for _ in range(16)]),
    ("Worst case: 1 long + 15 short", [10]*15 + [500]),
]
random.seed(42)
for name, lengths in scenarios:
    pf_val = padding_fraction(lengths)
    print(f"  {name:<30}  {statistics.mean(lengths):>8.0f}  {max(lengths):>8}  {pf_val:>10.1%}")

# Worst case should be highest padding
pf_worst = padding_fraction([10]*15 + [500])
pf_equal = padding_fraction([100] * 16)
assert pf_worst > pf_equal, "High variance → more padding waste"
print("  ✓ Section 2 passed — padding fraction grows with output length variance")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Simulating static batching throughput
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Simulating Static Batching Throughput ──")
print("""
  We model a server processing 128 requests with static batching.
  Each decode step takes decode_ms_per_batch ms (linear in batch size).
  Prefill is also linear in batch size (parallel, but takes batch * prefill_ms).

  Throughput = total real tokens generated / total wall-clock seconds.
""")


def run_static_batching(
    requests: list,
    batch_size: int,
    prefill_ms_per_req: float,
    decode_ms_per_step: float,
) -> dict:
    """
    Simulate static batching.
    requests: list of output_lengths (one int per request)
    Returns dict with real_tokens, total_ms, throughput_tps, padding_fraction.
    """
    total_ms       = 0.0
    real_tokens    = 0
    padded_tokens  = 0
    latencies      = []

    for i in range(0, len(requests), batch_size):
        batch   = requests[i : i + batch_size]
        max_len = max(batch)

        # Prefill: all requests in batch processed together
        prefill_ms  = prefill_ms_per_req * len(batch)
        # Decode: max_len steps × constant cost per step (full batch always active)
        decode_ms   = decode_ms_per_step * max_len
        batch_ms    = prefill_ms + decode_ms
        total_ms   += batch_ms

        for req_len in batch:
            real_tokens   += req_len
            padded_tokens += max_len - req_len
            latencies.append(batch_ms)

    throughput  = real_tokens / (total_ms / 1000) if total_ms > 0 else 0
    total_steps = real_tokens + padded_tokens
    pf          = padded_tokens / total_steps if total_steps > 0 else 0

    return {
        "real_tokens":      real_tokens,
        "padded_tokens":    padded_tokens,
        "total_ms":         total_ms,
        "throughput_tps":   throughput,
        "padding_fraction": pf,
        "latencies":        latencies,
    }


random.seed(42)
N_REQUESTS   = 128
PREFILL_MS   = 20
DECODE_MS    = 0.5   # ms per decode step for full batch

# Realistic output lengths (most short, some long)
requests = [
    random.choices([20, 50, 100, 200, 500],
                   weights=[0.35, 0.30, 0.20, 0.10, 0.05])[0]
    for _ in range(N_REQUESTS)
]

print(f"  {N_REQUESTS} requests, prefill={PREFILL_MS}ms/req, decode={DECODE_MS}ms/step")
print(f"  Output length distribution: mean={statistics.mean(requests):.0f}, "
      f"max={max(requests)}, p99={sorted(requests)[int(0.99*len(requests))]}")

print(f"\n  {'Batch':>6}  {'Throughput':>12}  {'Padding%':>10}  {'Padded tokens':>14}")
print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*14}")

batch_sizes = [1, 4, 8, 16, 32]
results_by_batch = {}
for bs in batch_sizes:
    res = run_static_batching(requests, bs, PREFILL_MS, DECODE_MS)
    results_by_batch[bs] = res
    print(f"  {bs:>6}  {res['throughput_tps']:>10.0f}/s  "
          f"{res['padding_fraction']:>10.1%}  {res['padded_tokens']:>14,}")

# Larger batch → more padding waste (because variance within batch is higher)
pf_batch1  = results_by_batch[1]["padding_fraction"]
pf_batch32 = results_by_batch[32]["padding_fraction"]
assert pf_batch32 >= pf_batch1, \
    f"Larger batch should have >= padding: {pf_batch1:.1%} vs {pf_batch32:.1%}"
print(f"\n  Padding grows from {pf_batch1:.1%} (batch=1) to {pf_batch32:.1%} (batch=32)")
print("  ✓ Section 3 passed — static batching throughput limited by padding waste")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Wasted GPU cycles quantified
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Wasted Decode Steps ──")
print("""
  Each padded token-step costs GPU time (memory reads, compute) for a result
  that is immediately discarded. On a real GPU:

    Wasted DRAM reads  = padded_tokens × kv_bytes_per_token
    Wasted FLOPs       = padded_tokens × 2 × d_model × seq_len

  For Llama-7B with batch=32 and 50% padding, if the decode throughput is
  10,000 real tok/s, the GPU is actually running at 20,000 token-steps/s
  but throwing away half the work.

  TODO 2: Compute wasted_pct for batch=32 scenario and confirm > 20%.
""")

res32 = results_by_batch[32]

# TODO 2: Compute wasted percentage from the simulation result
wasted_pct = res32["padding_fraction"] * 100

assert wasted_pct > 20, f"Expected >20% waste for batch=32, got {wasted_pct:.1f}%"
print(f"  Batch=32: {res32['padded_tokens']:,} padded token-steps out of "
      f"{res32['real_tokens']+res32['padded_tokens']:,} total")
print(f"  Wasted GPU work: {wasted_pct:.1f}%")
print(f"""
  WHAT THIS MEANS IN PRACTICE
    If padding_fraction = {res32['padding_fraction']:.1%}:
      GPU effective utilisation ≈ {100 - wasted_pct:.0f}% (the rest is thrown away)
      Throughput could improve {1/(1-res32['padding_fraction']):.1f}× if padding were eliminated
    → This is exactly what continuous batching achieves (Exercise 11.2)
""")
print("  ✓ Section 4 passed — padding fraction directly converts to lost throughput")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 11.1 complete!")
print("  You can now measure and explain padding waste in static batching.")
print("  The next exercise shows how continuous batching eliminates it.")
print("  Next: IV.LLM_Inference_Systems/11.Batching_Strategies/11.2_continuous_batching.py")
print("=" * 60)
