#!/usr/bin/env python3
"""
Appendices/E.Advanced_Capstones/E.5_disaggregated_serving.py  --  Appendix E.5
=======================================================================
Disaggregated Serving: show why separating prefill and decode pools protects
inter-token latency from long-prompt interference (Section 11.4).

Difficulty: *****  (5/5 - Expert; real 2-pool prototype is the live extension)
Est. time:  1-2 days to build the prototype
Expected ranges:
  - Single pool: a long prefill stalls other requests -> high P99 inter-token gap
  - Disaggregated: decode P99 jitter drops sharply; KV transfer adds a fixed cost
Troubleshooting:
  - No GPU needed here; this models the latency distributions you will measure.
Live run (the real capstone):
  - Stand up a prefill pool and a decode pool, transfer KV between them, and
    compare latency CDFs (single-pool vs disaggregated) under identical load.

Run:  python Appendices/E.Advanced_Capstones/E.5_disaggregated_serving.py
"""
print("=" * 70)
print("  Exercise E.5 - Disaggregated Prefill/Decode Serving (latency model)")
print("=" * 70)

import random
random.seed(0)

# A mixed stream: most requests short, a few very long prompts.
# prefill_steps is proportional to prompt length; decode is 1 step/token.
def make_stream(n=200):
    s = []
    for _ in range(n):
        if random.random() < 0.15:
            prefill = random.randint(40, 80)      # long prompt
        else:
            prefill = random.randint(2, 8)        # short prompt
        s.append(prefill)
    return s

PREFILL = make_stream()

# ---------------------------------------------------------------------------
# SECTION 1: Single pool - prefill blocks decode (head-of-line)
# ---------------------------------------------------------------------------
print("\n-- Section 1: Single Pool Inter-Token Gap --")
print("""
  On one pool, a decode step cannot run while a long prefill occupies the GPU.
  The inter-token gap for in-flight requests = 1 (their own step) + any prefill
  steps that jumped ahead of them. We record the per-step gap distribution.
""")

def percentile(xs, p):
    xs = sorted(xs)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[k]

# Each scheduling 'tick' serves one unit of work. A long prefill is a burst of
# work that delays the next decode token of everyone else.
single_gaps = []
for pf in PREFILL:
    # a decode token waits behind this prefill burst when they collide
    single_gaps.append(1 + pf)        # worst-case inter-token gap during a prefill
single_p50 = percentile(single_gaps, 50)
single_p99 = percentile(single_gaps, 99)
print(f"  Inter-token gap  P50: {single_p50:>4} steps")
print(f"  Inter-token gap  P99: {single_p99:>4} steps  <- long prompts spike the tail")
assert single_p99 > single_p50 * 3
print("  [check] single pool has a heavy P99 inter-token tail")

# ---------------------------------------------------------------------------
# SECTION 2: Disaggregated - decode pool only ever decodes
# ---------------------------------------------------------------------------
print("\n-- Section 2: Disaggregated Inter-Token Gap --")
print("""
  With prefill on a separate pool, the decode pool runs only single-token steps,
  so its inter-token gap is ~1 step regardless of how long any prompt is. The
  cost moves to a one-time KV-cache transfer between pools.
""")
disagg_gaps = [1 for _ in PREFILL]         # decode pool never blocked by prefill
disagg_p99 = percentile(disagg_gaps, 99)
print(f"  Inter-token gap  P50: {percentile(disagg_gaps,50):>4} steps")
print(f"  Inter-token gap  P99: {disagg_p99:>4} steps")
assert disagg_p99 < single_p99
print(f"  P99 jitter reduction : {single_p99/disagg_p99:.0f}x")
print("  [check] isolating prefill flattens the decode-latency tail")

# ---------------------------------------------------------------------------
# SECTION 3: The KV-transfer cost - when disaggregation is worth it
# ---------------------------------------------------------------------------
print("\n-- Section 3: KV Transfer Overhead --")
print("""
  Disaggregation only pays off if moving the KV cache between pools is cheap
  relative to the decode it protects. Fast interconnect (NVLink/RDMA) is the
  enabling condition.
""")
# KV for a 40-token prompt, Llama-3-8B GQA FP16: 40 * 128KB = 5 MB
kv_mb = 40 * 128 / 1024
for name, bw in [("NVLink ~600 GB/s", 600.0), ("RDMA/IB ~50 GB/s", 50.0),
                 ("PCIe ~25 GB/s", 25.0)]:
    t_ms = (kv_mb * 1e6) / (bw * 1e9) * 1000
    verdict = "worth it" if t_ms < 1.0 else "marginal"
    print(f"  Transfer {kv_mb:.1f} MB over {name:18}: {t_ms:6.3f} ms  ({verdict})")
nvlink_ms = (kv_mb * 1e6) / (600.0 * 1e9) * 1000
assert nvlink_ms < 1.0
print("  [check] on fast interconnect, KV transfer is sub-millisecond")

print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise E.5 complete")
print("  You modeled the latency win and the transfer cost of disaggregation.")
print("  Now build the two-pool prototype and compare real latency CDFs.")
print("=" * 70)
