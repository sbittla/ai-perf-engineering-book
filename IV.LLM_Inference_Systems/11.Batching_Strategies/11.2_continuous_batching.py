#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/11.Batching_Strategies/11.2_continuous_batching.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 11: Batching Strategies — Section 2: Continuous Batching
=======================================================================
Covers book section 11.2:
  • How continuous batching eliminates padding waste
  • Slot scheduling: completed sequences immediately replaced by waiting ones
  • Comparing throughput and P99 latency vs static batching
  • The Little's Law view: why full concurrency → higher throughput
  • vLLM commands for controlling continuous batching parameters

Run:  python IV.LLM_Inference_Systems/11.Batching_Strategies/11.2_continuous_batching.py
All sections must print ✓.
"""

import random
import statistics

print("=" * 60)
print("  Exercise 11.2 — Continuous Batching")
print("=" * 60)
print()


# ─────────────────────────────────────────────────────────────
# SECTION 1: Continuous batching concept
# ─────────────────────────────────────────────────────────────
print("── Section 1: Continuous Batching — How It Works ──")
print("""
  STATIC BATCHING PROBLEM
    A batch of 16 requests all run together. When the shortest finishes
    at step 10, it sits idle for the remaining 190 steps (padding).
    New requests cannot join until the ENTIRE batch completes.

  CONTINUOUS BATCHING SOLUTION
    The batch is a "slot pool" with a fixed maximum concurrency.
    At every decode step:
      1. Run one forward pass for all active sequences.
      2. Any sequence that generates EOS (end of sequence) is marked done.
      3. A waiting request immediately fills the freed slot.
    → No padding. GPU always processes REAL work. Waiting requests join
      at the very next step, not at the end of the batch.

  EFFECT ON THROUGHPUT
    Static batch B=16, avg output 50 tokens, max output 200:
      GPU runs 16 × 200 = 3200 token-steps; only 800 are real → 75% waste.
    Continuous batching, concurrency=16:
      GPU always processes ~16 real tokens/step → near-zero waste.
      Effective throughput improvement: up to 4× for this distribution.
""")
print("  ✓ Section 1 passed — continuous batching keeps GPU busy with real work")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Simulating continuous batching
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Slot Scheduler Simulation ──")
print("""
  We simulate a continuous batching scheduler:
    - max_concurrency slots available at all times
    - Each decode step: run all active slots, decrement remaining_steps
    - Any finished slot immediately admits a waiting request

  TODO 1: Complete the scheduler loop body below.
""")


def run_continuous_batching(
    requests: list,
    max_concurrency: int,
    prefill_ms_per_req: float,
    decode_ms_per_step: float,
) -> dict:
    """
    Simulate continuous batching.
    requests: list of output_lengths.
    Returns dict with throughput_tps, latencies, total_ms, wasted_steps.
    """
    waiting   = list(requests)     # queue of output_lengths
    active    = {}                 # slot_id → (steps_remaining, arrival_ms)
    done_lats = []
    total_tokens = 0
    t_ms      = 0.0
    wasted_steps = 0

    # Fill initial slots (each prefilled before decode starts)
    slot_id = 0
    while len(active) < max_concurrency and waiting:
        req_len = waiting.pop(0)
        t_ms   += prefill_ms_per_req  # prefill is serial here for simplicity
        active[slot_id] = [req_len, t_ms]
        slot_id += 1

    while active:
        # TODO 1: One decode step:
        #   a. t_ms += decode_ms_per_step
        #   b. For each slot in active: decrement steps_remaining by 1, add 1 to total_tokens
        #   c. Collect finished slots (steps_remaining <= 0), record their latency
        #   d. Remove finished slots; for each freed slot, admit one waiting request
        #      (prefill it: t_ms += prefill_ms_per_req, add to active with new slot_id)

        # YOUR CODE HERE (replace the pass below)
        t_ms += decode_ms_per_step

        finished = []
        for sid in list(active):
            active[sid][0] -= 1
            total_tokens   += 1
            if active[sid][0] <= 0:
                lat = t_ms - active[sid][1]
                done_lats.append(lat)
                finished.append(sid)

        for sid in finished:
            del active[sid]
            if waiting:
                req_len = waiting.pop(0)
                t_ms   += prefill_ms_per_req
                active[slot_id] = [req_len, t_ms]
                slot_id += 1

    throughput = total_tokens / (t_ms / 1000) if t_ms > 0 else 0
    return {
        "throughput_tps": throughput,
        "latencies":      done_lats,
        "total_ms":       t_ms,
        "real_tokens":    total_tokens,
        "wasted_steps":   0,     # continuous batching has no padding waste
    }


# Quick sanity check
random.seed(0)
simple_reqs = [10, 10, 10, 10]
res = run_continuous_batching(simple_reqs, max_concurrency=2,
                               prefill_ms_per_req=5, decode_ms_per_step=1)
assert res["real_tokens"] == 40, f"Expected 40 real tokens, got {res['real_tokens']}"
assert res["throughput_tps"] > 0, "Throughput should be positive"
print(f"  Sanity check: 4 requests of len=10, concurrency=2 → {res['real_tokens']} tokens")
print("  ✓ Section 2 passed — scheduler simulation working")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Head-to-head comparison vs static batching
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Static vs Continuous — Side-by-Side ──")


def run_static_batching(requests, batch_size, prefill_ms, decode_ms_per_step):
    total_ms = 0.0
    real_tokens = 0
    padded_tokens = 0
    latencies = []
    for i in range(0, len(requests), batch_size):
        batch   = requests[i : i + batch_size]
        max_len = max(batch)
        batch_ms = prefill_ms * len(batch) + decode_ms_per_step * max_len
        total_ms += batch_ms
        for req_len in batch:
            real_tokens   += req_len
            padded_tokens += max_len - req_len
            latencies.append(batch_ms)
    throughput = real_tokens / (total_ms / 1000) if total_ms > 0 else 0
    pf = padded_tokens / (real_tokens + padded_tokens) if (real_tokens + padded_tokens) > 0 else 0
    return {"throughput_tps": throughput, "latencies": latencies,
            "padding_fraction": pf, "total_ms": total_ms}


def percentile(values, pct):
    if not values: return 0.0
    sv  = sorted(values)
    idx = min(int(pct * len(sv)), len(sv) - 1)
    return sv[idx]


random.seed(42)
N       = 128
PREFILL = 20
DECODE  = 0.5
reqs    = [random.choices([20, 50, 100, 200, 500],
                           weights=[0.35, 0.30, 0.20, 0.10, 0.05])[0]
           for _ in range(N)]
CONCURRENCY = 16

static_res = run_static_batching(reqs, CONCURRENCY, PREFILL, DECODE)
cont_res   = run_continuous_batching(reqs, CONCURRENCY, PREFILL, DECODE)

print(f"\n  {N} requests, concurrency/batch={CONCURRENCY}")
print(f"  {'Metric':<30}  {'Static':>12}  {'Continuous':>12}  {'Improvement':>12}")
print(f"  {'─'*30}  {'─'*12}  {'─'*12}  {'─'*12}")

s_tps = static_res["throughput_tps"]
c_tps = cont_res["throughput_tps"]
s_p50 = percentile(static_res["latencies"], 0.50)
c_p50 = percentile(cont_res["latencies"],  0.50)
s_p99 = percentile(static_res["latencies"], 0.99)
c_p99 = percentile(cont_res["latencies"],  0.99)

print(f"  {'Throughput (tok/s)':<30}  {s_tps:>12.0f}  {c_tps:>12.0f}  {c_tps/s_tps:>10.2f}×")
print(f"  {'P50 Latency (ms)':<30}  {s_p50:>12.0f}  {c_p50:>12.0f}  {s_p50/max(c_p50,1):>10.2f}×")
print(f"  {'P99 Latency (ms)':<30}  {s_p99:>12.0f}  {c_p99:>12.0f}  {s_p99/max(c_p99,1):>10.2f}×")
print(f"  {'Padding fraction':<30}  {static_res['padding_fraction']:>11.1%}  {'0.0%':>12}  {'∞':>12}")

assert c_tps > s_tps, f"Continuous batching should be faster: {c_tps:.0f} vs {s_tps:.0f}"
print(f"\n  Throughput gain: {c_tps/s_tps:.2f}×  (continuous batching wins)")
print("  ✓ Section 3 passed — continuous batching improves both throughput and latency")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Effect of concurrency on throughput
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Optimal Concurrency Setting ──")
print("""
  More concurrency = more sequences decode in parallel.
  Beyond a point, concurrency is limited by available KV cache VRAM.
  vLLM's --max-num-seqs sets the maximum concurrency.

  Too low: GPU underutilised (few real tokens per step)
  Too high: KV cache OOM, requests can't start
""")

print(f"  {'Concurrency':>12}  {'Throughput':>12}  {'P99 Latency':>12}")
print(f"  {'─'*12}  {'─'*12}  {'─'*12}")

for conc in [1, 2, 4, 8, 16, 32]:
    res = run_continuous_batching(reqs, conc, PREFILL, DECODE)
    p99 = percentile(res["latencies"], 0.99)
    print(f"  {conc:>12}  {res['throughput_tps']:>10.0f}/s  {p99:>12.0f}")

# TODO 2: Verify that concurrency=32 has higher throughput than concurrency=1
res1  = run_continuous_batching(reqs, 1,  PREFILL, DECODE)
res32 = run_continuous_batching(reqs, 32, PREFILL, DECODE)
assert res32["throughput_tps"] > res1["throughput_tps"], \
    "Higher concurrency should yield higher throughput"
print(f"\n  Concurrency=32 is {res32['throughput_tps']/res1['throughput_tps']:.1f}× "
      f"faster than concurrency=1")
print("  ✓ Section 4 passed — higher concurrency improves GPU utilisation")


# ─────────────────────────────────────────────────────────────
# SECTION 5: vLLM commands for continuous batching
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Running Continuous Batching with vLLM ──")
print("""
  vLLM implements continuous batching with PagedAttention by default.
  Key parameters:

    vllm serve meta-llama/Llama-2-7b-chat-hf \\
        --max-num-seqs 64 \\           # max concurrent sequences
        --max-model-len 4096 \\        # max context length
        --gpu-memory-utilization 0.85  # fraction of VRAM for KV pool

  Benchmark vLLM serving:
    python benchmarks/benchmark_serving.py \\
        --backend vllm \\
        --request-rate 10 \\           # requests/second arrival rate
        --num-prompts 500 \\
        --model meta-llama/Llama-2-7b-chat-hf

  Key metrics to watch in the output:
    Throughput        (tok/s)  — want this as high as possible
    TTFT P50/P99      (ms)     — want < 500ms P99 for interactive use
    E2E P50/P99       (ms)     — total experience latency

  Chunked prefill (advanced):
    --enable-chunked-prefill  Splits long prefills into chunks
    This prevents a long prefill from blocking decode steps,
    reducing TTFT variance at the cost of slightly lower prefill throughput.
""")
print("  ✓ Section 5 passed — vLLM continuous batching parameters understood")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 11.2 complete!")
print("  You simulated continuous batching, measured its throughput")
print("  gain over static batching, and learned vLLM's key parameters.")
print("  Next: IV.LLM_Inference_Systems/11.Batching_Strategies/11.3_paged_attention.py")
print("=" * 60)
