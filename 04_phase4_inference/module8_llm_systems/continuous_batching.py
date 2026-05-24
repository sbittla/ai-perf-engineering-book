#!/usr/bin/env python3
"""
continuous_batching.py  ─  Phase 4 / Module 8: Continuous Batching Simulation
benchmark_vllm_vs_hf.py  ─  (bundled)
===============================================================================

HOW TO RUN
    python continuous_batching.py
    python continuous_batching.py --exp compare   # vLLM vs HF comparison


"""

import argparse, os, sys, time, random, statistics, threading, queue
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
    choices=["all","static","continuous","compare"])
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: model step time
# ─────────────────────────────────────────────────────────────────────────────
class SimpleEngine:
    """Tiny inference engine that simulates per-step decode cost."""
    def __init__(self, ms_per_step_per_seq=2.0):
        """ms_per_step_per_seq: how many ms one decode step takes per active sequence."""
        self.ms_per_step_per_seq = ms_per_step_per_seq

    def step(self, active_seqs):
        """Simulate one decode step for `active_seqs` sequences."""
        time.sleep(self.ms_per_step_per_seq * active_seqs / 1000.0)
        return active_seqs   # tokens produced

# ─────────────────────────────────────────────────────────────────────────────
# STATIC BATCHING
# ─────────────────────────────────────────────────────────────────────────────
def exp_static():
    sep("STATIC BATCHING (HuggingFace generate() style)")
    print("""
  Requests are grouped into fixed batches.
  The batch runs until ALL requests complete (padded to max_output_len).
  New requests wait for the ENTIRE batch to finish before joining.
    """)
    engine     = SimpleEngine(ms_per_step_per_seq=0.5)
    BATCH_SIZE = 8
    N_REQUESTS = 32

    # Variable output lengths (realistic: most short, some long)
    output_lens = [random.choices([20,50,100,200], weights=[0.4,0.3,0.2,0.1])[0]
                   for _ in range(N_REQUESTS)]

    t_start      = time.perf_counter()
    total_tokens = 0
    latencies    = []
    request_idx  = 0

    while request_idx < N_REQUESTS:
        # Form a batch
        batch = output_lens[request_idx:request_idx + BATCH_SIZE]
        batch_start = time.perf_counter()
        max_len = max(batch)

        # Run max_len steps for ALL sequences (even finished ones)
        for step in range(max_len):
            engine.step(len(batch))   # always runs full batch even with padding

        tokens_in_batch = sum(batch)
        total_tokens   += tokens_in_batch

        for req_len in batch:
            # Latency = time from batch_start to when this request finishes
            # (Static: every request waits for the longest one)
            latencies.append((time.perf_counter() - batch_start) * 1000)

        request_idx += BATCH_SIZE

    total_elapsed = time.perf_counter() - t_start
    tps = total_tokens / total_elapsed

    print(f"  Requests: {N_REQUESTS}  Batch size: {BATCH_SIZE}")
    print(f"  Avg output len   : {statistics.mean(output_lens):.0f} tokens")
    print(f"  Max output len   : {max(output_lens)} tokens")
    print(f"  Total elapsed    : {total_elapsed:.2f}s")
    print(f"  Throughput       : {tps:.0f} tok/s")
    print(f"  Latency P50      : {statistics.median(latencies):.0f} ms")
    print(f"  Latency P99      : {sorted(latencies)[int(0.99*len(latencies))]:.0f} ms")
    print(f"  Wasted steps     : {sum(max(output_lens[i:i+BATCH_SIZE])*BATCH_SIZE - sum(output_lens[i:i+BATCH_SIZE]) for i in range(0,N_REQUESTS,BATCH_SIZE))} token-steps")
    return tps, statistics.median(latencies)

# ─────────────────────────────────────────────────────────────────────────────
# CONTINUOUS BATCHING
# ─────────────────────────────────────────────────────────────────────────────
def exp_continuous():
    sep("CONTINUOUS BATCHING (vLLM PagedAttention style)")
    print("""
  Completed sequences leave the batch immediately.
  New sequences join at the next step.
  GPU always has a FULL batch of real work.
    """)
    engine     = SimpleEngine(ms_per_step_per_seq=0.5)
    MAX_BATCH  = 8
    N_REQUESTS = 32

    output_lens = [random.choices([20,50,100,200], weights=[0.4,0.3,0.2,0.1])[0]
                   for _ in range(N_REQUESTS)]

    # State for each active sequence: (remaining_steps, start_time)
    active    = {}   # seq_id → (steps_remaining, start_time_ms)
    waiting   = list(range(N_REQUESTS))  # request IDs in queue
    done_lats = []
    total_tokens = 0
    seq_id_counter = 0
    t_start  = time.perf_counter()

    # Fill initial batch
    while len(active) < MAX_BATCH and waiting:
        rid  = waiting.pop(0)
        active[rid] = [output_lens[rid], time.perf_counter()]

    while active:
        # One decode step for all active sequences
        engine.step(len(active))

        # Decrement remaining steps; collect completions
        finished = []
        for rid in list(active):
            active[rid][0] -= 1
            total_tokens   += 1
            if active[rid][0] <= 0:
                lat = (time.perf_counter() - active[rid][1]) * 1000
                done_lats.append(lat)
                finished.append(rid)

        # Remove finished, admit new
        for rid in finished:
            del active[rid]
        while len(active) < MAX_BATCH and waiting:
            rid = waiting.pop(0)
            active[rid] = [output_lens[rid], time.perf_counter()]

    total_elapsed = time.perf_counter() - t_start
    tps = total_tokens / total_elapsed

    print(f"  Requests: {N_REQUESTS}  Max concurrency: {MAX_BATCH}")
    print(f"  Total elapsed    : {total_elapsed:.2f}s")
    print(f"  Throughput       : {tps:.0f} tok/s")
    print(f"  Latency P50      : {statistics.median(done_lats):.0f} ms")
    print(f"  Latency P99      : {sorted(done_lats)[int(0.99*len(done_lats))]:.0f} ms")
    print(f"  Wasted steps     : 0  (no padding needed)")
    return tps, statistics.median(done_lats)

# ─────────────────────────────────────────────────────────────────────────────
# VLLM VS HF COMPARISON (with actual models if available)
# ─────────────────────────────────────────────────────────────────────────────
def exp_compare():
    sep("SIMULATION COMPARISON: Static vs Continuous Batching")
    random.seed(42)
    s_tps, s_lat = exp_static()
    random.seed(42)
    c_tps, c_lat = exp_continuous()

    print(f"\n{'='*55}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<25}  {'Static':>12}  {'Continuous':>12}  {'Gain':>8}")
    print(f"  {'─'*25}  {'─'*12}  {'─'*12}  {'─'*8}")
    print(f"  {'Throughput (tok/s)':<25}  {s_tps:>12.0f}  {c_tps:>12.0f}  {c_tps/s_tps:>7.2f}×")
    print(f"  {'P50 Latency (ms)':<25}  {s_lat:>12.0f}  {c_lat:>12.0f}  {s_lat/c_lat:>7.2f}×")
    print(f"""
  KEY TAKEAWAY
    Continuous batching improves throughput {c_tps/s_tps:.1f}× and
    latency {s_lat/c_lat:.1f}× vs static batching on variable-length outputs.

  IN PRODUCTION (vLLM)
    vllm serve <model> \\
        --max-num-seqs 64 \\        # max concurrent sequences
        --enable-chunked-prefill \\ # split long prefills to keep decode going
        --max-model-len 4096

  MEASURE IN PRACTICE
    python benchmarks/benchmark_serving.py \\
        --backend vllm --request-rate 10 --num-prompts 500
  """)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  continuous_batching.py  ─  Phase 4 Module 8")
    print(f"{'='*60}")
    d = {"static": exp_static, "continuous": exp_continuous, "compare": exp_compare}
    if args.exp == "all":
        for fn in d.values(): fn()
    else:
        d[args.exp]()
