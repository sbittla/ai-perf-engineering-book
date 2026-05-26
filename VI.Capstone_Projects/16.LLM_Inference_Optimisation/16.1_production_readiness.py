"""
Exercise 16.1 — From Research to Production: The AI Deployment Gap

Chapter 16 (Background): What Changes When a Model Goes Live
Book: AI Systems Performance Engineering

Run:
    python 16.1_production_readiness.py

What this exercise does:
  1. Contrasts research code vs. production code across six dimensions.
  2. Models the three production constraints: Latency SLA, Throughput Target, Cost Budget.
  3. Calculates cost-per-million-tokens and shows how throughput halves cost.
  4. Lists and diagnoses the five most common production failure modes (OOM, DataLoader,
     P99 spikes, low TPS at 100% GPU, memory leaks).
  5. Presents a 24-item production readiness self-assessment checklist across
     Latency, Throughput, Memory, Precision, Observability, and Reliability.
  6. Maps each of the four capstone chapters to a specific failure mode and skill.
  7. Calculates KV cache memory per request for LLaMA-7B through LLaMA-70B.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then explore the capstone chapters in Part VI for hands-on work.
"""

import sys
import math

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: The Research-to-Production Gap
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: The Research-to-Production Gap")
print("=" * 60)

print("""
Research code and production code solve different problems:

  Research:                          Production:
  ─────────────────────────────────────────────────────────
  Single GPU, often                  Multi-GPU, multi-node
  FP32 by default                    FP16 / BF16 / INT8 / FP8
  Fixed batch size                   Dynamic batching + request queuing
  DataLoader with shuffle            Preprocessed, cached, streaming
  Minimise loss                      Minimise cost under latency SLA
  Correctness only                   Correctness + performance + reliability
  Run once                           Run 24/7 with zero-downtime deploys
  Fail is OK (exception → fix)       Fail is money lost + user impact
  Metrics: accuracy                  Metrics: TTFT, TPS, P99, cost/$M tokens

The skills to bridge this gap are what this book teaches.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Three Production Constraints
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: The Three Production Constraints")
print("=" * 60)

print("""
Every production AI system is simultaneously constrained by three forces:

CONSTRAINT 1 — Latency SLA
  Users tolerate specific wait times. Violating them triggers escalations,
  SLA penalties, and churn. Common targets:
    Chatbot first-token latency  (TTFT) : P99 < 2 seconds
    Autocomplete response time          : P99 < 100 milliseconds
    Batch job turnaround                : P99 < 4 hours

CONSTRAINT 2 — Throughput Target
  The system must sustain a peak request rate without degradation.
  tokens_per_second = total_tokens_generated / wall_clock_seconds
  Capacity planning: peak_rps × avg_tokens_per_request = min tokens/s needed

CONSTRAINT 3 — Cost Budget
  GPU time is expensive. Common unit: cost per million output tokens.
  cost_per_M_tokens = (hourly_GPU_cost × 3600) / (tokens_per_second × 1e6)
""")

# Compute cost example
hourly_cost_h100 = 3.00   # USD/hr, ballpark A100/H100 on-demand
tokens_per_second = 150.0  # typical vLLM LLaMA-70B throughput

cost_per_m = (hourly_cost_h100 * 3600) / (tokens_per_second * 1e6) * 1e6
print(f"Example: H100 at ${hourly_cost_h100:.2f}/hr, {tokens_per_second:.0f} tokens/sec")
print(f"  cost per 1M tokens = ${cost_per_m:.2f}")
print()

# Show how throughput affects cost
print("  How throughput affects cost (same GPU, same hourly rate):")
print(f"  {'Tokens/sec':>12}  {'Cost / 1M tokens':>18}  {'vs baseline'}")
print(f"  {'-'*12}  {'-'*18}  {'-'*15}")
baseline = 50.0
for tps in [50, 100, 150, 300, 600]:
    c = (hourly_cost_h100 * 3600) / (tps * 1e6) * 1e6
    ratio = baseline / tps
    print(f"  {tps:>12.0f}  ${c:>17.2f}  {ratio:>8.2f}× cost")

print()
print("2× throughput = 2× cost reduction on the same hardware.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Common Failure Modes
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: The Five Most Common Production Failures")
print("=" * 60)

failures = [
    (
        "OOM on first large batch",
        "FP32 residuals, no gradient checkpointing, KV cache unbounded",
        "Switch to BF16, enable KV eviction, reduce max_batch_tokens",
        "Chapter 16, 18"
    ),
    (
        "DataLoader starves GPU",
        "Num_workers=0, data on slow NFS, PIL decode on single thread",
        "Set num_workers = 4-8, prefetch_factor = 2, use webdataset",
        "Chapter 17"
    ),
    (
        "P99 >> P50 latency",
        "GC pauses, Python GIL contention, cuDNN first-call auto-tune",
        "Pre-warm model, disable GC during serving, use torch.compile",
        "Chapter 16"
    ),
    (
        "GPU 100%, low TPS",
        "Tiny batch size, memory-bound decode, suboptimal attention",
        "Increase batch size, use FlashAttention, quantise weights",
        "Chapter 16, 18"
    ),
    (
        "Memory leak over time",
        "Unreleased activations, growing KV cache, Python ref cycles",
        "Profile with torch.cuda.memory_stats(), set cache eviction",
        "Chapter 18"
    ),
]

print(f"  {'Symptom':<32} {'Root Cause'}")
print(f"  {'-'*32} {'-'*45}")
for symptom, cause, fix, chapter in failures:
    print(f"  {symptom:<32} {cause}")
    print(f"  {'':32} Fix: {fix}")
    print(f"  {'':32} See: {chapter}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Production Readiness Self-Assessment
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: Production Readiness Self-Assessment Checklist")
print("=" * 60)

print("""
Before deploying an AI system, run this checklist. A 'NO' is not a blocker
by itself — it is a tracked risk that must be documented and mitigated.
""")

dimensions = [
    ("Latency", [
        ("TTFT measured under peak concurrency (not synthetic single-request)",),
        ("P99 latency < SLA target with at least 50 samples",),
        ("Model pre-warmed before traffic hits (cuDNN auto-tune complete)",),
        ("Latency budget allocated across tokenise/prefill/decode/network",),
    ]),
    ("Throughput", [
        ("Sustained tokens/sec measured, not burst peak",),
        ("Batching strategy validated (static vs continuous vs dynamic)",),
        ("DataLoader not the bottleneck (GPU util > 80% during training)",),
        ("Load test run at 110% of expected peak",),
    ]),
    ("Memory", [
        ("Max batch size determined by OOM test, not guessed",),
        ("KV cache size computed: 2 × layers × heads × d_head × seq × precision",),
        ("Memory growth monitored over 10,000+ requests (no leak)",),
        ("Gradient checkpointing enabled if training memory is tight",),
    ]),
    ("Precision", [
        ("Using BF16 or FP16 (not FP32) for inference weights",),
        ("Autocast enabled for inference if using PyTorch",),
        ("Accuracy regression tested after any precision change",),
        ("INT8 / FP8 quantisation impact on accuracy is documented",),
    ]),
    ("Observability", [
        ("TTFT, TPS, P50/P99 latency exported to metrics system",),
        ("GPU utilisation and memory tracked per-request",),
        ("Alerting configured for P99 > 2× P50 (tail anomaly)",),
        ("Flamegraph capture path tested (py-spy or Nsight available)",),
    ]),
    ("Reliability", [
        ("OOM handled gracefully (returns error, does not crash process)",),
        ("Timeout enforced on decode (unbounded token generation capped)",),
        ("Health check endpoint verified under load",),
        ("Rollback procedure documented and tested",),
    ]),
]

total_checks = 0
for dim, checks in dimensions:
    print(f"  [{dim}]")
    for (check,) in checks:
        print(f"    [ ] {check}")
        total_checks += 1
    print()

print(f"Total: {total_checks} checks across {len(dimensions)} dimensions.")
print("Target: all checks GREEN before production launch.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Capstone Chapter Map
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: How the Four Capstones Map to Production Skills")
print("=" * 60)

capstones = [
    ("Chapter 16", "LLM Inference Optimisation",
     "OOM on large sequences",
     "FlashAttention, KV cache tuning, quantisation",
     "Latency, Memory, Precision"),
    ("Chapter 17", "DataLoader Bottleneck Hunt",
     "Training throughput < 40% of GPU peak",
     "Worker tuning, prefetch, format conversion",
     "Throughput, Observability"),
    ("Chapter 18", "KV Cache Memory Pressure",
     "Memory grows unboundedly with request volume",
     "Eviction policies, paged attention, measurement",
     "Memory, Reliability"),
    ("Chapter 19", "Flamegraph Challenge",
     "P99 latency 5× worse than P50",
     "py-spy, perf, eBPF, identify root cause",
     "Observability, Reliability"),
]

for ch, title, failure, skill, dimensions_covered in capstones:
    print(f"  {ch}: {title}")
    print(f"    Failure reproduced : {failure}")
    print(f"    Skill developed    : {skill}")
    print(f"    Checklist dims     : {dimensions_covered}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: KV Cache Size Calculator
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: KV Cache Size Calculator")
print("=" * 60)

print("""
The KV cache stores past key and value projections for each request.
Its size determines how many concurrent requests can be in flight.

  kv_bytes = 2 × num_layers × num_heads × head_dim × max_seq_len × bytes_per_element
""")

models_kv = [
    ("LLaMA-7B",   32, 32, 128, 4096),
    ("LLaMA-13B",  40, 40, 128, 4096),
    ("LLaMA-70B",  80, 64, 128, 4096),
    ("Mistral-7B", 32, 8,  128, 32768),  # GQA: 8 KV heads
]

bpe = 2  # FP16

print(f"  {'Model':<16} {'Layers':<8} {'KV Heads':<10} {'Max Seq':<10} {'KV Cache / req':<20} {'Batch 32'}")
print(f"  {'-'*16} {'-'*8} {'-'*10} {'-'*10} {'-'*20} {'-'*15}")
for name, layers, kv_heads, head_dim, max_seq in models_kv:
    kv_bytes = 2 * layers * kv_heads * head_dim * max_seq * bpe
    kv_mb = kv_bytes / 1024 / 1024
    batch32_gb = kv_mb * 32 / 1024
    print(f"  {name:<16} {layers:<8} {kv_heads:<10} {max_seq:<10} {kv_mb:>12.1f} MB   {batch32_gb:>8.2f} GB")

print()
print("For LLaMA-70B at batch=32: KV cache alone needs ~32+ GB.")
print("On an 80 GB H100, that leaves ~48 GB for the 140 GB model weights —")
print("meaning LLaMA-70B requires tensor parallelism across 2+ GPUs in production.")
print()
print("Explore next: VI.Capstone_Projects/16.LLM_Inference_Optimisation/")
print("              VI.Capstone_Projects/17.DataLoader_Bottleneck_Hunt/")
print("              VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/")
print("              VI.Capstone_Projects/19.Flamegraph_Challenge/")
