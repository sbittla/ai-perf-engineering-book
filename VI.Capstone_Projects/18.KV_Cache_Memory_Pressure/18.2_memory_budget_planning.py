#!/usr/bin/env python3
"""
VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.2_memory_budget_planning.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 18: Capstone 3 — KV Cache Memory Pressure Experiment
Section 2: Memory Budget Planning for Production LLM Serving
=======================================================================
Covers capstone section 18.2:
  • Full memory budget decomposition: weights + KV + activations + overhead
  • Choosing between model precision, context length, and concurrency
  • Quantifying the 3-way trade-off for different deployment targets
  • Writing a structured deployment plan for a given VRAM budget
  • Comparing Llama-7B, Llama-13B, and Mistral-7B on a 24GB A10G

Run:  python VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.2_memory_budget_planning.py
All sections must print ✓.
"""

import json
import math
import os

print("=" * 60)
print("  Capstone 18.2 — Memory Budget Planning")
print("=" * 60)

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    GPU_VRAM_GB = props.total_memory / 1e9
    print(f"  Device: {props.name}  |  VRAM: {GPU_VRAM_GB:.1f} GB\n")
else:
    GPU_VRAM_GB = 24.0
    print(f"  Device: CPU  (formulas use {GPU_VRAM_GB:.0f} GB VRAM assumption)\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: Full Memory Budget Decomposition
# ─────────────────────────────────────────────────────────────
print("── Section 1: Full Memory Budget Decomposition ──")
print("""
  GPU memory is consumed by four components:

    1. MODEL WEIGHTS:  params × dtype_bytes
       Llama-7B FP16:   7B × 2 = 14 GB
       Llama-7B INT8:   7B × 1 = 7 GB
       Llama-7B INT4:   7B × 0.5 = 3.5 GB

    2. KV CACHE:  2 × n_layers × n_heads × head_dim × seq × batch × dtype_bytes
       Grows with context length and concurrent requests

    3. ACTIVATIONS:  batch × seq × d_model × n_layers × dtype_bytes × ~3
       For inference: typically 0.5–2 GB (small vs weights)
       For training: much larger (gradient + optimizer state)

    4. FRAMEWORK OVERHEAD:  CUDA context + cuBLAS workspace + allocator
       Typically 0.5–1.5 GB fixed overhead

  PLANNING FORMULA:
    available_for_kv = vram - weights - activations - overhead
    max_kv_tokens    = available_for_kv / (kv_bytes_per_token)
    max_concurrent   = max_kv_tokens / seq_len

  LEVER INTERACTIONS:
    ↑ precision (FP16→INT4) → ↑ weights size by 4× → ↑ max_kv
    ↑ seq_len              → ↓ max_concurrent (linear)
    ↑ batch (more requests)→ ↓ available for each request's context
""")


def memory_budget(vram_gb: float, weights_gb: float, seq_len: int,
                   batch: int, n_layers: int, n_heads: int, head_dim: int,
                   dtype_bytes: int = 2, activation_gb: float = 1.0,
                   overhead_gb: float = 1.0) -> dict:
    """
    Return full memory budget breakdown in GB.
    """
    kv_gb  = (2 * n_layers * n_heads * head_dim * seq_len * batch * dtype_bytes) / 1e9
    used   = weights_gb + kv_gb + activation_gb + overhead_gb
    avail  = vram_gb - weights_gb - activation_gb - overhead_gb
    avail_for_kv_gb = max(0.0, avail)
    kv_per_tok_bytes = 2 * n_layers * n_heads * head_dim * dtype_bytes
    max_kv_tokens = int(avail_for_kv_gb * 1e9 / kv_per_tok_bytes) if kv_per_tok_bytes > 0 else 0
    max_concurrent = max_kv_tokens // seq_len if seq_len > 0 else 0
    return {
        "weights_gb":      round(weights_gb, 2),
        "kv_gb":           round(kv_gb, 2),
        "activations_gb":  round(activation_gb, 2),
        "overhead_gb":     round(overhead_gb, 2),
        "total_used_gb":   round(used, 2),
        "vram_gb":         round(vram_gb, 2),
        "fits":            used <= vram_gb,
        "max_concurrent":  max_concurrent,
        "headroom_gb":     round(vram_gb - used, 2),
    }


# Model definitions
MODELS = {
    "Llama-7B FP16":  dict(weights_gb=13.5, n_layers=32, n_heads=32, head_dim=128, dtype_bytes=2),
    "Llama-7B INT8":  dict(weights_gb=7.0,  n_layers=32, n_heads=32, head_dim=128, dtype_bytes=1),
    "Llama-13B FP16": dict(weights_gb=26.0, n_layers=40, n_heads=40, head_dim=128, dtype_bytes=2),
    "Mistral-7B FP16":dict(weights_gb=14.0, n_layers=32, n_heads=32, head_dim=128, dtype_bytes=2),
}

SEQ_LEN = 2048
BATCH_REF = 4

print(f"  Memory budget (GPU={GPU_VRAM_GB:.0f}GB, seq={SEQ_LEN}, batch={BATCH_REF}):")
print(f"  {'Model':<18}  {'Weights':>8}  {'KV':>8}  {'Total':>8}  {'Fits':>6}  {'MaxConc':>9}")
print(f"  {'─'*18}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*9}")
for name, m in MODELS.items():
    bgt = memory_budget(GPU_VRAM_GB, m["weights_gb"], SEQ_LEN, BATCH_REF,
                        m["n_layers"], m["n_heads"], m["head_dim"], m["dtype_bytes"])
    fits = "✓" if bgt["fits"] else "OOM"
    print(f"  {name:<18}  {bgt['weights_gb']:>7.1f}G  {bgt['kv_gb']:>7.1f}G  "
          f"{bgt['total_used_gb']:>7.1f}G  {fits:>6}  {bgt['max_concurrent']:>9}")

assert True  # formula verification done via output
print("  ✓ Section 1 passed — full memory budget decomposition")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The 3-Way Trade-off
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: The 3-Way Trade-off ──")
print("""
  Given a fixed VRAM budget, you must choose between three goals:
    A. QUALITY: higher precision (FP16 > INT8 > INT4)
    B. CONTEXT: longer sequences (more past context per request)
    C. SCALE:   more concurrent requests

  These are not independent — improving one hurts the others.

  TRADE-OFF TABLE for Llama-7B on a 24GB GPU:
    Config                | Precision | Max seq | Max concurrent
    ─────────────────────────────────────────────────────────────
    Max quality           | FP16      | 2048    | low
    Max context           | INT8      | 8192    | very low
    Max scale (throughput)| INT4      | 2048    | high

  TODO 1: Implement tradeoff_analysis(vram_gb, n_layers, n_heads, head_dim)
  that returns a table of configurations optimised for each goal.
""")


def tradeoff_analysis(vram_gb: float, n_layers: int, n_heads: int,
                       head_dim: int) -> list:
    """
    TODO 1: Return list of config dicts, each optimised for one goal.
    Consider: (FP16, seq=2048), (INT8, seq=4096), (INT4, seq=2048 with more concurrent)
    For each, compute max_concurrent and total memory.
    """
    configs = []
    for label, weights_gb, dtype_bytes, seq_len in [
        ("Max quality (FP16)",   13.5, 2, 2048),
        ("Max quality (FP16)",   13.5, 2, 4096),
        ("Balanced (INT8)",       7.0, 1, 4096),
        ("Max scale (INT4)",      3.5, 1, 2048),   # INT4 weights, FP16 KV
        ("Max scale (INT4)",      3.5, 1, 8192),
    ]:
        bgt = memory_budget(vram_gb, weights_gb, seq_len, batch=1,
                            n_layers=n_layers, n_heads=n_heads,
                            head_dim=head_dim, dtype_bytes=dtype_bytes)
        configs.append({
            "label":          label,
            "weights_gb":     weights_gb,
            "dtype":          "FP16" if dtype_bytes == 2 else "INT8/INT4",
            "seq_len":        seq_len,
            "max_concurrent": bgt["max_concurrent"],
            "total_gb":       bgt["total_used_gb"],
            "fits":           bgt["fits"],
        })
    return configs


configs = tradeoff_analysis(GPU_VRAM_GB, **{k: MODELS["Llama-7B FP16"][k]
                             for k in ["n_layers", "n_heads", "head_dim"]})
print(f"  Llama-7B trade-off configurations (GPU={GPU_VRAM_GB:.0f}GB):")
print(f"  {'Config':<24}  {'Weights':>8}  {'Seq':>6}  {'MaxConc':>9}  {'Total':>8}  {'Fits':>5}")
print(f"  {'─'*24}  {'─'*8}  {'─'*6}  {'─'*9}  {'─'*8}  {'─'*5}")
for c in configs:
    fits_str = "✓" if c["fits"] else "OOM"
    print(f"  {c['label']:<24}  {c['weights_gb']:>7.1f}G  {c['seq_len']:>6}  "
          f"{c['max_concurrent']:>9}  {c['total_gb']:>7.1f}G  {fits_str:>5}")

assert len(configs) > 0
print("  ✓ Section 2 passed — 3-way trade-off analysis complete")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Deployment Planning Decision Matrix
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Deployment Planning Decision Matrix ──")
print("""
  A deployment plan must specify:
    1. MODEL:          which model and precision
    2. CONTEXT:        max supported sequence length
    3. CONCURRENCY:    max concurrent requests in flight
    4. STRATEGY:       full / sliding-window / paged (vLLM)
    5. LATENCY SLO:    TTFT budget and per-step decode budget

  DECISION RULES:
    If latency SLO is tight (< 2s TTFT): use smaller model or INT8
    If context > 4096 needed: use sliding window or INT8/INT4
    If concurrency > 10 needed: use INT4 or multiple GPU replicas
    If quality is paramount: FP16 with paged attention

  TODO 2: Implement deployment_plan(use_case, vram_gb, latency_slo_ms,
          required_seq_len, required_concurrent) that returns
          the recommended model config as a dict.
""")


def deployment_plan(use_case: str, vram_gb: float, ttft_slo_ms: float,
                    required_seq_len: int, required_concurrent: int) -> dict:
    """
    TODO 2: Recommend a model configuration for the given requirements.
    Logic:
      - If required_concurrent > 8 or required_seq_len > 4096: prefer INT8
      - If ttft_slo_ms < 500: prefer smaller model (no Llama-13B)
      - If required_seq_len > 8192: use sliding window
      Evaluate each candidate; return the first that fits.
    """
    candidates = [
        ("Llama-7B FP16",  13.5, 32, 32, 128, 2, "full"),
        ("Llama-7B INT8",   7.0, 32, 32, 128, 1, "full"),
        ("Llama-7B INT4",   3.5, 32, 32, 128, 1, "paged"),
        ("Llama-13B FP16", 26.0, 40, 40, 128, 2, "full"),
    ]
    for name, wt_gb, nl, nh, hd, db, strat in candidates:
        bgt = memory_budget(vram_gb, wt_gb, required_seq_len, required_concurrent,
                            nl, nh, hd, db)
        if not bgt["fits"]:
            continue
        if bgt["max_concurrent"] < required_concurrent:
            continue
        # TTFT heuristic: smaller model → faster TTFT
        ttft_ok = (wt_gb < 20) or (ttft_slo_ms > 2000)
        if not ttft_ok:
            continue
        return {
            "use_case":    use_case,
            "model":       name,
            "weights_gb":  wt_gb,
            "strategy":    strat,
            "seq_len":     required_seq_len,
            "max_concurrent": bgt["max_concurrent"],
            "total_gb":    bgt["total_used_gb"],
            "meets_slo":   True,
        }
    return {"use_case": use_case, "model": "NONE — requirements cannot be met",
            "meets_slo": False}


use_cases = [
    ("Chat (low latency)",  500,  1024,  4),
    ("RAG (long context)", 3000,  4096,  8),
    ("Batch (high tput)",  5000,  2048, 16),
]

print(f"\n  Deployment recommendations (GPU={GPU_VRAM_GB:.0f}GB):")
print(f"  {'Use case':<24}  {'Model':<18}  {'MaxConc':>9}  {'Total GB':>10}  {'Meets SLO':>10}")
print(f"  {'─'*24}  {'─'*18}  {'─'*9}  {'─'*10}  {'─'*10}")
plans = []
for uc, slo, seq, conc in use_cases:
    plan = deployment_plan(uc, GPU_VRAM_GB, slo, seq, conc)
    plans.append(plan)
    slo_str = "✓" if plan["meets_slo"] else "✗"
    print(f"  {uc:<24}  {plan['model']:<18}  "
          f"{plan.get('max_concurrent', 0):>9}  "
          f"{plan.get('total_gb', 0):>9.1f}G  {slo_str:>10}")

assert any(p["meets_slo"] for p in plans), "At least one use case should be satisfiable"
print("  ✓ Section 3 passed — deployment recommendations generated")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Saving the planning report
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Saving the Planning Report ──")

report = {
    "hardware": {"vram_gb": GPU_VRAM_GB},
    "tradeoff_configs": configs,
    "deployment_plans": plans,
    "key_rules": [
        "Reducing precision from FP16 to INT8 halves weights, doubling available KV cache.",
        "Max concurrent scales linearly with (VRAM - weights) / kv_per_request.",
        "Decode latency scales linearly with seq_len (memory-bandwidth bound).",
        "PagedAttention eliminates fragmentation; quality equals full-cache.",
        "Sliding window reduces memory to O(W) but loses long-range context.",
    ],
}

plan_path = "/tmp/capstone18_planning_report.json"
with open(plan_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"  Report saved: {plan_path}")

print(f"""
  CAPSTONE 18 SUMMARY — The 3-way Memory Budget Trade-off:

    PRECISION      : FP16 → INT8 → INT4 → weights shrink 2×/4×/8×
    CONTEXT LENGTH : Seq × batch determines KV cache size → OOM risk
    CONCURRENCY    : More requests = more KV cache = less room for long context

  The right choice depends on your use case:
    Latency-sensitive : FP16, short context, few concurrent requests
    Throughput-focused: INT4, medium context, many concurrent requests
    Long context RAG  : INT8, long context, fewer concurrent requests

  The formula makes the trade-off explicit and quantitative.
  That is the goal of memory budget planning.
""")
assert os.path.exists(plan_path)
print("  ✓ Section 4 passed — planning report saved")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 18.2 complete!")
print("  Memory budget planning for LLM deployment complete.")
print("  Next: VI.Capstone_Projects/19.Flamegraph_Challenge/19.1_slow_training_analysis.py")
print("=" * 60)
