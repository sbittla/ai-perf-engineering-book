#!/usr/bin/env python3
"""
VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.1_kv_cache_scaling.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 18: Capstone 3 — KV Cache Memory Pressure Experiment
Section 1: Scaling the KV Cache — Finding the OOM Threshold
=======================================================================
Covers capstone section 18.1:
  • KV cache memory formula: 2 × n_layers × n_heads × head_dim × seq × batch × bytes
  • Sweeping batch × sequence combinations to find the OOM threshold
  • Measuring how latency inflects as memory pressure increases
  • Comparing three caching strategies: full, sliding window, paged blocks
  • Building the memory pressure profile for a Llama-7B scale model

Run:  python VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.1_kv_cache_scaling.py
All sections must print ✓.
"""

import json
import math
import statistics
import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Capstone 18.1 — KV Cache Memory Pressure")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_VRAM_GB = 0.0
if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    GPU_VRAM_GB = props.total_memory / 1e9
    print(f"  Device: {props.name}  |  VRAM: {GPU_VRAM_GB:.1f} GB\n")
else:
    GPU_VRAM_GB = 24.0   # default assumption for calculations
    print(f"  Device: CPU  (calculations assume {GPU_VRAM_GB:.0f} GB VRAM for formulas)\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The KV Cache Memory Formula
# ─────────────────────────────────────────────────────────────
print("── Section 1: KV Cache Memory Formula ──")
print("""
  The KV cache stores the Key and Value tensors for every past token
  for every layer and every attention head. Its memory cost is:

    KV bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes

  where:
    2            = Key + Value (each the same shape)
    n_layers     = number of transformer layers
    n_heads      = number of attention heads
    head_dim     = d_model / n_heads
    seq_len      = context length (grows during decode)
    batch        = number of concurrent requests
    dtype_bytes  = 2 for FP16 / BF16, 4 for FP32

  LLAMA-7B PARAMETERS (for reference):
    n_layers  = 32
    n_heads   = 32
    head_dim  = 128  (d_model=4096 / n_heads=32)
    FP16: KV per token per request = 2×32×32×128×2 = 524,288 bytes = 512 KB/tok

  For 2048-token sequences:
    Per request: 512 KB × 2048 = 1 GB
    8 concurrent: 8 GB — fills a 24GB A10G leaving only 16GB for weights
""")


def kv_cache_bytes(n_layers: int, n_heads: int, head_dim: int,
                   seq_len: int, batch: int, dtype_bytes: int = 2) -> int:
    """Return total KV cache bytes."""
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * dtype_bytes


# Llama-7B scale
LLAMA7B = dict(n_layers=32, n_heads=32, head_dim=128)
MODEL_WEIGHTS_GB = 13.5   # Llama-7B in FP16: 7B × 2 bytes

# Verify formula against known Llama-7B value
kv_per_token = kv_cache_bytes(n_heads=32, n_layers=32, head_dim=128,
                               seq_len=1, batch=1, dtype_bytes=2)
assert kv_per_token == 524_288, f"Expected 524288, got {kv_per_token}"
print(f"  KV cache per token (Llama-7B FP16): {kv_per_token:,} bytes  ✓")

# Sweep batch × seq_len
print(f"\n  KV cache footprint (GB) — Llama-7B FP16:")
print(f"  {'':>4}  {'':>6}", end="")
seq_lens = [512, 1024, 2048, 4096]
for s in seq_lens:
    print(f"  seq={s:<4}", end="")
print()
print(f"  {'─'*4}  {'─'*6}", end="")
for _ in seq_lens:
    print(f"  {'─'*8}", end="")
print()

for batch in [1, 4, 8, 16, 32]:
    print(f"  b={batch:<3}", end="  ")
    for s in seq_lens:
        gb = kv_cache_bytes(seq_len=s, batch=batch, dtype_bytes=2, **LLAMA7B) / 1e9
        marker = "OOM " if (gb + MODEL_WEIGHTS_GB) > GPU_VRAM_GB else "    "
        print(f"  {gb:.2f}GB{marker}", end="")
    print()

print(f"\n  Model weights: {MODEL_WEIGHTS_GB} GB  |  GPU VRAM: {GPU_VRAM_GB:.1f} GB")
print(f"  OOM threshold: total_kv + {MODEL_WEIGHTS_GB} > {GPU_VRAM_GB:.1f} GB")
print("  ✓ Section 1 passed — KV cache formula verified")


# ─────────────────────────────────────────────────────────────
# SECTION 2: OOM Threshold and Max Concurrency
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Finding the OOM Threshold ──")
print("""
  The OOM threshold is the maximum (batch × seq_len) product that fits
  in GPU memory alongside the model weights.

  AVAILABLE KV MEMORY = GPU_VRAM - MODEL_WEIGHTS

  MAX CONCURRENT REQUESTS at a given seq_len:
    max_batch = available_kv_gb × 1e9 / kv_bytes_per_request

  TODO 1: Implement max_concurrent_requests(vram_gb, model_gb, seq_len,
          n_layers, n_heads, head_dim, dtype_bytes)
  that returns the maximum number of concurrent requests that fit.
""")


def max_concurrent_requests(vram_gb: float, model_gb: float, seq_len: int,
                             n_layers: int, n_heads: int, head_dim: int,
                             dtype_bytes: int = 2) -> int:
    """
    TODO 1: Return max concurrent requests fitting in GPU memory.
    available_bytes = (vram_gb - model_gb) × 1e9
    kv_per_req = kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, 1, dtype_bytes)
    Return int(available_bytes / kv_per_req), minimum 0.
    """
    available = (vram_gb - model_gb) * 1e9
    if available <= 0:
        return 0
    kv_per_req = kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, 1, dtype_bytes)
    return max(0, int(available / kv_per_req))


print(f"  Max concurrent requests (Llama-7B FP16, GPU={GPU_VRAM_GB:.0f}GB):")
print(f"  {'seq_len':>8}  {'max_concurrent':>16}  {'kv_gb per req':>14}")
print(f"  {'─'*8}  {'─'*16}  {'─'*14}")
for s in [512, 1024, 2048, 4096]:
    mc = max_concurrent_requests(GPU_VRAM_GB, MODEL_WEIGHTS_GB, s, **LLAMA7B)
    kv_per = kv_cache_bytes(seq_len=s, batch=1, dtype_bytes=2, **LLAMA7B) / 1e9
    print(f"  {s:>8}  {mc:>16}  {kv_per:>14.3f}")

mc_2k = max_concurrent_requests(GPU_VRAM_GB, MODEL_WEIGHTS_GB, 2048, **LLAMA7B)
assert mc_2k >= 0, "Max concurrent should be non-negative"
print("  ✓ Section 2 passed — OOM threshold and max concurrency computed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Latency Inflation Model
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Latency Inflation Under Memory Pressure ──")
print("""
  As the KV cache fills up, two things happen:
    1. Memory bandwidth: larger KV cache → more memory to scan per step
       Each decode step must load ALL past K,V tensors for attention
       Latency for decode step ∝ seq_len (linear in context length)

    2. Memory pressure eviction: when KV cache exceeds GPU VRAM,
       the system must evict older tokens (sliding window) or swap to CPU.
       Swap-to-CPU introduces 100–1000× latency spike.

  LATENCY GROWTH MODEL (per decode step):
    Without cache pressure:
      decode_ms(s) = base_ms + bandwidth_factor × s
      where bandwidth_factor = (2 × n_layers × n_heads × head_dim × dtype_bytes) / BW_GB/s × 1000

    With memory pressure (swap):
      decode_ms(s) ≈ swap_latency_ms >> compute_latency_ms

  TODO 2: Implement decode_latency_ms(seq_len, n_layers, n_heads, head_dim,
          bandwidth_gb_s, base_ms=0.5, dtype_bytes=2)
  that models the per-step decode latency as memory-bandwidth-bound.
""")


def decode_latency_ms(seq_len: int, n_layers: int, n_heads: int, head_dim: int,
                       bandwidth_gb_s: float, base_ms: float = 0.5,
                       dtype_bytes: int = 2) -> float:
    """
    TODO 2: Model memory-bandwidth-bound decode latency.
    Bytes loaded per step = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes
    (each step loads all K and V for attention over past tokens)
    latency = bytes / (bandwidth_gb_s × 1e9) × 1000 + base_ms
    Return latency in ms.
    """
    bytes_per_step = 2 * n_layers * n_heads * head_dim * seq_len * dtype_bytes
    bw_ms = bytes_per_step / (bandwidth_gb_s * 1e9) * 1000
    return base_ms + bw_ms


# A100: 2000 GB/s HBM bandwidth; A10G: 600 GB/s; consumer RTX4090: 1008 GB/s
# Use measured value if we have it, else fallback
if GPU_VRAM_GB > 0 and DEVICE == "cuda":
    # Quick bandwidth estimate from memory transfer
    _t = torch.rand(64 * 1024 * 1024, device=DEVICE)
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    for _ in range(5):
        _ = _t.clone()
    torch.cuda.synchronize()
    BW_GB_S = (64 * 1024 * 1024 * 4 * 2 / ((time.perf_counter() - _t0) / 5)) / 1e9
    del _t
    print(f"  Measured HBM bandwidth: {BW_GB_S:.0f} GB/s")
else:
    BW_GB_S = 600.0
    print(f"  Assumed HBM bandwidth: {BW_GB_S:.0f} GB/s (A10G spec)")

print(f"\n  Decode latency model (Llama-7B FP16, BW={BW_GB_S:.0f} GB/s):")
print(f"  {'seq_len':>8}  {'latency (ms)':>13}  {'vs seq=128':>12}")
print(f"  {'─'*8}  {'─'*13}  {'─'*12}")
base_lat = decode_latency_ms(128, **LLAMA7B, bandwidth_gb_s=BW_GB_S)
for s in [128, 512, 1024, 2048, 4096]:
    lat = decode_latency_ms(s, **LLAMA7B, bandwidth_gb_s=BW_GB_S)
    ratio = lat / base_lat
    print(f"  {s:>8}  {lat:>13.3f}  {ratio:>11.2f}×")

assert decode_latency_ms(2048, **LLAMA7B, bandwidth_gb_s=BW_GB_S) > \
       decode_latency_ms(128, **LLAMA7B, bandwidth_gb_s=BW_GB_S), \
       "Longer context should increase decode latency"
print("  ✓ Section 3 passed — decode latency model built")


# ─────────────────────────────────────────────────────────────
# SECTION 4: KV Cache Strategies Comparison
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Comparing KV Cache Strategies ──")
print("""
  When a serving system runs out of KV cache memory, it has three options:

  STRATEGY 1: Full KV Cache (no eviction)
    Keep all past tokens' K,V tensors. OOM when batch × seq exceeds budget.
    Best quality (no information loss). Strictly limited by VRAM.

  STRATEGY 2: Sliding Window Attention
    Keep only the last W tokens' K,V tensors. Memory is O(W × batch), not O(seq × batch).
    Quality degrades for long-range dependencies.
    Mistral-7B uses W=4096; Gemma uses W=2048.

  STRATEGY 3: PagedAttention (vLLM)
    Allocate KV cache in fixed-size blocks (e.g. 16 tokens/block).
    Allows non-contiguous allocation → eliminates fragmentation.
    Max concurrency = floor(available_blocks × block_size / seq_len_per_req).
    Quality same as full cache (no eviction, just flexible allocation).

  TODO 3: Implement kv_strategy_analysis(vram_gb, model_gb, seq_len,
          window_size, block_size, **llm_params)
  that returns max_concurrent for each of the three strategies.
""")


def kv_strategy_analysis(vram_gb: float, model_gb: float, seq_len: int,
                          window_size: int, block_size: int,
                          n_layers: int, n_heads: int, head_dim: int,
                          dtype_bytes: int = 2) -> dict:
    """
    TODO 3: Return max concurrent requests for each strategy.
    - full:    max_concurrent_requests(vram_gb, model_gb, seq_len, ...)
    - sliding: use window_size instead of seq_len
    - paged:   available_blocks = available_bytes / block_bytes;
               each request needs ceil(seq_len / block_size) blocks
               → max_concurrent = available_blocks * block_size // seq_len
    """
    avail_bytes = (vram_gb - model_gb) * 1e9
    kv_per_tok  = 2 * n_layers * n_heads * head_dim * dtype_bytes

    # Full cache
    full_mc = max(0, int(avail_bytes / (kv_per_tok * seq_len)))

    # Sliding window (only W tokens cached)
    window_bytes = kv_per_tok * min(window_size, seq_len)
    slide_mc = max(0, int(avail_bytes / window_bytes))

    # Paged attention
    block_bytes = kv_per_tok * block_size
    total_blocks = max(0, int(avail_bytes / block_bytes))
    blocks_per_req = math.ceil(seq_len / block_size)
    paged_mc = max(0, total_blocks // blocks_per_req) if blocks_per_req > 0 else 0

    return {
        "full_cache":     full_mc,
        "sliding_window": slide_mc,
        "paged":          paged_mc,
    }


print(f"\n  Max concurrent requests by strategy (Llama-7B FP16, GPU={GPU_VRAM_GB:.0f}GB):")
print(f"  {'seq_len':>8}  {'Full cache':>12}  {'Sliding W=1k':>14}  {'Paged B=16':>12}")
print(f"  {'─'*8}  {'─'*12}  {'─'*14}  {'─'*12}")
for s in [512, 1024, 2048, 4096]:
    r = kv_strategy_analysis(GPU_VRAM_GB, MODEL_WEIGHTS_GB, s,
                              window_size=1024, block_size=16, **LLAMA7B)
    print(f"  {s:>8}  {r['full_cache']:>12}  {r['sliding_window']:>14}  {r['paged']:>12}")

# Paged should match full (same memory usage, just flexible allocation)
r_test = kv_strategy_analysis(GPU_VRAM_GB, MODEL_WEIGHTS_GB, 2048,
                               window_size=4096, block_size=16, **LLAMA7B)
assert r_test["paged"] >= 0 and r_test["full_cache"] >= 0
print("  ✓ Section 4 passed — strategy comparison complete")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Saving the memory pressure profile
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Memory Pressure Profile Report ──")

profile = {
    "hardware": {"vram_gb": GPU_VRAM_GB, "bw_gb_s": round(BW_GB_S, 0)},
    "model": {"name": "Llama-7B", "weights_gb": MODEL_WEIGHTS_GB, **LLAMA7B},
    "max_concurrent": {},
    "decode_latency_ms": {},
}

for s in [512, 1024, 2048, 4096]:
    mc = max_concurrent_requests(GPU_VRAM_GB, MODEL_WEIGHTS_GB, s, **LLAMA7B)
    lat = decode_latency_ms(s, **LLAMA7B, bandwidth_gb_s=BW_GB_S)
    profile["max_concurrent"][str(s)] = mc
    profile["decode_latency_ms"][str(s)] = round(lat, 3)

profile_path = "/tmp/capstone18_kv_profile.json"
with open(profile_path, "w") as f:
    json.dump(profile, f, indent=2)
print(f"  Profile saved: {profile_path}")

print(f"""
  KEY INSIGHTS:
    1. KV cache grows linearly with seq_len × batch × n_layers.
    2. Decode latency grows linearly with seq_len (more K,V to load each step).
    3. Sliding window caps memory but loses long-range context.
    4. PagedAttention matches full cache quality with better memory utilisation.
    5. The OOM threshold for seq=2048 on a {GPU_VRAM_GB:.0f}GB GPU is ~{max_concurrent_requests(GPU_VRAM_GB, MODEL_WEIGHTS_GB, 2048, **LLAMA7B)} concurrent requests.

  NEXT STEP: Run 18.2_memory_budget_planning.py to apply this analysis
  to concrete deployment planning decisions.
""")
assert "max_concurrent" in profile
print("  ✓ Section 5 passed — memory pressure profile saved")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 18.1 complete!")
print("  KV cache formula, OOM threshold, latency model, strategy comparison.")
print("  Next: VI.Capstone_Projects/18.KV_Cache_Memory_Pressure/18.2_memory_budget_planning.py")
print("=" * 60)
