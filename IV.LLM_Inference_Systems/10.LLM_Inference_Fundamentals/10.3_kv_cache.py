#!/usr/bin/env python3
"""
IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_kv_cache.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 10: LLM Inference Fundamentals — Section 2: KV Cache
=======================================================================
Covers book section 10.2:
  • What the KV cache stores and why it exists
  • Memory formula: 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
  • Estimating KV cache size for common models (GPT-2, Llama-7B, Llama-70B)
  • How KV cache size limits batch size and sequence length
  • Practical constraints: when you run out of VRAM

Run:  python IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_kv_cache.py
All sections must print ✓.
"""

import math
import torch

print("=" * 60)
print("  Exercise 10.2 — KV Cache Memory Management")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: What the KV cache stores and why
# ─────────────────────────────────────────────────────────────
print("── Section 1: The KV Cache — Purpose and Contents ──")
print("""
  In a transformer, each attention layer computes keys (K) and values (V)
  for every token in the sequence. During decode, the new token attends to
  ALL previous tokens. Without caching, we would recompute K and V for the
  full context on every decode step — O(seq_len) work per step per layer.

  The KV cache stores those computed K and V tensors after each step.
  On the next step, only the new token's K and V need computing; the rest
  are read from cache.

  KV CACHE CONTENTS per layer per request:
    Keys   : [n_heads, seq_len, head_dim]  shape
    Values : [n_heads, seq_len, head_dim]  shape

  Total KV cache bytes for one request:
    2  (K and V)
    × n_layers
    × n_heads
    × head_dim
    × seq_len
    × dtype_bytes

  This formula appears in every LLM memory budget — memorise it.
""")
print("  ✓ Section 1 passed — KV cache avoids recomputing attention over history")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The KV cache formula
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: KV Cache Memory Formula ──")
print("""
  KV_bytes = 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes

  For Llama-2-7B (FP16):
    n_layers=32, n_heads=32, head_dim=128, dtype=float16 (2 bytes)
    Per token per batch element:
      2 × 32 × 32 × 128 × 2 = 524,288 bytes ≈ 512 KB
    For 2048-token context: 512 KB × 2048 = 1 GB per request

  TODO 1: Implement kv_cache_bytes() below.
""")


def kv_cache_bytes(n_layers: int, n_heads: int, head_dim: int,
                   seq_len: int, batch: int, dtype_bytes: int) -> int:
    """
    TODO 1: Return total KV cache bytes.
    Formula: 2 × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
    """
    # YOUR CODE HERE
    return 2 * n_layers * n_heads * head_dim * seq_len * batch * dtype_bytes


# Verify against known value for Llama-2-7B
llama7b_kv_per_token = kv_cache_bytes(32, 32, 128, 1, 1, 2)
assert llama7b_kv_per_token == 524_288, \
    f"Expected 524288 bytes/token for Llama-7B, got {llama7b_kv_per_token}"
print(f"  Llama-7B KV per token : {llama7b_kv_per_token:,} bytes = {llama7b_kv_per_token/1024:.0f} KB")

llama7b_2k = kv_cache_bytes(32, 32, 128, 2048, 1, 2)
print(f"  Llama-7B KV @ 2048t   : {llama7b_2k/1e9:.2f} GB per request")
print("  ✓ Section 2 passed — kv_cache_bytes() formula verified")


# ─────────────────────────────────────────────────────────────
# SECTION 3: KV cache size across common models
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: KV Cache Survey Across Models ──")
print("""
  Different model sizes have dramatically different KV cache requirements.
  This directly limits how many concurrent requests a GPU can serve.
""")

MODEL_SPECS = {
    "GPT-2 (117M)":   dict(n_layers=12, n_heads=12, head_dim=64,  params_b=0.117),
    "Llama-2-7B":     dict(n_layers=32, n_heads=32, head_dim=128, params_b=7.0),
    "Llama-2-13B":    dict(n_layers=40, n_heads=40, head_dim=128, params_b=13.0),
    "Llama-2-70B":    dict(n_layers=80, n_heads=64, head_dim=128, params_b=70.0),
    "Mistral-7B":     dict(n_layers=32, n_heads=32, head_dim=128, params_b=7.0),
}

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "int8": 1}
SEQ_LENS    = [512, 2048, 4096]
DTYPE       = "fp16"
DB          = DTYPE_BYTES[DTYPE]

print(f"  dtype={DTYPE} ({DB} bytes/element)\n")
print(f"  {'Model':<16}  {'Params':>8}  {'KV@512':>10}  {'KV@2048':>10}  {'KV@4096':>10}")
print(f"  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

for name, spec in MODEL_SPECS.items():
    rows = []
    for sl in SEQ_LENS:
        b = kv_cache_bytes(spec["n_layers"], spec["n_heads"], spec["head_dim"], sl, 1, DB)
        if b >= 1e9:
            rows.append(f"{b/1e9:.2f}GB")
        else:
            rows.append(f"{b/1e6:.0f}MB")
    print(f"  {name:<16}  {spec['params_b']:>7.1f}B  {rows[0]:>10}  {rows[1]:>10}  {rows[2]:>10}")

# TODO 2: Verify GPT-2 KV at 512 tokens is smaller than Llama-7B KV at 512 tokens
gpt2_kv_512   = kv_cache_bytes(12, 12, 64,  512, 1, DB)
llama7b_kv_512 = kv_cache_bytes(32, 32, 128, 512, 1, DB)
assert gpt2_kv_512 < llama7b_kv_512, \
    f"GPT-2 KV should be smaller than Llama-7B: {gpt2_kv_512} vs {llama7b_kv_512}"
print(f"\n  GPT-2 KV@512  : {gpt2_kv_512/1e6:.1f} MB")
print(f"  Llama-7B KV@512: {llama7b_kv_512/1e6:.1f} MB  ({llama7b_kv_512//gpt2_kv_512}× larger)")
print("  ✓ Section 3 passed — larger models have proportionally larger KV caches")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Max batch size given a VRAM budget
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: VRAM Budget and Concurrency Limits ──")
print("""
  A GPU has fixed VRAM. After loading model weights, the remaining
  VRAM is shared between KV caches and activations.

  For Llama-2-7B on a 24 GB GPU (e.g. RTX 4090):
    Weights (FP16) : 7B × 2 bytes = 14 GB
    Remaining      : 24 - 14 = 10 GB for KV + activations
    Activations    : ~1–2 GB (depends on batch size)
    KV pool        : ~8 GB

  Max concurrent requests at 2048-token context:
    8 GB / (1 GB per request) = 8 requests

  TODO 3: Implement max_concurrent_requests() below.
""")


def max_concurrent_requests(
    n_layers: int, n_heads: int, head_dim: int,
    seq_len: int, dtype_bytes: int,
    vram_for_kv_gb: float
) -> int:
    """
    TODO 3: Return the maximum number of concurrent requests that fit in vram_for_kv_gb.
    Use kv_cache_bytes() for a single request (batch=1).
    Return floor(vram_for_kv_bytes / kv_per_request).
    """
    # YOUR CODE HERE
    kv_per_req = kv_cache_bytes(n_layers, n_heads, head_dim, seq_len, 1, dtype_bytes)
    vram_bytes  = vram_for_kv_gb * 1e9
    return int(vram_bytes // kv_per_req)


# Verify Llama-7B estimate
llama7b_max = max_concurrent_requests(32, 32, 128, 2048, 2, vram_for_kv_gb=8.0)
print(f"  Llama-7B on 24GB GPU (8GB for KV, 2048-tok context):")
print(f"    Max concurrent requests: {llama7b_max}")
assert llama7b_max >= 1, "Should fit at least 1 request"

vram_scenarios = [
    ("A100 80GB", 80, "Llama-7B"),
    ("RTX 4090 24GB", 24, "Llama-7B"),
    ("RTX 3090 24GB", 24, "GPT-2"),
    ("A10G 24GB", 24, "Mistral-7B"),
]

print(f"\n  {'GPU':>18}  {'Model':<12}  {'KV pool':>8}  {'Max concurrent (2048t)':>22}")
print(f"  {'─'*18}  {'─'*12}  {'─'*8}  {'─'*22}")
for gpu, vram_gb, model_name in vram_scenarios:
    spec  = MODEL_SPECS.get(model_name, MODEL_SPECS["Llama-2-7B"])
    w_gb  = spec["params_b"] * 2  # FP16 weights
    kv_gb = max(0.0, vram_gb - w_gb - 2)  # 2 GB for activations
    n_req = max_concurrent_requests(
        spec["n_layers"], spec["n_heads"], spec["head_dim"], 2048, 2, kv_gb
    )
    print(f"  {gpu:>18}  {model_name:<12}  {kv_gb:>7.0f}GB  {n_req:>22}")

print("  ✓ Section 4 passed — VRAM budget directly limits concurrency")


# ─────────────────────────────────────────────────────────────
# SECTION 5: KV cache growth during generation
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: KV Cache Growth During a Request ──")
print("""
  The KV cache for a single request grows linearly with each generated token.
  If a request starts with a 512-token prompt and generates 512 tokens:
    Start  : KV cache holds 512 tokens worth of K and V
    End    : KV cache holds 1024 tokens worth of K and V

  This is why long-context requests are expensive — they consume
  proportionally more KV cache VRAM than short ones.

  Implication: a request with max_len=4096 could need 8× the KV memory
  of a request with max_len=512, even if the actual output is 100 tokens.
  Static allocation (naive) reserves the max up-front. PagedAttention
  allocates on demand (Exercise 11.3).
""")

spec   = MODEL_SPECS["Llama-2-7B"]
DB_FP16 = 2
prompt_len = 512
generate_len = 512

print(f"  Llama-7B FP16, prompt={prompt_len}, generate={generate_len}:")
print(f"  {'Token #':>8}  {'Context len':>12}  {'KV Cache (MB)':>14}")
print(f"  {'─'*8}  {'─'*12}  {'─'*14}")
checkpoints = [0, generate_len // 4, generate_len // 2, generate_len]
for step in checkpoints:
    ctx_len = prompt_len + step
    kv_mb   = kv_cache_bytes(spec["n_layers"], spec["n_heads"], spec["head_dim"],
                              ctx_len, 1, DB_FP16) / 1e6
    print(f"  {step:>8}  {ctx_len:>12}  {kv_mb:>14.1f}")

kv_start = kv_cache_bytes(spec["n_layers"], spec["n_heads"], spec["head_dim"],
                           prompt_len, 1, DB_FP16)
kv_end   = kv_cache_bytes(spec["n_layers"], spec["n_heads"], spec["head_dim"],
                           prompt_len + generate_len, 1, DB_FP16)

# TODO 4: Verify KV cache doubles when context length doubles
assert kv_end == kv_start * 2, \
    f"KV cache should double with doubled context: {kv_start} vs {kv_end}"
print(f"\n  KV cache at end = {kv_end/kv_start:.0f}× start  (context doubled)")
print("  ✓ Section 5 passed — KV cache grows linearly with context length")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 10.2 complete!")
print("  You can now calculate KV cache memory for any model,")
print("  predict max concurrency, and understand the memory budget.")
print("  Next: IV.LLM_Inference_Systems/10.LLM_Inference_Fundamentals/10.3_inference_metrics.py")
print("=" * 60)
