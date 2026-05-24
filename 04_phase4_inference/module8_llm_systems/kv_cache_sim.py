#!/usr/bin/env python3
"""
kv_cache_sim.py  —  Phase 4, Module 8: KV Cache Size Estimator & Simulator
===========================================================================

HOW TO RUN:
    # Estimate KV cache for common models:
    python kv_cache_sim.py --estimate

    # Simulate how many concurrent requests fit in 8GB:
    python kv_cache_sim.py --simulate --vram-gb 8

    # Estimate a specific model:
    python kv_cache_sim.py --model llama-7b --seq-len 2048 --batch 10


"""

import argparse
import math
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--estimate",  action="store_true", help="Show KV cache estimates for common models")
parser.add_argument("--simulate",  action="store_true", help="Simulate PagedAttention vs static allocation")
parser.add_argument("--vram-gb",   type=float, default=8.0, help="Available GPU VRAM in GB")
parser.add_argument("--model",     default="llama-7b",
                    choices=["tiny-transformer","gpt2","llama-7b","llama-13b","llama-70b","mistral-7b"],
                    help="Model to estimate for")
parser.add_argument("--seq-len",   type=int, default=2048, help="Sequence length per request")
parser.add_argument("--batch",     type=int, default=1,    help="Number of concurrent requests")
parser.add_argument("--dtype",     default="float16", choices=["float16","float32","int8"])
args = parser.parse_args()


# ── Model architecture specs ──────────────────────────────────────────────────
MODEL_SPECS = {
    "tiny-transformer": dict(n_layers=4,  n_heads=4,  head_dim=64,  params_b=0.007),
    "gpt2":             dict(n_layers=12, n_heads=12, head_dim=64,  params_b=0.117),
    "llama-7b":         dict(n_layers=32, n_heads=32, head_dim=128, params_b=7.0),
    "llama-13b":        dict(n_layers=40, n_heads=40, head_dim=128, params_b=13.0),
    "llama-70b":        dict(n_layers=80, n_heads=64, head_dim=128, params_b=70.0),
    "mistral-7b":       dict(n_layers=32, n_heads=32, head_dim=128, params_b=7.0),
}

DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1}


def kv_cache_bytes(model_spec, seq_len, batch, dtype_bytes):
    """
    Compute total KV cache memory in bytes.

    Formula:
        2 (K and V) × n_layers × n_heads × head_dim × seq_len × batch × dtype_bytes
    """
    L  = model_spec["n_layers"]
    H  = model_spec["n_heads"]
    D  = model_spec["head_dim"]
    return 2 * L * H * D * seq_len * batch * dtype_bytes


def weight_bytes(model_spec, dtype_bytes):
    """Approximate model weight memory."""
    return model_spec["params_b"] * 1e9 * dtype_bytes


def format_bytes(b):
    """Human-readable byte count."""
    if b >= 1e9:
        return f"{b/1e9:.2f}GB"
    elif b >= 1e6:
        return f"{b/1e6:.0f}MB"
    else:
        return f"{b/1e3:.0f}KB"


# =============================================================================
# SECTION 1 — KV Cache Size Estimates for Common Models
# =============================================================================
if args.estimate or (not args.simulate and not args.model):
    print(f"\n{'='*70}")
    print(f"  KV Cache Size Estimates")
    print(f"  dtype={args.dtype}  ({DTYPE_BYTES[args.dtype]} bytes/element)")
    print(f"{'='*70}")

    dtype_b = DTYPE_BYTES[args.dtype]

    # Header
    print(f"\n  {'Model':<16}  {'Params':>8}  {'Weights':>10}  {'KV@512t':>10}  "
          f"{'KV@2048t':>10}  {'Max concurrent @8GB':>22}")
    print(f"  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*22}")

    for name, spec in MODEL_SPECS.items():
        w_bytes   = weight_bytes(spec, dtype_b)
        kv_512    = kv_cache_bytes(spec, 512,  1, dtype_b)
        kv_2048   = kv_cache_bytes(spec, 2048, 1, dtype_b)
        vram_avail = args.vram_gb * 1e9 - w_bytes   # VRAM after weights
        max_concurrent = max(0, int(vram_avail / kv_2048)) if kv_2048 > 0 else 0

        print(f"  {name:<16}  {spec['params_b']:>7.1f}B  {format_bytes(w_bytes):>10}  "
              f"{format_bytes(kv_512):>10}  {format_bytes(kv_2048):>10}  "
              f"{max_concurrent:>12} req @ 2048tok")

    print(f"""
  KEY INSIGHT:
    At FP16, Llama-7B weights alone = ~14GB (exceeds RTX 4060 8GB).
    You need quantization (INT8→7GB, INT4→3.5GB) to run on consumer GPU.
    Even with INT4 weights, each 2048-token request needs ~500MB KV cache.
    RTX 4060 can serve ~4-6 concurrent Llama-7B requests at INT4.
  """)


# =============================================================================
# SECTION 2 — Estimate for a Specific Model + Configuration
# =============================================================================
dtype_b = DTYPE_BYTES[args.dtype]
spec = MODEL_SPECS.get(args.model)

if spec:
    print(f"\n{'='*60}")
    print(f"  Detailed Estimate: {args.model}  dtype={args.dtype}")
    print(f"{'='*60}")

    w_gb   = weight_bytes(spec, dtype_b) / 1e9
    kv_gb  = kv_cache_bytes(spec, args.seq_len, args.batch, dtype_b) / 1e9
    total  = w_gb + kv_gb
    avail  = args.vram_gb

    print(f"""
  Configuration:
    Batch size (concurrent requests) : {args.batch}
    Sequence length per request      : {args.seq_len} tokens
    dtype                            : {args.dtype} ({dtype_b} bytes)

  Memory Breakdown:
    Model weights                    : {w_gb:.2f} GB
    KV cache ({args.batch} req × {args.seq_len} tok)  : {kv_gb:.2f} GB
    Activations (est. ~10% of weights): {w_gb*0.1:.2f} GB
    ─────────────────────────────────────────
    Total estimated                  : {total + w_gb*0.1:.2f} GB

  GPU VRAM                           : {avail:.1f} GB
  Fits?                              : {'✓ YES' if total + w_gb*0.1 < avail else '✗ NO — need quantization or smaller batch'}
    """)

    if total + w_gb*0.1 > avail:
        # Suggest what would work
        print(f"  Alternatives:")
        for batch_try in [args.batch // 2, 1]:
            kv_try = kv_cache_bytes(spec, args.seq_len, batch_try, dtype_b) / 1e9
            total_try = w_gb + kv_try + w_gb*0.1
            print(f"    batch={batch_try}: {total_try:.2f}GB  {'✓' if total_try < avail else '✗'}")

        # INT8 option
        kv_int8 = kv_cache_bytes(spec, args.seq_len, args.batch, 1) / 1e9
        w_int8  = weight_bytes(spec, 1) / 1e9
        print(f"    INT8 quantized:  weights={w_int8:.2f}GB  kv={kv_int8:.2f}GB  "
              f"total={w_int8+kv_int8:.2f}GB  {'✓' if w_int8+kv_int8 < avail else '✗'}")


# =============================================================================
# SECTION 3 — PagedAttention vs Static Allocation Simulation
# =============================================================================
if args.simulate:
    print(f"\n{'='*60}")
    print(f"  PagedAttention Simulation")
    print(f"  VRAM: {args.vram_gb}GB  Model: {args.model}")
    print(f"{'='*60}")
    print("""
  STATIC ALLOCATION (naive HuggingFace generate):
    Each request pre-allocates max_seq_len worth of KV cache up-front.
    Memory is reserved even if the actual output is 10 tokens.
    Result: many GPU memory pages are wasted → fewer concurrent requests.

  PAGED ATTENTION (vLLM):
    KV cache is divided into fixed-size "blocks" (pages), e.g. 16 tokens each.
    Blocks are allocated lazily as each request generates more tokens.
    Completed requests' blocks are returned to the pool immediately.
    Result: no fragmentation → more concurrent requests.
  """)

    if spec:
        dtype_b = DTYPE_BYTES["float16"]   # simulate FP16
        BLOCK_SIZE = 16   # tokens per block (vLLM default)

        kv_per_token = 2 * spec["n_layers"] * spec["n_heads"] * spec["head_dim"] * dtype_b
        kv_per_block = kv_per_token * BLOCK_SIZE
        w_bytes      = weight_bytes(spec, dtype_b)

        vram_total   = args.vram_gb * 1e9
        kv_pool      = vram_total - w_bytes   # VRAM after weights (80% of remaining)
        kv_pool      *= 0.85                  # leave 15% for activations/overhead
        n_blocks_total = int(kv_pool / kv_per_block)

        max_seq = 2048   # max sequence length to consider

        # Static allocation: each request reserves max_seq_len
        blocks_per_req_static = math.ceil(max_seq / BLOCK_SIZE)
        n_concurrent_static   = max(0, n_blocks_total // blocks_per_req_static)

        # PagedAttention: requests consume only what they need (average ~512 tokens)
        avg_output_tokens = 512
        blocks_per_req_paged = math.ceil(avg_output_tokens / BLOCK_SIZE)
        n_concurrent_paged   = max(0, n_blocks_total // blocks_per_req_paged)

        print(f"  Model: {args.model}  (weights: {format_bytes(w_bytes)})")
        print(f"  KV pool available: {format_bytes(kv_pool)}")
        print(f"  Total KV blocks ({BLOCK_SIZE} tok each): {n_blocks_total:,}")
        print()
        print(f"  {'Method':<25}  {'Blocks/req':>12}  {'Max concurrent':>15}  {'GPU util est.':>14}")
        print(f"  {'─'*25}  {'─'*12}  {'─'*15}  {'─'*14}")
        print(f"  {'Static (HF generate)':<25}  {blocks_per_req_static:>12}  {n_concurrent_static:>15}  {'low (wasted)':>14}")
        print(f"  {'PagedAttention (vLLM)':<25}  {blocks_per_req_paged:>12}  {n_concurrent_paged:>15}  {'high (reused)':>14}")

        improvement = n_concurrent_paged / max(n_concurrent_static, 1)
        print(f"\n  PagedAttention serves {improvement:.1f}× more concurrent requests")
        print(f"  at the SAME memory (because pages aren't wasted on unused slots)")

    print(f"""
  vLLM COMMANDS TO CONTROL KV CACHE:
    --gpu-memory-utilization 0.85   # fraction of VRAM for KV pool
    --max-model-len 2048            # max sequence length (limits max block count)
    --block-size 16                 # tokens per KV block (default 16)
    --enable-prefix-caching         # share KV blocks for identical prefixes
    --swap-space 4                  # GB of CPU RAM as KV overflow (slower)
  """)
