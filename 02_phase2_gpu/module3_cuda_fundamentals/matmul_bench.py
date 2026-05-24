#!/usr/bin/env python3
"""
matmul_bench.py  —  Phase 2, Module 4: Matrix Multiply Benchmark & Roofline
============================================================================

HOW TO RUN:
    # Basic benchmark:
    python matmul_bench.py

    # Profile specific size with ncu:
    ncu --set roofline --kernel-name ".*gemm.*" python matmul_bench.py --size 2048

    # Profile all sizes:
    ncu --set basic --csv --log-file ncu_matmul.csv python matmul_bench.py


"""

import argparse
import torch
import math

parser = argparse.ArgumentParser()
parser.add_argument("--size",   type=int, default=0,
                    help="Single matrix size to benchmark (0 = sweep all sizes)")
parser.add_argument("--dtype",  default="float16", choices=["float32","float16","bfloat16"])
parser.add_argument("--iters",  type=int, default=50)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: Running on CPU — results not meaningful for GPU roofline analysis")

dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
dtype = dtype_map[args.dtype]
dtype_bytes = 4 if args.dtype == "float32" else 2


def cuda_time_ms(fn, warmup=5, iters=None):
    iters = iters or args.iters
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def arithmetic_intensity(M, K, N, dtype_bytes):
    """
    Compute arithmetic intensity for an M×K @ K×N matrix multiply.
    AI = FLOPs / bytes_accessed
    """
    flops = 2 * M * K * N                              # each multiply-add = 2 ops
    bytes_in  = (M * K + K * N) * dtype_bytes          # read both input matrices
    bytes_out = M * N * dtype_bytes                     # write result matrix
    return flops / (bytes_in + bytes_out)


print(f"\n{'='*70}")
print(f"  Matrix Multiply Benchmark & Roofline Analysis")
print(f"{'='*70}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {props.name}")
    print(f"  VRAM    : {props.total_memory/1e9:.1f}GB")
    print(f"  SMs     : {props.multi_processor_count}")
    print(f"  cc      : {props.major}.{props.minor}")
print(f"  dtype   : {args.dtype} ({dtype_bytes} bytes)")
print()

# ── Step 1: Measure peak memory bandwidth ─────────────────────────────────────
print("── Step 1: Measuring Peak Memory Bandwidth ─────────────────────────────")
print("   (copy a large tensor: bandwidth-only, no compute)")

n = 128 * 1024 * 1024 // dtype_bytes   # ~128MB or ~256MB
src = torch.randn(n, device=device, dtype=dtype)

ms_bw = cuda_time_ms(lambda: src.clone(), warmup=10)
peak_bw_gbs = (n * dtype_bytes * 2) / (ms_bw / 1000) / 1e9   # read + write = ×2
print(f"  Peak memory bandwidth : {peak_bw_gbs:.1f} GB/s")
print(f"  (Expected for RTX 4060 GDDR6: ~272 GB/s)")

# ── Step 2: Measure peak compute (large square matmul) ────────────────────────
print("\n── Step 2: Measuring Peak Compute (FP16 8192×8192 matmul) ──────────────")

M_peak = 8192
if device == "cuda" and torch.cuda.get_device_properties(0).total_memory > 4e9:
    try:
        A = torch.randn(M_peak, M_peak, device=device, dtype=dtype)
        B = torch.randn(M_peak, M_peak, device=device, dtype=dtype)
        ms_peak = cuda_time_ms(lambda: torch.mm(A, B), warmup=5, iters=20)
        peak_flops = 2 * M_peak**3
        peak_tflops = peak_flops / (ms_peak / 1000) / 1e12
        print(f"  Peak compute (measured) : {peak_tflops:.1f} TFLOP/s")
        print(f"  (Expected for RTX 4060 FP16 tensor cores: ~136 TFLOP/s)")
        ridge_point = peak_tflops * 1e12 / (peak_bw_gbs * 1e9)
        print(f"  Ridge point (compute/bandwidth): {ridge_point:.1f} FLOP/byte")
        print(f"  Matrices with AI > {ridge_point:.0f} are compute-bound")
        del A, B
    except RuntimeError:
        peak_tflops = 136.0
        ridge_point = 50.0
        print(f"  (Too large for available VRAM — using theoretical: {peak_tflops} TFLOP/s)")
else:
    peak_tflops, ridge_point = 10.0, 20.0
    print(f"  (Skipped — VRAM too small or CPU mode)")

# ── Step 3: Sweep matrix sizes ────────────────────────────────────────────────
print(f"\n── Step 3: Matrix Size Sweep ────────────────────────────────────────────")
print(f"   Showing: size, time, TFLOP/s, arithmetic intensity, bound type")
print()
print(f"  {'Size (M=K=N)':>14}  {'Time (ms)':>10}  {'TFLOP/s':>10}  {'AI':>8}  {'% Peak':>8}  {'Bound':>12}")
print(f"  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*12}")

sizes = [args.size] if args.size > 0 else [64, 128, 256, 512, 1024, 2048, 4096]

for M in sizes:
    try:
        A = torch.randn(M, M, device=device, dtype=dtype)
        B = torch.randn(M, M, device=device, dtype=dtype)

        ms     = cuda_time_ms(lambda: torch.mm(A, B))
        flops  = 2 * M**3
        tflops = flops / (ms / 1000) / 1e12
        ai     = arithmetic_intensity(M, M, M, dtype_bytes)
        pct    = tflops / peak_tflops * 100

        bound = "compute" if ai > ridge_point else "memory-BW"
        flag  = "" if pct > 50 else " ← LOW"

        print(f"  {M:>14}  {ms:>10.3f}  {tflops:>10.2f}  {ai:>8.1f}  {pct:>7.1f}%  {bound:>12}{flag}")
        del A, B

    except RuntimeError as e:
        print(f"  {M:>14}  (OOM — {str(e)[:30]})")

# ── Step 4: Simulate LLM attention matrix size ─────────────────────────────────
print(f"\n── Step 4: Realistic LLM Attention Matmuls ──────────────────────────────")
print(f"   Q @ K.T for different batch × seq_len × head_dim combinations")
print()
print(f"  {'Config':>30}  {'Time (ms)':>10}  {'TFLOP/s':>10}  {'AI':>8}  {'Bound':>12}")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")

# In transformer attention: Q shape = (B × n_heads, T, head_dim)
# Q @ K.T = (B × n_heads, T, T)  — attention scores
configs = [
    # (batch_size, n_heads, seq_len, head_dim, label)
    (1,  32, 16,   128, "batch=1, T=16 (decode step)"),
    (1,  32, 128,  128, "batch=1, T=128"),
    (1,  32, 512,  128, "batch=1, T=512"),
    (8,  32, 128,  128, "batch=8, T=128"),
    (32, 32, 128,  128, "batch=32, T=128 (serving)"),
]

for B, H, T, D, label in configs:
    try:
        # Q shape: (B×H, T, D)  —  as seen inside the attention kernel
        Q = torch.randn(B*H, T, D, device=device, dtype=dtype)
        K = torch.randn(B*H, T, D, device=device, dtype=dtype)

        ms    = cuda_time_ms(lambda: torch.bmm(Q, K.transpose(-2, -1)))
        M_, K_, N_ = B*H*T, D, T
        flops  = 2 * M_ * K_ * N_
        tflops = flops / (ms/1000) / 1e12
        ai     = arithmetic_intensity(M_, K_, N_, dtype_bytes)
        bound  = "compute" if ai > ridge_point else "memory-BW"

        print(f"  {label:>30}  {ms:>10.4f}  {tflops:>10.3f}  {ai:>8.1f}  {bound:>12}")
        del Q, K
    except RuntimeError:
        print(f"  {label:>30}  (OOM)")

print(f"""
  KEY OBSERVATIONS:
  1. decode step (batch=1, T=16)  → tiny matmuls → MEMORY-BOUND → low GPU util
     This is why single-user LLM inference under-uses the GPU.
  2. Large batches + long sequences → higher AI → approaches compute-bound
     This is why batching improves GPU utilisation in production.
  3. Use ncu --set roofline to verify these predictions with hardware counters.

  PROFILING COMMANDS:
    ncu --set roofline --kernel-name ".*gemm.*" --launch-count 3 \\
        python matmul_bench.py --size 1024
    
    ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,\\
        sm__throughput.avg.pct_of_peak_sustained_elapsed \\
        --kernel-name ".*gemm.*" python matmul_bench.py --size 512
""")
