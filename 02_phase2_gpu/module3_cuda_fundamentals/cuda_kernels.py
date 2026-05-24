#!/usr/bin/env python3
"""
cuda_kernels.py  —  Phase 2, Module 3+4: CUDA Programming Experiments
======================================================================

HOW TO RUN:
    # All experiments:
    python cuda_kernels.py

    # Single experiment:
    python cuda_kernels.py --exp occupancy
    python cuda_kernels.py --exp coalescing
    python cuda_kernels.py --exp tensorcores

    # Profile with ncu to see hardware counters:
    ncu --set basic python cuda_kernels.py --exp coalescing


"""

import argparse
import time
import torch
import math

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
                    choices=["all","occupancy","coalescing","divergence",
                             "tensorcores","memory_hierarchy","fusion"])
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: CUDA not available. Some experiments will not be meaningful on CPU.")

props = torch.cuda.get_device_properties(0) if device == "cuda" else None


def separator(title):
    print(f"\n{'═'*60}")
    print(f"  EXP: {title}")
    print(f"{'─'*60}")


def cuda_time(fn, warmup=3, iters=20):
    """Accurate GPU timing using CUDA events. Excludes warmup runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms per iteration


# =============================================================================
# EXP 1 — Occupancy: Effect of Batch Size on GPU Utilisation
# =============================================================================
def exp_occupancy():
    separator("OCCUPANCY — Batch Size vs SM Utilisation")
    print("""
  CONCEPT:
    The GPU has many SMs (Streaming Multiprocessors). Each SM can run
    multiple warps. With a tiny batch, there may not be enough warps to
    keep all SMs busy — the GPU is UNDER-OCCUPIED.

    Practical rule: batch_size × seq_len × model_size must be large
    enough to generate thousands of thread blocks to fill the GPU.

  WHAT WE MEASURE:
    FLOP/s throughput at different batch sizes. If the GPU scales linearly
    with batch, we're in the "occupancy-limited" regime. When throughput
    plateaus, the GPU is saturated (we've hit compute or memory limits).
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    d = 512   # matrix dimension
    # Create weight matrices for a simple linear layer matmul
    W = torch.randn(d, d, device=device, dtype=torch.float16)

    results = []
    for batch in [1, 4, 8, 16, 32, 64, 128, 256]:
        X = torch.randn(batch, d, device=device, dtype=torch.float16)
        # Each call = one matrix multiply: (batch, d) @ (d, d) = (batch, d)
        # FLOPs = 2 × batch × d × d (multiply-add = 2 ops)
        ms   = cuda_time(lambda: torch.mm(X, W))
        flops = 2 * batch * d * d
        tflops = flops / (ms / 1000) / 1e12
        results.append((batch, ms, tflops))
        print(f"  batch={batch:4d}  time={ms:.3f}ms  throughput={tflops:.2f} TFLOP/s")

    # Find the "knee" where throughput stops increasing
    max_tflops = max(r[2] for r in results)
    print(f"\n  Peak throughput: {max_tflops:.2f} TFLOP/s")
    print(f"  Throughput at batch=1: {results[0][2]:.2f} TFLOP/s  ({results[0][2]/max_tflops*100:.0f}% of peak)")
    print(f"\n  → Small batch = GPU under-occupied. Increase batch to hit peak throughput.")
    print(f"  → Profile with ncu: ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active ...")


# =============================================================================
# EXP 2 — Memory Coalescing: Stride-1 vs Stride-N Access
# =============================================================================
def exp_coalescing():
    separator("MEMORY COALESCING — Stride-1 vs Stride-N Access Patterns")
    print("""
  CONCEPT:
    When 32 threads in a warp access memory, the GPU can coalesce them
    into a single 128-byte transaction IF the addresses are contiguous
    (stride-1 = access every element: t0→addr[0], t1→addr[1], ...).

    With stride-4 (t0→addr[0], t1→addr[4], t2→addr[8], ...):
    The warp needs MULTIPLE transactions to cover the spread-out addresses.
    Effective memory bandwidth drops proportionally.

  PRACTICAL IMPACT:
    Transposed matrix access, scattered lookups, or wrong tensor layouts
    (e.g., NHWC vs NCHW) all cause poor coalescing and cut bandwidth.
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    n = 64 * 1024 * 1024    # 64M floats = 256MB
    src = torch.randn(n, device=device, dtype=torch.float32)

    print(f"  Source tensor: {n/1e6:.0f}M floats ({n*4/1e6:.0f}MB)")
    print(f"  {'Stride':>8}  {'Time (ms)':>12}  {'Bandwidth (GB/s)':>18}  {'Efficiency':>12}")
    print(f"  {'─'*8}  {'─'*12}  {'─'*18}  {'─'*12}")

    baseline_bw = None
    for stride in [1, 2, 4, 8, 16, 32]:
        # Access every stride-th element
        indices = torch.arange(0, n // stride, device=device) * stride
        indices = indices[:min(len(indices), n // stride)]

        ms = cuda_time(lambda: src[indices])
        # Bytes read = number of elements accessed × 4 bytes
        bytes_accessed = len(indices) * 4
        bw_gbs = bytes_accessed / (ms / 1000) / 1e9

        if stride == 1:
            baseline_bw = bw_gbs
        eff = f"{bw_gbs/baseline_bw*100:.0f}%" if baseline_bw else "—"

        print(f"  {stride:>8}  {ms:>12.3f}  {bw_gbs:>18.1f}  {eff:>12}")

    print(f"\n  → Stride-1 (coalesced) achieves peak bandwidth.")
    print(f"  → Stride-32 loses ~{100-float(eff[:-1]):.0f}% of bandwidth.")
    print(f"  → ncu metric to check: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum")


# =============================================================================
# EXP 3 — Warp Divergence
# =============================================================================
def exp_divergence():
    separator("WARP DIVERGENCE — Branch Overhead in Parallel Kernels")
    print("""
  CONCEPT:
    All 32 threads in a warp execute the same instruction simultaneously.
    When threads branch differently (if some threads take 'if' and others
    take 'else'), the GPU must SERIALISE both branches:
      - Execute 'if' branch with 'else' threads masked OFF
      - Execute 'else' branch with 'if' threads masked OFF
    This halves (or worse) effective throughput.

  WE SIMULATE:
    A = no branching (all threads do same work)
    B = 50% branching (half threads take if, half take else)
    C = predicated (compiler turns if/else into conditional move — no divergence)
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    n = 16 * 1024 * 1024
    x = torch.randn(n, device=device)

    # A: No divergence — all threads apply the same operation
    def no_divergence():
        return torch.relu(x)                  # uniform operation: all threads = same path

    # B: Simulated divergence — conditional on value (some positive, some negative)
    # In practice, the compiler may eliminate this with predication.
    # For true divergence in a custom kernel, you'd need a CUDA C kernel.
    def simulated_divergence():
        return torch.where(x > 0, x * 2.0, x * 0.5)   # two different ops per thread

    # C: abs() — compiler uses abs instruction, no branching
    def no_branch_abs():
        return torch.abs(x)

    ms_a = cuda_time(no_divergence)
    ms_b = cuda_time(simulated_divergence)
    ms_c = cuda_time(no_branch_abs)

    print(f"  No divergence (relu)     : {ms_a:.3f}ms")
    print(f"  Divergent paths (where)  : {ms_b:.3f}ms  ({ms_b/ms_a:.2f}x slower)")
    print(f"  No-branch (abs)          : {ms_c:.3f}ms")
    print(f"\n  → In production: replace if/else in hot paths with torch.where,")
    print(f"     masked operations, or lookup tables.")
    print(f"  → ncu metric: sm__sass_average_branch_targets_threads_uniform.pct")


# =============================================================================
# EXP 4 — Tensor Cores: FP16 vs FP32 Matmul Throughput
# =============================================================================
def exp_tensorcores():
    separator("TENSOR CORES — FP16 (Tensor Core) vs FP32 (CUDA Core) Matmul")
    print("""
  CONCEPT:
    Tensor Cores are special execution units that perform 4×4 matrix
    multiply-accumulate in a single cycle. They only operate on:
      FP16 inputs  (Ampere: also BF16, TF32, INT8)

    Theoretical speedup: ~8–16× over FP32 CUDA cores for matrix multiplies.
    Real-world speedup for LLMs: typically 2–4× (memory bandwidth limits).

  FOR LLMs:
    Most compute in transformers is matmuls (attention, FFN, lm_head).
    Using FP16 (or BF16) lets ALL of these use Tensor Cores.
    This is why FP16 training/inference is the default for production LLMs.
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    print(f"  GPU: {props.name}")
    print(f"  Tensor Core support: {'Yes (Volta+)' if props.major >= 7 else 'No (too old)'}\n")

    results = []
    for M in [256, 512, 1024, 2048, 4096]:
        A32 = torch.randn(M, M, device=device, dtype=torch.float32)
        B32 = torch.randn(M, M, device=device, dtype=torch.float32)
        A16 = A32.half()
        B16 = B32.half()

        ms_f32 = cuda_time(lambda: torch.mm(A32, B32))
        ms_f16 = cuda_time(lambda: torch.mm(A16, B16))

        # Theoretical FLOPS for M×M matmul: 2M³
        flops   = 2 * M**3
        tf_f32  = flops / (ms_f32/1000) / 1e12
        tf_f16  = flops / (ms_f16/1000) / 1e12
        speedup = ms_f32 / ms_f16

        results.append((M, ms_f32, ms_f16, speedup))
        print(f"  M={M:4d}  FP32: {ms_f32:.3f}ms ({tf_f32:.2f}TF/s)  "
              f"FP16: {ms_f16:.3f}ms ({tf_f16:.2f}TF/s)  "
              f"speedup: {speedup:.2f}x")

    print(f"\n  → Use FP16/BF16 everywhere: automatic {results[-1][3]:.1f}x faster matmuls.")
    print(f"  → In PyTorch: model.half() or torch.autocast(device_type='cuda', dtype=torch.float16)")


# =============================================================================
# EXP 5 — Memory Hierarchy: L1 vs L2 vs DRAM Latency
# =============================================================================
def exp_memory_hierarchy():
    separator("MEMORY HIERARCHY — L1 Cache vs L2 Cache vs GPU DRAM")
    print("""
  GPU Memory Hierarchy (RTX 4060 approximate sizes):
    L1 / Shared Memory : ~128 KB per SM  — ~10–50 ns latency
    L2 Cache           : ~32 MB total    — ~100–200 ns latency
    GDDR6 DRAM         : ~8 GB total     — ~200–500 ns latency

  CONCEPT:
    Data that fits in L1/L2 cache is accessed much faster.
    Repeatedly accessing the same small tensor = cached = fast.
    Accessing a large tensor that exceeds cache = cache miss = slow.
    This is why batch size and sequence length matter for LLM performance.
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    print(f"  GPU: {props.name}")
    print(f"  L2 cache size: {props.L2_cache_size / 1e6:.0f}MB\n")

    print(f"  {'Size':>12}  {'Time (μs)':>12}  {'BW (GB/s)':>12}  {'Cache Level':>14}")
    print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*14}")

    l2_size = props.L2_cache_size

    for size_mb in [0.01, 0.1, 0.5, 1, 4, 16, 32, 64, 128]:
        n = int(size_mb * 1e6 / 4)   # number of float32 elements
        n = max(n, 1024)
        t = torch.randn(n, device=device, dtype=torch.float32)

        # Force a read of the entire tensor by computing a reduction
        ms  = cuda_time(lambda: t.sum())
        us  = ms * 1000
        bw  = n * 4 / (ms/1000) / 1e9   # GB/s

        actual_mb = n * 4 / 1e6
        if actual_mb < l2_size / 1e6 * 0.1:
            level = "L1/L2 hit"
        elif actual_mb < l2_size / 1e6:
            level = "L2 hit"
        else:
            level = "DRAM (miss)"

        print(f"  {actual_mb:>10.2f}MB  {us:>12.1f}  {bw:>12.1f}  {level:>14}")

    print(f"\n  → Small tensors (< L2) get ~10× higher bandwidth than DRAM.")
    print(f"  → Kernels that reuse data (tiling) exploit this hierarchy.")
    print(f"  → FlashAttention uses tiling to keep attention scores in SRAM.")


# =============================================================================
# EXP 6 — Kernel Fusion: Two separate ops vs one fused op
# =============================================================================
def exp_fusion():
    separator("KERNEL FUSION — Separate Kernels vs Fused Kernel")
    print("""
  CONCEPT:
    Every GPU kernel launch has overhead: ~5–20μs to schedule and start.
    More importantly, data between kernels passes through DRAM:
      Kernel 1 writes output to DRAM → Kernel 2 reads it from DRAM.

    Kernel fusion combines multiple ops into one kernel:
      - Single DRAM round trip (write once at the end)
      - Less kernel launch overhead
      - Enables register-level data sharing between ops

  EXAMPLE:
    Unfused:  x = linear(x)  # DRAM write
              x = relu(x)    # DRAM read + write
              x = dropout(x) # DRAM read + write
    Fused:    x = fused_linear_relu_dropout(x)  # one DRAM write at end

  torch.compile performs automatic kernel fusion using Triton.
  """)

    if device != "cuda":
        print("  (Requires CUDA)")
        return

    B, T, D = 32, 128, 512
    x = torch.randn(B, T, D, device=device, dtype=torch.float16)
    W = torch.randn(D, D, device=device, dtype=torch.float16)

    # Unfused: each operation is a separate kernel launch + DRAM round-trip
    def unfused():
        y = x @ W                      # kernel 1: matmul
        y = torch.relu(y)              # kernel 2: relu (reads result of kernel 1)
        y = y / (y.std() + 1e-5)       # kernel 3: normalise
        return y

    # Torch compile with default mode performs kernel fusion automatically
    # The fused version combines relu + normalise into fewer kernel calls
    fused = torch.compile(unfused, mode="default")

    # Warmup compiled version (triggers JIT compilation)
    for _ in range(5):
        _ = fused()
    torch.cuda.synchronize()

    ms_unfused = cuda_time(unfused)
    ms_fused   = cuda_time(fused)

    print(f"  Input shape: ({B}, {T}, {D})  FP16")
    print(f"  Unfused (3 kernels) : {ms_unfused:.3f}ms")
    print(f"  Fused (torch.compile): {ms_fused:.3f}ms  ({ms_unfused/ms_fused:.2f}x speedup)")
    print(f"\n  → torch.compile automatically fuses pointwise ops.")
    print(f"  → For attention: FlashAttention manually fuses Q@K softmax @V.")
    print(f"  → View fused kernels in nsys timeline — fewer kernel launches.")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  CUDA Kernel Experiments — Phase 2")
    print(f"{'='*60}")
    if device == "cuda":
        print(f"  GPU: {props.name}")
        print(f"  SMs: {props.multi_processor_count}")
        print(f"  VRAM: {props.total_memory/1e9:.1f}GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    print()

    exp_map = {
        "occupancy":         exp_occupancy,
        "coalescing":        exp_coalescing,
        "divergence":        exp_divergence,
        "tensorcores":       exp_tensorcores,
        "memory_hierarchy":  exp_memory_hierarchy,
        "fusion":            exp_fusion,
    }

    if args.exp == "all":
        for fn in exp_map.values():
            fn()
    else:
        exp_map[args.exp]()

    print(f"\n{'='*60}")
    print(f"  NEXT: Profile these with ncu to verify hardware counters:")
    print(f"  ncu --set basic python cuda_kernels.py --exp coalescing")
    print(f"  ncu --set roofline python cuda_kernels.py --exp tensorcores")
    print(f"{'='*60}")
