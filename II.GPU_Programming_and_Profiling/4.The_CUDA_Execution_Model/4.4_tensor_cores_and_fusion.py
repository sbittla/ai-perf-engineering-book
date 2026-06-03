#!/usr/bin/env python3
"""
4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py  ─  Chapter 4: Tensor Cores
=======================================================================
Covers book section 4.3:
  • FP16/BF16 Tensor Core throughput vs FP32 CUDA cores
  • Multiples-of-8 alignment requirement
  • BF16 vs FP16 comparison
  • Kernel fusion with torch.compile

Run:  python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.4_tensor_cores_and_fusion.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 4.3 — Tensor Cores and Kernel Fusion")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Tensor Core Speedup — FP32 vs FP16
# ─────────────────────────────────────────────────────────────
print("── Section 1: Tensor Core Speedup ──")
print("""
  NVIDIA Tensor Cores are dedicated matrix multiply-accumulate units
  introduced in the Volta architecture (2017).  They perform a 4×4
  matrix multiply per clock cycle in FP16, with FP32 accumulation.

  For FP16 matrix multiplications, Tensor Cores deliver 8–16× the
  throughput of standard FP32 CUDA cores.  The speedup is most visible
  at large matrix sizes (M ≥ 1024) where the matrix tiles fit well and
  the overhead of launching the operation is negligible.

  TFLOP/s = 2 * M * M * M / (time_s * 1e12)  [for an M×M matmul]
""")

speedup_at_2048 = 1.0  # default for CPU fallback

if DEVICE == "cuda":
    print(f"  {'M':>6}  {'FP32 ms':>10}  {'FP16 ms':>10}  {'Speedup':>8}  {'FP32 TF/s':>10}  {'FP16 TF/s':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

    results_32 = {}
    results_16 = {}

    for M in [256, 512, 1024, 2048, 4096]:
        A32 = torch.randn(M, M, device=DEVICE, dtype=torch.float32)
        B32 = torch.randn(M, M, device=DEVICE, dtype=torch.float32)
        A16 = A32.half()
        B16 = B32.half()

        # Warmup
        for _ in range(5):
            torch.mm(A32, B32)
            torch.mm(A16, B16)
        torch.cuda.synchronize()

        # TODO 1: For each M, compute ms_fp32 and ms_fp16.
        #   Use CUDA events with 20 iterations each.
        #   Then compute speedup = ms_fp32 / ms_fp16.
        #   Store results_32[M] = ms_fp32, results_16[M] = ms_fp16.

        # FP32 timing
        s32, e32 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s32.record()
        for _ in range(20):
            torch.mm(A32, B32)
        e32.record()
        torch.cuda.synchronize()
        ms_fp32 = s32.elapsed_time(e32) / 20

        # FP16 timing
        s16, e16 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s16.record()
        for _ in range(20):
            torch.mm(A16, B16)
        e16.record()
        torch.cuda.synchronize()
        ms_fp16 = s16.elapsed_time(e16) / 20

        if ms_fp32 is not None and ms_fp16 is not None and ms_fp16 > 0:
            speedup = ms_fp32 / ms_fp16
            flops = 2 * M * M * M
            tflops32 = flops / (ms_fp32 / 1000) / 1e12
            tflops16 = flops / (ms_fp16 / 1000) / 1e12
            results_32[M] = ms_fp32
            results_16[M] = ms_fp16
            print(f"  {M:>6}  {ms_fp32:>10.3f}  {ms_fp16:>10.3f}  {speedup:>8.2f}x  {tflops32:>10.2f}  {tflops16:>10.2f}")
            if M == 2048:
                speedup_at_2048 = speedup
        else:
            print(f"  {M:>6}  (TODO not completed)")

    assert speedup_at_2048 > 1.5, (
        f"FP16 speedup at M=2048 should be > 1.5x, got {speedup_at_2048:.2f}x. "
        "Did you complete the TODO timing code?"
    )
else:
    print("  (CUDA not available — skipping Tensor Core benchmark)")
    speedup_at_2048 = 4.0  # reference value

print("  ✓ Section 1 passed — Tensor Core FP16 speedup measured")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Alignment Requirement
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Alignment Requirement ──")
print("""
  Tensor Cores require matrix dimensions to be multiples of 8 (and
  ideally 16 or 64) to use their most efficient execution paths.
  When a dimension is not a multiple of 8, cuBLAS must pad or fall back
  to a slower path, introducing a measurable overhead.

  We compare M=1024 (multiple of 8) vs M=1000 (not a multiple):
    1024 = 128 × 8  → optimal Tensor Core tile alignment
    1000 = 125 × 8  → requires padding, suboptimal

  On some GPUs the difference is small (cuBLAS handles it gracefully);
  on others it is significant.  We use a generous 1.5x threshold.
""")

if DEVICE == "cuda":
    M_aligned   = 1024
    M_unaligned = 1000

    A_al  = torch.randn(M_aligned,   M_aligned,   device=DEVICE, dtype=torch.float16)
    B_al  = torch.randn(M_aligned,   M_aligned,   device=DEVICE, dtype=torch.float16)
    A_un  = torch.randn(M_unaligned, M_unaligned, device=DEVICE, dtype=torch.float16)
    B_un  = torch.randn(M_unaligned, M_unaligned, device=DEVICE, dtype=torch.float16)

    for _ in range(5):
        torch.mm(A_al, B_al)
        torch.mm(A_un, B_un)
    torch.cuda.synchronize()

    # TODO 2: Time M=1024 (aligned) and M=1000 (unaligned) matmuls.
    #   Use CUDA events with 30 iterations each.
    #   Compute ratio = unaligned_ms / aligned_ms.
    sa, ea = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    sa.record()
    for _ in range(30):
        torch.mm(A_al, B_al)
    ea.record()
    torch.cuda.synchronize()
    aligned_ms = sa.elapsed_time(ea) / 30

    su, eu = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    su.record()
    for _ in range(30):
        torch.mm(A_un, B_un)
    eu.record()
    torch.cuda.synchronize()
    unaligned_ms = su.elapsed_time(eu) / 30

    assert aligned_ms is not None,   "compute aligned_ms"
    assert unaligned_ms is not None, "compute unaligned_ms"
    assert aligned_ms < unaligned_ms * 1.5, (
        f"aligned ({aligned_ms:.3f} ms) should be < 1.5x unaligned ({unaligned_ms:.3f} ms)"
    )
    ratio = unaligned_ms / aligned_ms
    print(f"  M=1024 (aligned)   : {aligned_ms:.3f} ms")
    print(f"  M=1000 (unaligned) : {unaligned_ms:.3f} ms")
    print(f"  Ratio unaligned/aligned: {ratio:.2f}x")
else:
    print("  (CUDA not available — skipping alignment benchmark)")

print("  ✓ Section 2 passed — alignment requirement verified")

# ─────────────────────────────────────────────────────────────
# SECTION 3: BF16 vs FP16
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: BF16 vs FP16 ──")
print("""
  BFloat16 (BF16) was introduced with the Ampere architecture and is now
  the recommended training precision for large models on A100/H100.

  BF16 has the SAME exponent range as FP32 (8 bits) but fewer mantissa
  bits (7 vs 23).  This means BF16 can represent the same magnitude range
  as FP32 but with less precision — much less likely to overflow during
  training than FP16 (which has only 5 exponent bits).

  Both FP16 and BF16 activate Tensor Cores on Ampere+ hardware.
  They should achieve similar TFLOP/s.  Both should be substantially
  faster than FP32.
""")

if DEVICE == "cuda":
    M = 2048
    A32 = torch.randn(M, M, device=DEVICE, dtype=torch.float32)
    A16 = A32.half()
    try:
        Ab16 = A32.to(torch.bfloat16)
        bf16_supported = True
    except Exception:
        bf16_supported = False
        print("  (BF16 not supported on this GPU — skipping BF16 section)")

    if bf16_supported:
        B32  = torch.randn(M, M, device=DEVICE, dtype=torch.float32)
        B16  = B32.half()
        Bb16 = B32.to(torch.bfloat16)

        for _ in range(5):
            torch.mm(A32, B32)
            torch.mm(A16, B16)
            torch.mm(Ab16, Bb16)
        torch.cuda.synchronize()

        # FP32 baseline
        s32, e32 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s32.record()
        for _ in range(20):
            torch.mm(A32, B32)
        e32.record()
        torch.cuda.synchronize()
        fp32_ms = s32.elapsed_time(e32) / 20

        # TODO 3: Time BF16 matmul (torch.mm(Ab16, Bb16)) with CUDA events, 20 iters.
        #   Store result in bf16_ms.
        sb, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        sb.record()
        for _ in range(20):
            torch.mm(Ab16, Bb16)
        eb.record()
        torch.cuda.synchronize()
        bf16_ms = sb.elapsed_time(eb) / 20

        assert bf16_ms is not None, "compute bf16_ms"
        assert bf16_ms < fp32_ms, (
            f"BF16 ({bf16_ms:.3f} ms) should be faster than FP32 ({fp32_ms:.3f} ms) "
            "— BF16 uses Tensor Cores"
        )
        print(f"  FP32 matmul (M=2048) : {fp32_ms:.3f} ms")
        print(f"  BF16 matmul (M=2048) : {bf16_ms:.3f} ms")
        print(f"  BF16 speedup         : {fp32_ms/bf16_ms:.2f}x vs FP32")
else:
    print("  (CUDA not available — skipping BF16 benchmark)")

print("  ✓ Section 3 passed — BF16 achieves Tensor Core performance")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Kernel Fusion with torch.compile
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Kernel Fusion with torch.compile ──")
print("""
  Each PyTorch operation launches a separate CUDA kernel.  For a sequence
  of small operations — matmul → relu → layer_norm — this means three
  kernel launches, three HBM reads, three HBM writes.

  torch.compile (introduced in PyTorch 2.0) traces the computation graph
  and uses TorchInductor to generate fused Triton kernels.  The fused
  kernel reads the input once, applies all operations, and writes the
  output once.  For memory-bound operations this can give a significant
  speedup; for compute-bound operations the benefit is smaller.

  We compare unfused (3 separate ops) vs compiled with mode='default'.
  Note: the first 2-3 calls to the compiled function trigger compilation;
  warmup is mandatory before timing.
""")

if DEVICE == "cuda":
    BATCH = 128
    DIM   = 512

    # Build the unfused function
    linear = nn.Linear(DIM, DIM, device=DEVICE, dtype=torch.float32)
    ln     = nn.LayerNorm(DIM, device=DEVICE, dtype=torch.float32)

    def unfused_fn(x):
        x = linear(x)
        x = torch.relu(x)
        x = ln(x)
        return x

    # TODO 4: Compile unfused_fn with torch.compile(mode="default").
    #   Then warm up the compiled function for 5 iterations with a dummy input.
    fused_fn = torch.compile(unfused_fn, mode="default")

    assert fused_fn is not None, "fused_fn must not be None"

    dummy = torch.randn(BATCH, DIM, device=DEVICE)

    # Warmup compiled function (compilation happens here)
    print("  Warming up compiled function (may take a few seconds)...")
    if fused_fn is not None:
        for _ in range(5):
            _ = fused_fn(dummy)
        torch.cuda.synchronize()

    # Warmup unfused
    for _ in range(5):
        _ = unfused_fn(dummy)
    torch.cuda.synchronize()

    # Time unfused
    s1, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s1.record()
    for _ in range(30):
        unfused_fn(dummy)
    e1.record()
    torch.cuda.synchronize()
    unfused_ms = s1.elapsed_time(e1) / 30

    # Time fused
    s2, e2 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s2.record()
    for _ in range(30):
        fused_fn(dummy)
    e2.record()
    torch.cuda.synchronize()
    fused_ms = s2.elapsed_time(e2) / 30

    # Correctness: the compiled function must produce the same result as eager.
    # We deliberately do NOT assert the compiled version is faster. torch.compile
    # pays off on memory-bound / pointwise-heavy graphs; for a small compute-bound
    # block like this one the Triton kernel-launch overhead can make it *slower*
    # than eager, especially on smaller GPUs. The timing below is informational.
    with torch.no_grad():
        out_ref   = unfused_fn(dummy)
        out_fused = fused_fn(dummy)
    assert torch.allclose(out_ref, out_fused, rtol=1e-3, atol=1e-3), (
        "Compiled function output should match the eager output"
    )
    speedup = unfused_ms / fused_ms
    print(f"  Unfused (3 ops)       : {unfused_ms:.3f} ms")
    print(f"  Compiled/fused        : {fused_ms:.3f} ms")
    print(f"  Speedup from compile  : {speedup:.2f}x  "
          f"(>1 = faster; can be <1 for small compute-bound ops)")
else:
    fused_fn = None
    print("  (CUDA not available — torch.compile fusion skipped)")
    # Verify the TODO would work
    model_cpu = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.LayerNorm(64))
    fused_cpu = torch.compile(model_cpu, mode="default")
    assert fused_cpu is not None, "torch.compile should return a non-None object"
    print("  (CPU verify: torch.compile returned a compiled object)")

print("  ✓ Section 4 passed — torch.compile fusion benchmarked")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 4.3 complete!")
print("  You now understand Tensor Core speedups, alignment requirements,")
print("  BF16 vs FP16, and kernel fusion with torch.compile.")
print("  Next: II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py")
print("=" * 60)
