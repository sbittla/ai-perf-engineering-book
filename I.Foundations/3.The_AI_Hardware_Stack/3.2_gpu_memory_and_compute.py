#!/usr/bin/env python3
"""
3.2_gpu_memory_and_compute.py  ─  Chapter 3: GPU Hardware Internals
=============================================================================
Covers book section 3.3:
  • HBM bandwidth measurement (how fast can we saturate the memory bus?)
  • FP32 vs FP16 vs BF16 throughput — Tensor Core activation
  • Arithmetic intensity transition: when do Tensor Cores help?
  • SM utilization and occupancy concepts
  • The GPU memory hierarchy: registers → shared → L2 → HBM

Run:  python 3.2_gpu_memory_and_compute.py
Sections 1–4 require CUDA.  Section 5 runs on CPU for the concepts.
"""

import time
import torch
import torch.nn as nn

print("=" * 65)
print("  Exercise 02 — GPU Memory & Compute Internals")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

if DEVICE != "cuda":
    print("  Most sections require CUDA.  Install CUDA and re-run.")
    print("  Reading through the exercise still teaches the concepts.\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: HBM bandwidth — how fast can we read memory?
# ─────────────────────────────────────────────────────────────
print("── Section 1: HBM Bandwidth Measurement ──")
print("""
  A bandwidth-saturating kernel reads every byte of a large tensor once
  and does minimal arithmetic.  The measured throughput approximates the
  HBM bandwidth ceiling for memory-bound kernels.

  vector_add: C = A + B  (reads 2N bytes, writes N bytes, 1 FLOP per element)
  This is the standard 'STREAM benchmark' for GPUs.
""")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}")

    SIZES_MB = [64, 256, 1024]

    print(f"\n  {'Size':>8}  {'BW (GB/s)':>10}  {'Time (ms)':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")

    peak_bw_measured = 0.0
    for size_mb in SIZES_MB:
        n = size_mb * 1_000_000 // 4   # float32 elements
        A = torch.randn(n, device=DEVICE, dtype=torch.float32)
        B = torch.randn(n, device=DEVICE, dtype=torch.float32)

        # Warmup
        for _ in range(5): C = A + B
        torch.cuda.synchronize()

        # TODO 1: Measure the time for 20 iterations of C = A + B
        #   Use CUDA events (start.record(), ..., end.record(), synchronize)
        ITERS = 20
        avg_ms = None  # YOUR CODE HERE

        # TODO 2: Compute memory bandwidth in GB/s
        #   Bytes moved: read A (size_mb MB) + read B (size_mb MB) + write C (size_mb MB)
        #   bw_gb_s = (3 * size_mb / 1000) / (avg_ms / 1000)
        bw_gb_s = None  # YOUR CODE HERE

        assert avg_ms  is not None, f"measure time for {size_mb} MB"
        assert bw_gb_s is not None, f"compute bandwidth for {size_mb} MB"
        peak_bw_measured = max(peak_bw_measured, bw_gb_s)
        print(f"  {size_mb:>6} MB  {bw_gb_s:>10.1f}  {avg_ms:>10.3f}")

    print(f"\n  Peak measured HBM bandwidth: {peak_bw_measured:.1f} GB/s")
    print(f"  Compare to spec: look up '{props.name} memory bandwidth'")
    print("  ✓ Section 1 passed")
else:
    peak_bw_measured = 50.0   # placeholder
    print("  (CUDA not available — Section 1 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Tensor Core activation — FP32 vs FP16 throughput
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: FP32 vs FP16 vs BF16 Compute Throughput ──")
print("""
  Tensor Cores are special matrix multiply units:
    FP32 (no TC): CUDA cores only — baseline throughput
    TF32  (TC):   PyTorch enables by default on Ampere+, ~8× speedup
    FP16  (TC):   ~16× speedup vs FP32 CUDA cores
    BF16  (TC):   same throughput as FP16, same exponent range as FP32

  Tensor Cores activate when:
    (a) dtype is float16 or bfloat16  (or TF32 is enabled for float32)
    (b) matrix dimensions are multiples of 8 (ideally 16 or 64)
    (c) tensors are contiguous and aligned

  Check: look for gemm, hmma, or h884 kernels in ncu output.
""")

if DEVICE == "cuda":
    M = 4096   # large enough to saturate Tensor Cores

    def bench_matmul(dtype, iters=50, label=""):
        A = torch.randn(M, M, device=DEVICE, dtype=dtype)
        B = torch.randn(M, M, device=DEVICE, dtype=dtype)
        for _ in range(10): torch.mm(A, B)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): torch.mm(A, B)
        e.record()
        torch.cuda.synchronize()
        ms  = s.elapsed_time(e) / iters
        # FLOPs = 2 * M^3 for square matmul
        flops = 2 * M ** 3
        tflops = flops / (ms / 1000) / 1e12
        if label:
            print(f"  {label:<10}: {ms:.3f} ms  →  {tflops:.1f} TFLOP/s")
        return ms, tflops

    t_fp32, tf_fp32 = bench_matmul(torch.float32, label="FP32")

    # TODO 3: Benchmark FP16 matmul (torch.float16) — call bench_matmul
    t_fp16, tf_fp16 = None, None  # YOUR CODE HERE

    # TODO 4: Benchmark BF16 matmul (torch.bfloat16) — call bench_matmul
    t_bf16, tf_bf16 = None, None  # YOUR CODE HERE

    assert t_fp16 is not None, "benchmark FP16"
    assert t_bf16 is not None, "benchmark BF16"

    # TODO 5: Compute FP16 speedup over FP32
    fp16_speedup = None  # YOUR CODE HERE  → t_fp32 / t_fp16

    assert fp16_speedup is not None, "compute fp16_speedup"
    print(f"\n  FP16 speedup over FP32: {fp16_speedup:.1f}×")
    print(f"  Expected: 2–16× depending on GPU generation and Tensor Core support")
    assert fp16_speedup > 1.0, "FP16 should be faster than FP32 on a Tensor Core GPU"
    print("  ✓ Section 2 passed")
else:
    print("  (CUDA not available — Section 2 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Matrix size and Tensor Core alignment
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Matrix Size Alignment for Tensor Cores ──")
print("""
  Tensor Cores operate on tiles of specific sizes (multiples of 8 or 16).
  Non-aligned dimensions get padded internally, wasting SM cycles.
  Transformer architectures (GPT-2, LLaMA) are designed with dim=64×N
  to stay aligned.

  Compare: M=4096 (aligned) vs M=4095 (not aligned).
""")

if DEVICE == "cuda":
    def bench_dim(dim, dtype=torch.float16, iters=30):
        A = torch.randn(dim, dim, device=DEVICE, dtype=dtype)
        B = torch.randn(dim, dim, device=DEVICE, dtype=dtype)
        for _ in range(5): torch.mm(A, B)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): torch.mm(A, B)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    dims_to_test = [512, 511, 1024, 1023, 2048, 2047]

    print(f"  {'Dim':>6}  {'Aligned?':>9}  {'Time (ms)':>10}  {'Relative':>10}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*10}")

    ref_times = {}
    for dim in dims_to_test:
        # TODO 6: Determine if dim is a multiple of 8
        is_aligned = None  # YOUR CODE HERE  → dim % 8 == 0

        t = bench_dim(dim)
        base = ref_times.get(dim + 1, t)  # compare odd vs even
        ref_times[dim] = t

        assert is_aligned is not None, f"compute is_aligned for dim={dim}"
        print(f"  {dim:>6}  {'YES' if is_aligned else 'no':>9}  {t:>10.3f}  {'baseline' if is_aligned else f'{t/ref_times.get(dim+1,t):.2f}× slower'}")

    print("  ✓ Section 3 passed — always use dimensions that are multiples of 8")
else:
    print("  (CUDA not available — Section 3 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 4: GPU memory hierarchy — HBM vs shared memory
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: GPU Memory Hierarchy Concepts ──")
print("""
  GPU memory hierarchy (fastest to slowest):
    Registers     : per-thread, ~8 MB total, >10,000 GB/s  — private
    L1/Shared mem : per-SM,  128–256 KB, ~20,000 GB/s  — shared in block
    L2 cache      : per-GPU, 40–60 MB,    ~5,000 GB/s  — shared by all SMs
    HBM (global)  : per-GPU, 8–80 GB,   272–3,350 GB/s — main GPU memory

  FlashAttention achieves its speedup by tiling attention computation to
  fit Q, K, V tiles in shared memory — avoiding repeated HBM reads.
  The speedup is not from doing fewer FLOPs; it is from doing fewer
  bytes of HBM traffic.

  Kernel fusion (torch.compile) achieves the same pattern:
  instead of writing intermediate results to HBM and reading them back,
  fused kernels keep intermediates in registers/shared memory.
""")

if DEVICE == "cuda":
    # Demonstrate: elementwise ops back-to-back vs a single fused kernel
    # Unfused: ReLU → Dropout → LayerNorm  → 3 HBM round-trips
    # Fused:   torch.compile fuses them    → 1 HBM round-trip

    N = 4_000_000   # 16 MB tensor

    x = torch.randn(N, device=DEVICE)

    relu    = nn.ReLU()
    dropout = nn.Dropout(p=0.0)   # p=0 so output is identical
    norm    = nn.LayerNorm(N, device=DEVICE)

    # Unfused path (3 kernels, 3 HBM round-trips)
    def unfused():
        y = relu(x)
        y = dropout(y)
        y = norm(y)
        return y

    # TODO 7: Create a torch.compile'd version of the unfused sequence
    #   fused = torch.compile(unfused)
    #   Note: first call will be slow (compilation) — always warmup
    fused = None  # YOUR CODE HERE  → torch.compile(unfused)

    assert fused is not None, "create compiled version"

    # Warmup both (especially important for compiled version)
    for _ in range(5): unfused()
    for _ in range(5): fused()   # first 2-3 calls compile
    torch.cuda.synchronize()

    def time_fn(fn, iters=50):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_unfused = time_fn(unfused)
    t_fused   = time_fn(fused)

    # TODO 8: Compute speedup from fusion
    fusion_speedup = None  # YOUR CODE HERE  → t_unfused / t_fused

    assert fusion_speedup is not None, "compute fusion_speedup"
    print(f"  Tensor size    : {N * 4 / 1e6:.0f} MB  (float32)")
    print(f"  Unfused (3 kernels) : {t_unfused:.3f} ms")
    print(f"  Fused   (compiled)  : {t_fused:.3f} ms")
    print(f"  Fusion speedup      : {fusion_speedup:.2f}×")
    print(f"  Speedup comes from fewer HBM round-trips, not fewer FLOPs")
    print("  ✓ Section 4 passed")
else:
    print("  (CUDA not available — Section 4 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 5: SM occupancy — keeping the GPU busy
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: SM Occupancy and GPU utilization ──")
print("""
  SM occupancy = active warps / maximum warps per SM.
  High occupancy → the SM can hide memory latency by switching warps.
  Low occupancy  → the SM stalls waiting for memory, wasting cycles.

  Causes of low occupancy:
    • Small batch size (few active warps)
    • High register usage per thread (limits warps per SM)
    • Large shared memory allocation (limits blocks per SM)
    • synchronization barriers (all warps wait together)

  Practical impact: running a model with batch_size=1 at inference
  typically results in SM occupancy of 5–20%.  Continuous batching
  raises this to 50–80% by processing multiple requests together.

  Measurement:  ncu --metrics sm__occupancy.avg.pct_of_peak_sustained_active
                    python my_script.py
""")

if DEVICE == "cuda":
    # Demonstrate occupancy indirectly: compare latency at different batch sizes
    model5 = nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 128),
    ).to(DEVICE)
    model5.eval()

    print(f"  {'Batch':>6}  {'Latency (ms)':>13}  {'Throughput':>14}  {'Relative tput':>14}")
    print(f"  {'-'*6}  {'-'*13}  {'-'*14}  {'-'*14}")

    throughputs = {}
    for bs in [1, 4, 16, 64, 256]:
        x5 = torch.randn(bs, 512, device=DEVICE)
        with torch.no_grad():
            for _ in range(10): model5(x5)
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ITERS = 50
        s.record()
        with torch.no_grad():
            for _ in range(ITERS): model5(x5)
        e.record()
        torch.cuda.synchronize()
        avg_ms = s.elapsed_time(e) / ITERS

        # TODO 9: Compute throughput in samples/sec
        throughput = None  # YOUR CODE HERE  → bs / (avg_ms / 1000)

        assert throughput is not None, f"compute throughput for bs={bs}"
        throughputs[bs] = throughput
        rel = throughput / throughputs[1] if 1 in throughputs else 1.0
        print(f"  {bs:>6}  {avg_ms:>13.3f}  {throughput:>12.0f}/s  {rel:>13.1f}×")

    print(f"\n  At bs=1: GPU is under-utilized (low SM occupancy)")
    print(f"  At bs=256: GPU approaches full utilization")
    print(f"  This is why continuous batching exists for LLM inference")
    print("  ✓ Section 5 passed")
else:
    print("  (CUDA not available — Section 5 skipped)")

print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Exercise 02 (Chapter 3) done!")
print()
print("  Key takeaways:")
print("    • HBM is the memory bus — measure its bandwidth to know your ceiling")
print("    • FP16 Tensor Cores give 2–16× speedup — use them with AMP")
print("    • Matrix dimensions must be multiples of 8 for full Tensor Core use")
print("    • Kernel fusion (torch.compile) reduces HBM traffic — the real speedup")
print("    • Small batch → low SM occupancy → wasted GPU — use continuous batching")
print("=" * 65)
