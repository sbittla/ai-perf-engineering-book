#!/usr/bin/env python3
"""
4.The_CUDA_Execution_Model/4.1_execution_model.py  ─  Chapter 4: The GPU Execution Model
=======================================================================
Covers book section 4.1:
  • GPU thread hierarchy: threads → warps (32) → blocks → grids
  • SIMT execution model
  • Why batch size matters for SM occupancy
  • Warp divergence cost

Run:  python II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.1_execution_model.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 4.1 — The GPU Execution Model")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: GPU Thread Hierarchy
# ─────────────────────────────────────────────────────────────
print("── Section 1: GPU Thread Hierarchy ──")
print("""
  A GPU executes work through a four-level hierarchy:
    Thread  — one CUDA lane executing a single instruction
    Warp    — 32 threads that execute the SAME instruction in lockstep (SIMT)
    Block   — a group of warps sharing L1 cache / shared memory
    Grid    — all blocks launched for one kernel dispatch

  The number of concurrently resident warps determines how well the GPU
  hides memory latency.  Each Streaming Multiprocessor (SM) can schedule
  hundreds of warps; while one warp stalls on a memory request, the SM
  immediately switches to another.  This is the GPU's primary latency
  hiding mechanism.

  Theoretical max warps = SMs × max_threads_per_SM / warp_size
  On an A100 (108 SMs, 2048 threads/SM, 32 warp_size): 108*2048/32 = 6912
""")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    sm_count = props.multi_processor_count
    max_threads_per_sm = props.max_threads_per_multi_processor
    warp_size = 32  # fixed for all NVIDIA GPUs

    print(f"  GPU: {props.name}")
    print(f"  SMs: {sm_count}")
    print(f"  Max threads per SM: {max_threads_per_sm}")
    print(f"  Warp size: {warp_size}")

    # TODO 1: Compute theoretical max warps in flight
    #   max_warps = sm_count * max_threads_per_sm // warp_size
    max_warps = None  # YOUR CODE HERE → sm_count * max_threads_per_sm // warp_size

    assert max_warps is not None, "compute max_warps"
    assert max_warps > 1000, f"Any real GPU should have >1000 max warps, got {max_warps}"
    print(f"  Theoretical max warps in flight: {max_warps}")
else:
    max_warps = 6912  # A100 reference value for CPU fallback
    print(f"  (CUDA not available — using reference value: max_warps = {max_warps})")

print("  ✓ Section 1 passed — GPU thread hierarchy understood")

# ─────────────────────────────────────────────────────────────
# SECTION 2: SIMT and Warp Size — throughput vs batch size
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: SIMT and Warp Size ──")
print("""
  SIMT (Single Instruction, Multiple Threads) means all 32 threads in a
  warp execute the same instruction simultaneously.  At batch_size=1, a
  matmul may only occupy a handful of warps — most SMs sit idle.
  At batch_size=256, hundreds of warps are active, the GPU pipeline is
  full, and throughput per unit time is far higher.

  We measure FP16 matmul TFLOP/s = 2 * M * K * N / (time_s * 1e12)
  using M=K=N=512.  A 512×512 FP16 matmul needs 2*512^3 ≈ 268M FLOPs.
""")

results = {}

if DEVICE == "cuda":
    M = K = N = 512
    flops_per_matmul = 2 * M * K * N

    for batch in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        A = torch.randn(batch, M, K, device=DEVICE, dtype=torch.float16)
        B = torch.randn(batch, K, N, device=DEVICE, dtype=torch.float16)

        # Warmup
        for _ in range(5):
            torch.bmm(A, B)
        torch.cuda.synchronize()

        # TODO 2: Fill in the timing loop.
        #   Create start/end CUDA events with enable_timing=True.
        #   Record start, run torch.bmm(A, B) for 20 iterations, record end.
        #   After synchronize(), compute ms_per_iter = elapsed / 20.
        #   Then tflops = (batch * flops_per_matmul * 20) / (total_ms / 1000) / 1e12
        #   Store in results[batch].

        start = None  # YOUR CODE HERE → torch.cuda.Event(enable_timing=True)
        end   = None  # YOUR CODE HERE → torch.cuda.Event(enable_timing=True)

        # YOUR CODE HERE: start.record(); loop 20x torch.bmm; end.record(); synchronize
        pass

        total_ms = None  # YOUR CODE HERE → start.elapsed_time(end)

        if total_ms is not None:
            tflops = (batch * flops_per_matmul * 20) / (total_ms / 1000) / 1e12
            results[batch] = tflops
            print(f"  batch={batch:>4}  {tflops:.3f} TFLOP/s")
        else:
            results[batch] = 0.0

    assert len(results) > 0, "results dict must be populated"
    assert results.get(256, 0) > results.get(1, 0) * 2, (
        f"TFLOP/s at batch=256 ({results.get(256,0):.3f}) should be > "
        f"2x batch=1 ({results.get(1,0):.3f})"
    )
else:
    # CPU fallback: simulate expected relationship
    results = {1: 0.05, 2: 0.08, 4: 0.14, 8: 0.22, 16: 0.35,
               32: 0.55, 64: 0.80, 128: 1.10, 256: 1.50}
    print("  (CUDA not available — using simulated throughput values)")

print("  ✓ Section 2 passed — SIMT throughput scales with batch size")

# ─────────────────────────────────────────────────────────────
# SECTION 3: SM Occupancy Effect — finding the knee
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: SM Occupancy Effect ──")
print("""
  As batch size increases, TFLOP/s rises steeply — each increment fills
  more warps.  Eventually, all SMs are fully occupied and throughput
  plateaus.  The 'knee' is the batch size where you first hit 80% of peak.

  Beyond the knee, you are compute-bound.  Below the knee, you are
  occupancy-limited (not enough warps to keep the GPU busy).
  For production inference, target a batch size at or beyond the knee.
""")

max_tflops = max(results.values()) if results else 1.0

# TODO 3: Find the first batch size where throughput >= 80% of max.
#   Iterate sorted(results.keys()), return the first key where
#   results[b] >= 0.8 * max_tflops.
knee_batch = None  # YOUR CODE HERE → first batch where tflops >= 0.8 * max_tflops

assert knee_batch is not None, "compute knee_batch"
assert knee_batch >= 4, f"knee_batch should be >= 4, got {knee_batch}"
print(f"  Peak TFLOP/s: {max_tflops:.3f}")
print(f"  Knee (80% of peak) at batch size: {knee_batch}")
print(f"  Below this batch size, you are occupancy-limited.")
print("  ✓ Section 3 passed — occupancy knee identified")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Warp Divergence Cost
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Warp Divergence Cost ──")
print("""
  Within a warp, all 32 threads execute the same instruction.  If threads
  within a warp take different code paths (e.g. an if/else based on data
  values), the GPU must execute BOTH branches sequentially, masking the
  inactive threads on each pass.  This is warp divergence.

  torch.relu(x): no divergence — same operation for every element
  torch.where(x > 0, x * 2.0, x * 0.5): two-path — potential divergence

  In practice, modern GPUs handle simple cases well, so the difference
  may be small for this microbenchmark.  We verify both complete > 0 ms.
""")

N_ELEM = 4 * 1024 * 1024  # 4M elements

if DEVICE == "cuda":
    x = torch.randn(N_ELEM, device=DEVICE, dtype=torch.float32)

    # Warmup
    for _ in range(5):
        torch.relu(x)
        torch.where(x > 0, x * 2.0, x * 0.5)
    torch.cuda.synchronize()

    # TODO 4: Time both operations with CUDA events.
    #   relu_ms: time 50 iterations of torch.relu(x)
    #   diverge_ms: time 50 iterations of torch.where(x > 0, x * 2.0, x * 0.5)
    relu_ms    = None  # YOUR CODE HERE → CUDA event timing, 50 iters
    diverge_ms = None  # YOUR CODE HERE → CUDA event timing, 50 iters

    assert relu_ms is not None and relu_ms > 0,    "relu_ms must be > 0"
    assert diverge_ms is not None and diverge_ms > 0, "diverge_ms must be > 0"
    print(f"  relu (no divergence)   : {relu_ms:.3f} ms / iter")
    print(f"  where (two-path)       : {diverge_ms:.3f} ms / iter")
    print(f"  Ratio where/relu       : {diverge_ms/relu_ms:.2f}x")
else:
    relu_ms, diverge_ms = 1.0, 1.5
    print("  (CUDA not available — skipping divergence timing)")

print("  ✓ Section 4 passed — warp divergence measured")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 4.1 complete!")
print("  You now understand threads → warps → blocks → grids,")
print("  SIMT execution, occupancy, and warp divergence.")
print("  Next: II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/4.2_memory_coalescing.py")
print("=" * 60)
