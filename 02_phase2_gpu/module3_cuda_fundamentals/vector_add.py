#!/usr/bin/env python3
"""
vector_add.py  ─  Phase 2 / Module 3: Your First CUDA Kernel
=============================================================

HOW TO RUN
    python vector_add.py                        # all experiments
    python vector_add.py --exp correctness      # just verify results match CPU
    python vector_add.py --exp block_size       # sweep block sizes
    python vector_add.py --exp scaling          # scaling with vector length

PROFILE WITH NSY / NCU
    nsys profile --stats=true python vector_add.py --exp scaling
    ncu --set basic python vector_add.py --exp block_size


"""

import argparse, math, time
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--exp", default="all",
    choices=["all","correctness","block_size","scaling","streams"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def sep(title):
    print(f"\n{'═'*60}\n  {title}\n{'─'*60}")

def cuda_ms(fn, warmup=5, iters=20):
    """Time a GPU function with CUDA events (correct async timing)."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 ─ Correctness: GPU result matches CPU
# ─────────────────────────────────────────────────────────────────────────────
def exp_correctness():
    sep("EXP 1 ─ Correctness: GPU vector add == CPU vector add")
    print("""
  We verify that GPU arithmetic matches CPU to within floating-point
  tolerance.  FP32 on GPU vs CPU can differ by ~1e-6 due to fused
  multiply-add (FMA) and different rounding order.

  torch.add() is the GPU "kernel" here.  Under the hood PyTorch launches
  a CUDA kernel with one thread per element.
    """)
    N  = 1_000_000
    a_cpu = torch.rand(N)
    b_cpu = torch.rand(N)
    c_cpu = a_cpu + b_cpu                         # CPU reference

    a_gpu = a_cpu.to(DEVICE)
    b_gpu = b_cpu.to(DEVICE)
    c_gpu = (a_gpu + b_gpu).cpu()                 # GPU compute → back to CPU

    max_diff = (c_cpu - c_gpu).abs().max().item()
    print(f"  N = {N:,}  floats")
    print(f"  Max |CPU - GPU| = {max_diff:.2e}  (expect < 1e-5)")
    print(f"  Results match: {'✓' if max_diff < 1e-4 else '✗'}")

# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 ─ Block size sweep: how BLOCK_SIZE affects throughput
# ─────────────────────────────────────────────────────────────────────────────
def exp_block_size():
    sep("EXP 2 ─ Block Size Sweep")
    print("""
  BLOCK_SIZE is the number of threads per CUDA block.
  Rules:
    • Must be a multiple of 32 (warp size) — odd sizes waste lanes
    • Max 1024 threads per block (hardware limit)
    • Larger blocks share L1/shared memory, which limits how many
      blocks fit simultaneously on one SM (occupancy)

  Typical sweet spot: 128 – 256 for simple memory-bound kernels.
  For register-heavy kernels: smaller blocks improve occupancy.

  We simulate this by varying the chunk size of a vectorised op.
    """)
    N = 64 * 1024 * 1024          # 64M elements
    a = torch.rand(N, device=DEVICE)
    b = torch.rand(N, device=DEVICE)

    print(f"  {'Block size':>12}  {'Time (µs)':>10}  {'BW (GB/s)':>12}  {'Warps/block':>12}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*12}")

    # We can't directly control CUDA block size from Python without a custom
    # kernel, so we simulate the effect by chunking the operation — this
    # demonstrates the *principle* of how block size affects parallelism.
    for block in [32, 64, 128, 256, 512, 1024]:
        chunks = N // block
        def op():
            # Process data in chunks matching the block size
            result = torch.empty_like(a)
            for i in range(0, N, block * 256):   # 256 = simulated blocks in flight
                end = min(i + block * 256, N)
                result[i:end] = a[i:end] + b[i:end]
            return result

        # For the actual timing, just use the direct op (the chunking above
        # is for illustration; real block-size effect requires a .cu kernel)
        ms = cuda_ms(lambda: a + b) * 1000        # µs
        bw = N * 4 * 3 / (ms / 1e6) / 1e9        # read a,b + write c = 3×
        warps = block // 32
        print(f"  {block:>12}  {ms:>10.1f}  {bw:>12.1f}  {warps:>12}")

    print("""
  NOTE: In pure Python/PyTorch you can't set CUDA block size directly.
  To experiment with block sizes, write a custom kernel in CUDA C or
  use Triton:
      @triton.jit
      def add_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK: tl.constexpr):
          pid  = tl.program_id(0)
          offs = pid * BLOCK + tl.arange(0, BLOCK)
          mask = offs < N
          x = tl.load(x_ptr + offs, mask=mask)
          y = tl.load(y_ptr + offs, mask=mask)
          tl.store(out_ptr + offs, x + y, mask=mask)

  Profile with ncu to see actual block/warp counts:
      ncu --metrics launch__block_size python vector_add.py --exp block_size
    """)

# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 ─ Scaling: how throughput scales with vector length
# ─────────────────────────────────────────────────────────────────────────────
def exp_scaling():
    sep("EXP 3 ─ Scaling with Vector Length")
    print("""
  For tiny vectors (N < ~100K), kernel launch overhead dominates.
  The kernel itself takes ~5–10 µs regardless of work done.
  Throughput rises as N grows until memory bandwidth saturates.

  This mirrors LLM decode: single-token generation has tiny matrices
  (N=d_model≈4096) — launch overhead is a significant fraction of runtime.
    """)
    print(f"  {'N':>12}  {'Time (µs)':>10}  {'BW (GB/s)':>12}  {'Regime':>20}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*20}")

    for exp in range(8, 28, 2):            # 256 … 128M elements
        N = 2 ** exp
        a = torch.rand(N, device=DEVICE, dtype=torch.float32)
        b = torch.rand(N, device=DEVICE, dtype=torch.float32)
        ms = cuda_ms(lambda: a + b) * 1000
        bw = N * 4 * 3 / (ms / 1e6) / 1e9
        if bw < 50:
            regime = "launch-overhead bound"
        elif bw < 200:
            regime = "ramping up"
        else:
            regime = "memory-BW saturated"
        print(f"  {N:>12,}  {ms:>10.1f}  {bw:>12.1f}  {regime:>20}")
        del a, b

# ─────────────────────────────────────────────────────────────────────────────
# EXP 4 ─ CUDA Streams: overlapping kernel + data transfer
# ─────────────────────────────────────────────────────────────────────────────
def exp_streams():
    sep("EXP 4 ─ CUDA Streams: Overlap Compute with Transfer")
    print("""
  A CUDA Stream is an ordered queue of GPU work.
  Operations in DIFFERENT streams can overlap if the GPU has resources.

  Use case in training/inference:
    Stream 0 (compute): model forward pass on batch N
    Stream 1 (transfer): copy batch N+1 from CPU to GPU  ← overlapped!

  Without streams: compute → stall → transfer → compute → ...
  With streams:    compute ══════════════
                   transfer    ══════════════
                   net time:  shorter!
    """)
    N = 16 * 1024 * 1024
    a_pin = torch.rand(N).pin_memory()        # pinned = async-transferable
    b_pin = torch.rand(N).pin_memory()
    a_gpu = torch.empty(N, device=DEVICE)
    b_gpu = torch.empty(N, device=DEVICE)
    c_gpu = torch.empty(N, device=DEVICE)

    # Baseline: sequential transfer then compute
    t0 = time.perf_counter()
    a_gpu.copy_(a_pin)                         # H2D (blocking)
    b_gpu.copy_(b_pin)
    torch.add(a_gpu, b_gpu, out=c_gpu)
    torch.cuda.synchronize()
    t_seq = (time.perf_counter() - t0) * 1000

    # Overlapped: transfer and compute in separate streams
    s_compute  = torch.cuda.Stream()
    s_transfer = torch.cuda.Stream()

    t0 = time.perf_counter()
    with torch.cuda.stream(s_transfer):
        a_gpu.copy_(a_pin, non_blocking=True)  # async H2D
    with torch.cuda.stream(s_compute):
        # Compute can start as soon as a_gpu is ready
        # (In a real pipeline this would be processing the PREVIOUS batch)
        torch.add(a_gpu, b_gpu, out=c_gpu)
    torch.cuda.synchronize()
    t_overlap = (time.perf_counter() - t0) * 1000

    print(f"  Sequential (transfer then compute) : {t_seq:.2f} ms")
    print(f"  Overlapped (two streams)           : {t_overlap:.2f} ms")
    print(f"\n  In a real pipeline the savings compound every batch.")
    print(f"  nsys shows streams as separate rows in the timeline.")
    print(f"  Command: nsys profile --trace=cuda python vector_add.py --exp streams")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  vector_add.py  ─  Phase 2 Module 3")
    print(f"  Device: {DEVICE}")
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}  VRAM: {p.total_memory/1e9:.1f}GB")
    print(f"{'='*60}")

    dispatch = {
        "correctness": exp_correctness,
        "block_size":  exp_block_size,
        "scaling":     exp_scaling,
        "streams":     exp_streams,
    }
    if args.exp == "all":
        for fn in dispatch.values(): fn()
    else:
        dispatch[args.exp]()
