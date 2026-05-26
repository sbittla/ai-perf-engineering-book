"""
Exercise 4.1 — The CUDA Story: From Graphics to General-Purpose Computing

Chapter 4 (Background): How GPU Programming Became Accessible
Book: AI Systems Performance Engineering

Run:
    python 4.1_cuda_foundations.py

What this exercise does:
  1. Detects your GPU and reads SM count, warp size, and shared memory.
  2. Walks through the thread hierarchy (thread → warp → block → grid) with live numbers.
  3. Shows how warp divergence serialises execution and estimates overhead.
  4. Maps the full CUDA library stack: raw CUDA → Triton → cuBLAS/cuDNN → PyTorch.
  5. Traces the PyTorch dispatch chain from torch.mm() down to the cuBLAS kernel.
  6. Explains shared memory tiling and why it achieves near-peak bandwidth.

No TODOs here — this is a read-and-run exercise. Read each section's output,
understand what it means, then explore the chapters in Part II for hands-on work.
"""

import sys
import math

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Does this machine have a CUDA-capable GPU?
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1: GPU Detection")
print("=" * 60)

try:
    import torch
    if not torch.cuda.is_available():
        print("No CUDA GPU detected. Output below uses CPU-only mode.")
        print("Many sections will use reference values from an H100.")
        GPU_AVAILABLE = False
        gpu_name = "CPU (no GPU)"
        num_sms = 132          # H100 SXM5 reference
        warp_size = 32
        shared_mem_per_block = 48 * 1024
        compute_cap = (9, 0)
    else:
        GPU_AVAILABLE = True
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        num_sms = props.multi_processor_count
        warp_size = props.warp_size
        shared_mem_per_block = props.shared_memory_per_block
        compute_cap = (props.major, props.minor)

        print(f"GPU               : {gpu_name}")
        print(f"Streaming Multiprocessors (SMs): {num_sms}")
        print(f"Warp size         : {warp_size} threads")
        print(f"Shared memory/block: {shared_mem_per_block / 1024:.0f} KB")
        print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")
except ImportError:
    print("PyTorch not installed. Using H100 reference values.")
    GPU_AVAILABLE = False
    gpu_name = "Reference H100"
    num_sms = 132
    warp_size = 32
    shared_mem_per_block = 48 * 1024
    compute_cap = (9, 0)

print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: The CUDA Thread Hierarchy — Threads, Warps, Blocks, Grids
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 2: Thread Hierarchy")
print("=" * 60)

print("""
The CUDA execution model organises work in four nested levels:

  Thread  — the smallest unit of execution. Runs one instruction at a time.
  Warp    — 32 threads that execute in LOCKSTEP (SIMT: Single Instruction,
             Multiple Threads). The GPU scheduler always issues instructions
             to a whole warp, never to individual threads.
  Block   — a group of warps that share Shared Memory and can synchronise
             with __syncthreads(). Blocks run on a single SM.
  Grid    — the collection of all blocks launched by one kernel call.
             Blocks are scheduled across all SMs independently.
""")

# Show warp math concretely
threads_per_block = 256
warps_per_block = threads_per_block // warp_size
total_threads_in_kernel = 1_048_576          # 2^20, a common workload size
total_blocks = math.ceil(total_threads_in_kernel / threads_per_block)
total_warps = total_blocks * warps_per_block
blocks_per_sm = math.ceil(total_blocks / num_sms)

print(f"Example kernel launch: {total_threads_in_kernel:,} threads")
print(f"  threads_per_block = {threads_per_block}")
print(f"  warps_per_block   = {threads_per_block} / {warp_size} = {warps_per_block}")
print(f"  total_blocks      = {total_blocks:,}")
print(f"  total_warps       = {total_warps:,}")
print(f"  SMs available     = {num_sms}")
print(f"  blocks/SM (avg)   = {blocks_per_sm}")
print()
print("Key insight: the GPU hides memory latency by switching between warps.")
print("While one warp waits for a memory load (400-800 cycles), another warp")
print("runs. This is called LATENCY HIDING and it is the GPU's core trick.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Warp Divergence — The Silent Performance Killer
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 3: Warp Divergence")
print("=" * 60)

print("""
All 32 threads in a warp execute the SAME instruction at the same time.
If threads inside a warp take different paths through an if/else, the GPU
must run BOTH paths with some threads masked off — wasted work.

  // BAD: threads in the same warp will diverge
  if (thread_id % 2 == 0) {
      result = heavy_computation_A(data);   // even threads do this
  } else {
      result = heavy_computation_B(data);   // odd threads do this
  }
  // Cost: 2× the work of the longer branch

  // GOOD: divergence only across blocks, not within a warp
  if (block_id % 2 == 0) {
      result = heavy_computation_A(data);
  } else {
      result = heavy_computation_B(data);
  }
  // Cost: one branch per block — no serialisation within a warp
""")

divergent_fraction = 0.5
serialisation_overhead = 1 + divergent_fraction  # rough model
print(f"A kernel with {divergent_fraction:.0%} divergent warps runs ~{serialisation_overhead:.1f}× slower")
print("in the divergent sections compared to a fully uniform kernel.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: The CUDA Library Stack
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 4: The CUDA Library Stack")
print("=" * 60)

print("""
Most PyTorch operations never touch raw CUDA — they call into optimised
libraries that NVIDIA ships with every GPU driver:

┌─────────────────────────────────────────────────────────┐
│ Level          Library      What it does                 │
├─────────────────────────────────────────────────────────┤
│ Python         PyTorch      torch.mm, torch.conv2d, ...  │
│ C++ dispatch   ATen         aten::mm → mm_cuda           │
│ Math           cuBLAS       General matrix multiply       │
│ Deep learning  cuDNN        Fused conv/BN/ReLU, attn     │
│ Multi-GPU      NCCL         AllReduce, Broadcast          │
│ Custom kernels Triton       Python → PTX (FlashAttention) │
└─────────────────────────────────────────────────────────┘
""")

print("What this means for profiling:")
print("  When you see 'volta_sgemm_128x64_nn' in Nsight Systems, that is")
print("  cuBLAS choosing a tiled GEMM kernel for your matrix shape.")
print("  When you see 'cudnn_infer_ampere_...' that is cuDNN's fused")
print("  convolution+bias+activation kernel for the Ampere architecture.")
print("  You did not write these — but you need to recognise them.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: PyTorch Dispatch Chain
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 5: The PyTorch Dispatch Chain")
print("=" * 60)

print("""
  torch.mm(A, B)                     ← you write this
      │
      ▼
  Python dispatcher → selects backend (CUDA / CPU / MPS)
      │
      ▼
  aten::mm (C++ ATen operator)        ← defined in aten/src/ATen/native/
      │
      ▼
  at::native::mm_cuda(A, B)           ← CUDA-specific implementation
      │
      ▼
  cublasGemmEx(handle, ...)           ← cuBLAS call with tensor core flags
      │
      ▼
  GPU kernel: volta_sgemm_128x64_nn   ← chosen at runtime for your shape

The chain means: a slowdown can occur at ANY level.
  - Wrong dtype  → cuBLAS falls back from FP16 Tensor Core to FP32
  - Wrong shape  → cuBLAS picks a suboptimal tiling (padding waste)
  - Tiny batch   → kernel launch overhead dominates compute time
""")

# Show tensor core availability
cc = compute_cap[0] * 10 + compute_cap[1]
tensor_cores = cc >= 70
bf16 = cc >= 80
fp8 = cc >= 89

print(f"Your GPU (CC {compute_cap[0]}.{compute_cap[1]}):")
print(f"  Tensor Cores : {'YES — Volta+ (CC 7.0+)' if tensor_cores else 'NO — Pascal or older'}")
print(f"  BF16         : {'YES — Ampere+ (CC 8.0+)' if bf16 else 'NO'}")
print(f"  FP8          : {'YES — Ada/Hopper (CC 8.9+)' if fp8 else 'NO'}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Shared Memory — The Fast Scratchpad
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 6: Shared Memory vs Global Memory")
print("=" * 60)

print(f"""
Memory hierarchy on your GPU ({gpu_name}):

  Register file  : ~0 cycles  — per-thread, compiler-managed
  Shared memory  : ~30 cycles — per-block, programmer-managed scratchpad
  L2 cache       : ~200 cycles — on-chip, automatic
  Global (HBM)   : ~400-800 cycles — off-chip DRAM (the bandwidth bottleneck)

Shared memory per block: {shared_mem_per_block / 1024:.0f} KB

Tiled matrix multiplication uses shared memory like this:
  1. Load a tile of A and B from global memory into shared memory.
  2. Synchronise all threads in the block (__syncthreads).
  3. Compute the partial dot products using shared memory reads.
  4. Repeat for the next tile.

Without tiling: each element is loaded from global memory once per dot product.
With tiling   : each element is loaded once, then reused {int(math.sqrt(256))} times from shared memory.
                This is the key to achieving near-peak bandwidth utilisation.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Practical Takeaways for Performance Engineers
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 7: What This Means in Practice")
print("=" * 60)

print("""
You will rarely write raw CUDA kernels. But understanding the model lets you:

  1. READ profiler output accurately.
     Kernel names like 'ampere_sgemm_128x64_nn' tell you:
       - ampere = architecture family
       - sgemm  = single-precision GEMM (FP32, not Tensor Core)
       - 128x64 = tile shape (larger = better utilisation, usually)
       - nn     = no transpose on A or B

  2. DIAGNOSE low GPU utilisation.
     If SM occupancy is low, look for:
       - Too few warps per SM (block size too small)
       - Too much shared memory per block (limits concurrent blocks)
       - Register pressure spilling to local memory

  3. SPOT precision regressions.
     If you see 'sgemm' (FP32) where you expected 'hgemm' (FP16), your
     model has FP32 weights or activations that need .half() or autocast.

  4. UNDERSTAND Triton kernels.
     FlashAttention is implemented in Triton. When you see
     'triton__0d1d2d3...' in the profiler, it is a JIT-compiled Triton
     kernel. The number of registers and shared memory it uses determines
     how many blocks can run concurrently on each SM.
""")

print("Explore next: II.GPU_Programming_and_Profiling/4.The_CUDA_Execution_Model/")
print("              II.GPU_Programming_and_Profiling/5.GPU_Profiling/")
