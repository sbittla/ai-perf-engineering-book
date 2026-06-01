#!/usr/bin/env python3
"""
1.1_roofline_model.py  ─  Chapter 1: The Roofline Model
================================================================
Covers book sections 1.1, 1.2, 1.5, 1.6:
  • Hardware numbers: peak FLOP/s and memory bandwidth
  • Arithmetic intensity: FLOPs per byte of memory traffic
  • Classifying operations as compute-bound or memory-bound
  • The ridge point — where memory-bound becomes compute-bound
  • Applying the roofline to LLM inference (GPT-2 worked example)
  • Measuring actual throughput and computing Model FLOPs utilization (MFU)

This exercise is conceptual + measurement.  It runs on CPU or GPU.
All TODO blocks are calculations you fill in; assertions verify correctness.

Run:  python 1.1_roofline_model.py
"""

import torch

print("=" * 65)
print("  Exercise 01 — The Roofline Model")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Hardware numbers — know your peak specs
# ─────────────────────────────────────────────────────────────
print("── Section 1: Hardware Specifications ──")
print("""
  Before applying the roofline model you need two numbers for your GPU:
    peak_flops_per_sec   : theoretical maximum FLOP/s (FP16 Tensor Core)
    mem_bandwidth_bytes  : HBM or GDDR bandwidth in bytes/sec

  These set the ceilings your kernels cannot exceed.
  The roofline's ridge point = peak_flops / mem_bandwidth  (FLOPs/byte)
""")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU name        : {props.name}")
    print(f"  SM count        : {props.multi_processor_count}")
    print(f"  VRAM            : {props.total_memory / 1e9:.1f} GB")

    # Known hardware specs (approximate FP16 Tensor Core peak)
    # You can look these up at developer.nvidia.com/cuda-gpus
    # or use tools like gpu_burn to measure empirically.
    GPU_SPECS = {
        # (peak FP16 TFLOP/s, HBM bandwidth GB/s)
        "NVIDIA A100-SXM4-40GB": (312e12, 1555e9),
        "NVIDIA A100 80GB PCIe": (312e12, 1935e9),
        "NVIDIA H100 80GB HBM3": (989e12, 3350e9),
        "NVIDIA GeForce RTX 4090": (165.2e12, 1008e9),
        "NVIDIA GeForce RTX 4070": (95.9e12, 504e9),
        "NVIDIA GeForce RTX 3090": (142.6e12, 936e9),
        "NVIDIA GeForce RTX 3080": (119.4e12, 760e9),
        "Tesla T4": (65.1e12, 300e9),
    }

    # Find a match (partial name match)
    peak_flops = None
    mem_bw     = None
    for key, (pf, mb) in GPU_SPECS.items():
        if key.lower() in props.name.lower() or props.name.lower() in key.lower():
            peak_flops, mem_bw = pf, mb
            print(f"  Matched spec    : {key}")
            break

    if peak_flops is None:
        print(f"  GPU '{props.name}' not in spec table — using RTX 4060 as fallback")
        peak_flops = 51.5e12   # RTX 4060 FP16
        mem_bw     = 272e9

else:
    # CPU roofline: use rough estimates for a modern server CPU
    print("  Running on CPU — using approximate server CPU specs")
    peak_flops = 2e12    # ~2 TFLOP/s (FP32, AVX-512 on a Xeon)
    mem_bw     = 50e9    # ~50 GB/s DDR5

# TODO 1: Compute the ridge point in FLOPs/byte
#   ridge_point = peak_flops / mem_bw
ridge_point = None  # YOUR CODE HERE

assert ridge_point is not None and ridge_point > 0, "compute ridge point"
print(f"\n  Peak FLOP/s     : {peak_flops/1e12:.1f} TFLOP/s")
print(f"  Memory BW       : {mem_bw/1e9:.0f} GB/s")
print(f"  Ridge point     : {ridge_point:.1f} FLOPs/byte")
print(f"  Interpretation  : operations with AI > {ridge_point:.0f} FLOPs/byte are compute-bound")
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Arithmetic intensity of common operations
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Arithmetic Intensity ──")
print("""
  Arithmetic intensity (AI) = FLOPs performed / bytes of memory traffic.
  Unit: FLOPs/byte.

  High AI → compute-bound (large GEMMs, convolutions with big batch sizes).
  Low AI  → memory-bandwidth-bound (elementwise ops, small-batch inference).
""")

def linear_layer_ai(in_features: int, out_features: int, batch_size: int,
                     dtype_bytes: int = 2) -> float:
    """
    Compute arithmetic intensity for one linear layer (GEMM).
    FLOPs = 2 * batch * in_features * out_features  (multiply-accumulate pairs)
    Bytes = weight matrix + input activations  (written-to output is ignored here)
    """
    flops      = 2 * batch_size * in_features * out_features
    bytes_read = (in_features * out_features * dtype_bytes +   # weights
                  batch_size  * in_features  * dtype_bytes)    # input
    return flops / bytes_read

def elementwise_ai(n_elements: int, n_ops: int = 1, dtype_bytes: int = 4) -> float:
    """
    Arithmetic intensity for an elementwise op (ReLU, GELU, Dropout, etc.)
    Each element: load 1 value, do n_ops FLOPs, store 1 value.
    """
    flops      = n_elements * n_ops
    bytes_read = n_elements * dtype_bytes   # load
    bytes_writ = n_elements * dtype_bytes   # store
    return flops / (bytes_read + bytes_writ)

# TODO 2: Compute AI for a GPT-2 small QKV projection at batch_size=1
#   in_features=768, out_features=768, dtype_bytes=2 (FP16)
ai_gpt2_bs1 = None  # YOUR CODE HERE

# TODO 3: Same projection at batch_size=32
ai_gpt2_bs32 = None  # YOUR CODE HERE

# TODO 4: Same projection at batch_size=512
ai_gpt2_bs512 = None  # YOUR CODE HERE

# TODO 5: AI for a ReLU over 1M float32 elements (elementwise, 1 op)
ai_relu = None  # YOUR CODE HERE

assert ai_gpt2_bs1   is not None, "compute ai_gpt2_bs1"
assert ai_gpt2_bs32  is not None, "compute ai_gpt2_bs32"
assert ai_gpt2_bs512 is not None, "compute ai_gpt2_bs512"
assert ai_relu       is not None, "compute ai_relu"
assert ai_relu < 1.0,             "ReLU should have very low AI (< 1 FLOPs/byte)"

print(f"  GPT-2 QKV projection (768→768, FP16):")
print(f"    batch=1   : {ai_gpt2_bs1:.2f} FLOPs/byte  "
      f"{'COMPUTE-BOUND' if ai_gpt2_bs1 > ridge_point else 'memory-bound'}")
print(f"    batch=32  : {ai_gpt2_bs32:.2f} FLOPs/byte  "
      f"{'COMPUTE-BOUND' if ai_gpt2_bs32 > ridge_point else 'memory-bound'}")
print(f"    batch=512 : {ai_gpt2_bs512:.2f} FLOPs/byte  "
      f"{'COMPUTE-BOUND' if ai_gpt2_bs512 > ridge_point else 'memory-bound'}")
print(f"  ReLU (1M elements, FP32):")
print(f"    AI        : {ai_relu:.2f} FLOPs/byte  memory-bound")
print("  ✓ Section 2 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Classify a set of operations
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Classify Operations ──")
print("""
  For each operation, compute its arithmetic intensity and classify it
  as 'compute-bound' (AI > ridge_point) or 'memory-bound' (AI ≤ ridge_point).
""")

ops = [
    # (name, AI FLOPs/byte)  — use your computed ridge_point to classify
    ("GPT-2 QKV projection (bs=1, FP16)",    linear_layer_ai(768,  768,    1, 2)),
    ("GPT-2 QKV projection (bs=64, FP16)",   linear_layer_ai(768,  768,   64, 2)),
    ("Large GEMM (bs=1024, 1024→4096, FP16)",linear_layer_ai(1024, 4096, 1024, 2)),
    ("ReLU (1M elements, FP32)",              elementwise_ai(1_000_000, 1, 4)),
    ("Dropout (1M elements, FP32)",           elementwise_ai(1_000_000, 1, 4)),
]

print(f"  {'Operation':<45}  {'AI':>8}  {'Bound'}")
print(f"  {'-'*45}  {'-'*8}  {'-'*16}")
for name, ai in ops:
    # TODO 6: Fill in the bound for each operation
    #   bound = "compute-bound" if ai > ridge_point else "memory-bound"
    bound = None  # YOUR CODE HERE
    assert bound is not None, f"fill in bound for {name}"
    print(f"  {name:<45}  {ai:>8.2f}  {bound}")

print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Measure actual throughput and compute MFU
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Measuring Throughput and MFU ──")
print("""
  Model FLOPs utilization (MFU) = observed FLOP/s / peak FLOP/s.
  MFU tells you how efficiently you are using the hardware.
    MFU < 30%  → fixable bottleneck (DataLoader, FP32, GPU idle)
    MFU 40-60% → typical well-optimized training
    MFU > 60%  → excellent (hard to achieve for full training loops)
""")

import time

M = 2048   # matrix size for benchmarking

if DEVICE == "cuda":
    A = torch.randn(M, M, device=DEVICE, dtype=torch.float16)
    B = torch.randn(M, M, device=DEVICE, dtype=torch.float16)

    # Warmup
    for _ in range(10): torch.mm(A, B)
    torch.cuda.synchronize()

    # Timed runs
    ITERS = 100
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS): torch.mm(A, B)
    e.record()
    torch.cuda.synchronize()
    avg_ms = s.elapsed_time(e) / ITERS
else:
    A = torch.randn(M, M, dtype=torch.float32)
    B = torch.randn(M, M, dtype=torch.float32)
    for _ in range(5): torch.mm(A, B)
    ITERS = 20
    t0 = time.perf_counter()
    for _ in range(ITERS): torch.mm(A, B)
    avg_ms = (time.perf_counter() - t0) / ITERS * 1000

# TODO 7: Compute observed FLOPs for one M×M matrix multiply
#   FLOPs = 2 * M * M * M  (each output element = M multiply-adds)
flops_per_mm = None  # YOUR CODE HERE

# TODO 8: Compute observed FLOP/s (FLOPs / time_in_seconds)
observed_flops_per_sec = None  # YOUR CODE HERE

# TODO 9: Compute MFU as a percentage
mfu_pct = None  # YOUR CODE HERE

assert flops_per_mm           is not None, "compute flops_per_mm"
assert observed_flops_per_sec is not None, "compute observed FLOP/s"
assert mfu_pct                is not None, "compute MFU"
assert 0 < mfu_pct <= 105,                f"MFU should be 0–100%, got {mfu_pct:.1f}%"

print(f"  {M}×{M} matmul ({'FP16' if DEVICE=='cuda' else 'FP32'})")
print(f"  Average time          : {avg_ms:.3f} ms")
print(f"  FLOPs per call        : {flops_per_mm/1e9:.2f} GFLOP")
print(f"  Observed throughput   : {observed_flops_per_sec/1e12:.2f} TFLOP/s")
print(f"  Peak throughput       : {peak_flops/1e12:.1f} TFLOP/s")
print(f"  MFU                   : {mfu_pct:.1f}%")
print(f"  Interpretation: {'Good utilization' if mfu_pct > 40 else 'Room for improvement'}")
print("  ✓ Section 4 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Roofline prediction vs actual performance
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Roofline Prediction vs Actual ──")
print("""
  The roofline predicts the MAXIMUM achievable performance given AI.
  If actual < roofline prediction: there is headroom to optimize.
  If actual ≈ roofline ceiling: you are hardware-limited.

  For a memory-bound op:  predicted_perf = mem_bw × AI
  For a compute-bound op: predicted_perf = peak_flops
""")

# Compute roofline-predicted performance for the M×M matmul
ai_matmul = linear_layer_ai(M, M, M, dtype_bytes=2 if DEVICE=="cuda" else 4)

# TODO 10: Compute the roofline-predicted FLOP/s for our M×M matmul
#   If ai_matmul > ridge_point: predicted = peak_flops
#   Else:                        predicted = mem_bw * ai_matmul
predicted_flops_per_sec = None  # YOUR CODE HERE

efficiency = (observed_flops_per_sec / predicted_flops_per_sec * 100
              if predicted_flops_per_sec else 0)

assert predicted_flops_per_sec is not None, "compute roofline prediction"
print(f"  Matmul AI             : {ai_matmul:.1f} FLOPs/byte")
print(f"  Ridge point           : {ridge_point:.1f} FLOPs/byte")
print(f"  Bound                 : {'compute-bound' if ai_matmul > ridge_point else 'memory-bound'}")
print(f"  Roofline prediction   : {predicted_flops_per_sec/1e12:.2f} TFLOP/s")
print(f"  Actual performance    : {observed_flops_per_sec/1e12:.2f} TFLOP/s")
print(f"  Efficiency vs ceiling : {efficiency:.1f}%")
print("  ✓ Section 5 passed")

print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Exercise 01 done!")
print()
print("  Key takeaways:")
print(f"    Ridge point on your hardware : {ridge_point:.1f} FLOPs/byte")
print(f"    Your {M}×{M} matmul MFU      : {mfu_pct:.1f}%")
print()
print("  LLM inference at bs=1 is ALWAYS memory-bound (AI ≈ 1 FLOPs/byte).")
print("  The fix: continuous batching, quantisation, speculative decoding.")
print("  All three techniques are covered in Part IV of this book.")
print("=" * 65)
