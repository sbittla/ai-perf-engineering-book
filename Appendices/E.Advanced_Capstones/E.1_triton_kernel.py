#!/usr/bin/env python3
"""
Appendices/E.Advanced_Capstones/E.1_triton_kernel.py  --  Appendix E.1
=======================================================================
Triton Kernel Optimization: why fusing a memory-bound op wins, and how to
predict the speedup with a roofline BEFORE you write the kernel.

Difficulty: ****-  (4/5 - analytic model; real kernel is the live extension)
Est. time:  6-10 hours for the full Triton implementation
Expected ranges:
  - Fused softmax+dropout moves ~2-3x fewer bytes than the unfused chain
  - Softmax arithmetic intensity is well below the GPU ridge point (memory-bound)
Troubleshooting:
  - No Triton or GPU needed here; this models the traffic you will then measure.
Live run (the real capstone):
  - Implement a fused softmax+dropout (or tiled GEMM) in Triton, then profile
    with: ncu --set roofline python your_kernel.py
  - Goal: land the kernel near the bandwidth roof and within X% of cuBLAS.

Run:  python Appendices/E.Advanced_Capstones/E.1_triton_kernel.py
"""
print("=" * 70)
print("  Exercise E.1 - Triton Kernel Optimization (roofline model)")
print("=" * 70)

# A row-softmax + dropout over an (M, N) activation tensor, FP16 (2 bytes).
M, N = 4096, 4096
DT = 2
elems = M * N
tensor_bytes = elems * DT

# ---------------------------------------------------------------------------
# SECTION 1: Memory traffic - unfused chain vs single fused kernel
# ---------------------------------------------------------------------------
print("\n-- Section 1: Memory Traffic --")
print("""
  Unfused, each op reads its input from HBM and writes its output back:
    softmax : read X, write S          (2 passes)
    dropout : read S, write Y          (2 passes)
  A fused kernel reads X once and writes Y once (1 pass each).
""")

# unfused: softmax reads+writes, dropout reads+writes  => 4 tensor-sized transfers
unfused_bytes = 4 * tensor_bytes
# fused: read X once, write Y once => 2 transfers
fused_bytes = 2 * tensor_bytes
reduction = unfused_bytes / fused_bytes

print(f"  Tensor: {M}x{N} FP16 = {tensor_bytes/1e6:.1f} MB")
print(f"  Unfused HBM traffic : {unfused_bytes/1e6:7.1f} MB")
print(f"  Fused   HBM traffic : {fused_bytes/1e6:7.1f} MB")
print(f"  Traffic reduction   : {reduction:.1f}x")
assert reduction >= 2.0
print("  [check] fusion cuts HBM traffic by >= 2x")

# ---------------------------------------------------------------------------
# SECTION 2: Roofline - is this kernel memory-bound?
# ---------------------------------------------------------------------------
print("\n-- Section 2: Roofline Position --")
print("""
  Arithmetic intensity (AI) = FLOPs / bytes moved. Softmax does a handful of
  FLOPs per element (max, sub, exp, sum, div ~ 5) against a 2-byte read+write.
  If AI << ridge point, the kernel is memory-bound and fusion is the lever.
""")

flops_per_elem = 5          # exp/sub/div/max/sum, order-of-magnitude
ai = (flops_per_elem * elems) / fused_bytes      # FLOPs per byte
PEAK_TFLOPS = 50.0          # book reference laptop-class FP16
PEAK_BW_GBs = 270.0         # GB/s
ridge = (PEAK_TFLOPS * 1e12) / (PEAK_BW_GBs * 1e9)

print(f"  Kernel AI       : {ai:6.2f} FLOPs/byte")
print(f"  GPU ridge point : {ridge:6.2f} FLOPs/byte")
verdict = "memory-bound" if ai < ridge else "compute-bound"
print(f"  Verdict         : {verdict}")
assert ai < ridge
print("  [check] softmax is memory-bound -> minimize bytes, i.e. fuse")

# ---------------------------------------------------------------------------
# SECTION 3: Predicted speedup and the time budget
# ---------------------------------------------------------------------------
print("\n-- Section 3: Predicted Speedup --")
print("""
  For a memory-bound kernel, runtime ~ bytes / bandwidth. So the fusion speedup
  is approximately the traffic reduction - the number to validate with ncu.
""")

def hbm_time_ms(bytes_, bw_gbs):
    return bytes_ / (bw_gbs * 1e9) * 1000

t_unfused = hbm_time_ms(unfused_bytes, PEAK_BW_GBs)
t_fused = hbm_time_ms(fused_bytes, PEAK_BW_GBs)
predicted = t_unfused / t_fused
print(f"  Unfused (HBM-bound) : {t_unfused:6.3f} ms")
print(f"  Fused   (HBM-bound) : {t_fused:6.3f} ms")
print(f"  Predicted speedup   : {predicted:.1f}x  (validate with ncu)")
assert abs(predicted - reduction) < 1e-6
print("  [check] predicted speedup == traffic reduction (memory-bound regime)")

print("\n" + "=" * 70)
print("  ALL SECTIONS PASSED - Exercise E.1 complete")
print("  You modeled the traffic, confirmed the kernel is memory-bound, and")
print("  predicted the fusion speedup. Now write it in Triton and verify with ncu.")
print("=" * 70)
