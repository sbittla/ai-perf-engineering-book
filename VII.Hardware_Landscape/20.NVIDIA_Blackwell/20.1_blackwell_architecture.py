#!/usr/bin/env python3
"""
20.NVIDIA_Blackwell/20.1_blackwell_architecture.py  —  Chapter 20: NVIDIA Blackwell
=======================================================================
Covers book sections 20.1, 20.2, 20.3, 20.4:
  • 5th-generation Tensor Core throughput vs Hopper (Table 20.1)
  • Updated roofline ridge points across GPU generations (Table 20.2)
  • HBM3e memory feasibility for LLM models on B200
  • NVLink 5 AllReduce timing vs H100 and A100 clusters
  • Grace Blackwell (GB200) NVLink-C2C vs PCIe host-device transfer

Run:
    python VII.Hardware_Landscape/20.NVIDIA_Blackwell/20.1_blackwell_architecture.py

No GPU required — all sections use reference data tables.
TODOs are arithmetic calculations. All sections must print ✓.
"""

import math

print("=" * 70)
print("  Exercise 20.1 — NVIDIA Blackwell Architecture")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# SECTION 1: 5th-Generation Tensor Core Throughput (Table 20.1)
# ─────────────────────────────────────────────────────────────
print("\n── Section 1: Tensor Core Throughput — B200 vs H100 ──")
print("""
  Blackwell's 5th-generation Tensor Cores add FP4 and approximately
  double throughput at every precision level relative to Hopper (H100).

  FP4 format: 3-bit mantissa, 0-bit exponent, per-block scale factor.
  Weights stored in FP4, dequantized on-chip before accumulation.
  Accuracy: < 1% perplexity degradation vs FP16 for most LLMs.
""")

# Reference specs from Table 20.1
TENSOR_CORE_SPECS = [
    # (precision, b200_peak_str, h100_peak_str, b200_tflops, h100_tflops, use_case)
    ("FP4",     "~20 PFLOP/s",  "N/A",         20_000,  None,  "Weight-only LLM inference"),
    ("FP8",     "~10 PFLOP/s",  "~3.9 PFLOP/s", 10_000, 3_900,  "LLM training & inference"),
    ("FP16/BF16","~5 PFLOP/s",  "~1.98 PFLOP/s", 5_000, 1_979,  "Mixed-precision training"),
    ("TF32",    "~2.5 PFLOP/s", "~989 TFLOP/s",  2_500,   989,  "Default FP32 matmul"),
    ("INT8",    "~20 TOPS",     "~7.9 TOPS",     20_000, 7_900,  "Post-training quantization"),
]

print(f"  {'Precision':<12}  {'B200 Peak':>12}  {'H100 Peak':>12}  {'Ratio':>6}  Use Case")
print(f"  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*6}  {'─'*32}")
for prec, b200_str, h100_str, b200, h100, use in TENSOR_CORE_SPECS:
    ratio = f"{b200/h100:.1f}×" if h100 else "NEW"
    print(f"  {prec:<12}  {b200_str:>12}  {h100_str:>12}  {ratio:>6}  {use}")

# TODO 1: B200 FP16/BF16 is ~5,000 TFLOP/s, H100 is ~1,979 TFLOP/s.
#   Compute the ratio B200/H100 for FP16/BF16 Tensor Core throughput.
fp16_b200 = 5_000
fp16_h100 = 1_979
fp16_ratio = None  # YOUR CODE HERE → fp16_b200 / fp16_h100

assert fp16_ratio is not None, "compute fp16_ratio"
assert 2.0 < fp16_ratio < 3.0, f"FP16 ratio should be ~2.5x, got {fp16_ratio:.2f}"
print(f"\n  B200 FP16/BF16 Tensor Core speedup over H100: {fp16_ratio:.2f}×")
print("  ✓ Section 1 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Updated Roofline Ridge Points (Table 20.2)
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Roofline Ridge Points Across Generations ──")
print("""
  The ridge point = peak FLOP/s / memory bandwidth.
  It is the arithmetic intensity (FLOPs/byte) above which a kernel is
  compute-bound; below which it is memory-bandwidth-bound.

  Blackwell's proportional improvement in both FLOP/s and bandwidth
  keeps the ridge point nearly constant vs Hopper — the SAME kernels
  that are memory-bound on H100 remain memory-bound on B200.
  The benefit for memory-bound workloads is raw bandwidth (8 vs 3.35 TB/s).
""")

# Reference data: (gpu_name, fp16_tflops, hbm_bw_gbs)
GPU_ROOFLINE_DATA = [
    ("A100 SXM4",   312,   2_000),
    ("H100 SXM5",  1_979,  3_350),
    ("B200 SXM",   5_000,  8_000),
    ("RTX 5090",     838,  1_792),
    ("RTX 4060",     136,    272),
    ("MI300X",     1_307,  5_300),   # AMD — for comparison in Section 3
]

print(f"  {'GPU':<16}  {'FP16 TFLOP/s':>14}  {'HBM BW GB/s':>12}  {'Ridge (FLOP/byte)':>18}")
print(f"  {'─'*16}  {'─'*14}  {'─'*12}  {'─'*18}")

ridge_points = {}
for name, fp16, bw in GPU_ROOFLINE_DATA:
    # TODO 2: Compute ridge_point = fp16_tflops * 1e12 / (bw * 1e9)
    #   = fp16_tflops * 1000 / bw   (simplifies to TFLOP/s * 1000 / GB/s = FLOP/byte)
    ridge = None  # YOUR CODE HERE → fp16 * 1000 / bw

    assert ridge is not None, f"compute ridge point for {name}"
    ridge_points[name] = ridge
    print(f"  {name:<16}  {fp16:>14,}  {bw:>12,}  {ridge:>18.1f}")

# Verify B200 ridge point is close to H100 (within 10%)
b200_ridge = ridge_points["B200 SXM"]
h100_ridge = ridge_points["H100 SXM5"]
assert abs(b200_ridge - h100_ridge) / h100_ridge < 0.10, (
    f"B200 ridge ({b200_ridge:.1f}) should be within 10% of H100 ({h100_ridge:.1f})"
)
print(f"\n  H100 ridge: {h100_ridge:.1f}  B200 ridge: {b200_ridge:.1f}  "
      f"Difference: {abs(b200_ridge - h100_ridge)/h100_ridge*100:.1f}%")
print("  → Same workloads remain memory-bound on B200 as on H100.")
print("  ✓ Section 2 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: HBM3e Memory Feasibility for LLM Models
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Model Fit on B200 (192 GB HBM3e) ──")
print("""
  Memory required for LLM inference ≈ params × bytes_per_param.
  B200 has 192 GB HBM3e; H100 has 80 GB HBM3.

  FP16: 2 bytes/param   FP8: 1 byte/param   FP4: 0.5 bytes/param
""")

B200_VRAM_GB = 192
H100_VRAM_GB = 80

MODELS = [
    ("Llama 3  8B",  8e9),
    ("Llama 3 70B",  70e9),
    ("Llama 3 405B", 405e9),
]

PRECISIONS = [
    ("FP16", 2),
    ("FP8",  1),
    ("FP4",  0.5),
]

print(f"  {'Model':<16}  {'Precision':<8}  {'GB Required':>12}  {'Fits B200?':>12}  {'H100s needed':>14}")
print(f"  {'─'*16}  {'─'*8}  {'─'*12}  {'─'*12}  {'─'*14}")
for model_name, params in MODELS:
    for prec_name, bytes_per_param in PRECISIONS:
        # TODO 3: Compute model size in GB.
        #   size_gb = params * bytes_per_param / 1e9
        size_gb = None  # YOUR CODE HERE

        assert size_gb is not None, f"compute size_gb for {model_name} {prec_name}"
        fits_b200 = size_gb <= B200_VRAM_GB
        h100s = math.ceil(size_gb / H100_VRAM_GB)
        print(f"  {model_name:<16}  {prec_name:<8}  {size_gb:>12.0f}  "
              f"{'Yes ✓' if fits_b200 else 'No  ✗':>12}  {h100s:>14}")

# Verify Llama 3 70B FP16 does NOT fit on one H100 but DOES fit on one B200
llama70_fp16_gb = 70e9 * 2 / 1e9
assert llama70_fp16_gb <= B200_VRAM_GB, "Llama 70B FP16 should fit on B200"
assert llama70_fp16_gb >  H100_VRAM_GB, "Llama 70B FP16 should not fit on single H100"
print(f"\n  Llama 3 70B FP16: {llama70_fp16_gb:.0f} GB — fits B200 ✓, needs 2× H100 ✓")
print("  ✓ Section 3 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: NVLink 5 AllReduce Timing
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: AllReduce Timing — A100 vs H100 vs B200 Cluster ──")
print("""
  Ring AllReduce formula (Chapter 13):
    t = 2 × (N-1)/N × tensor_bytes / link_bw_bytes_per_sec

  NVLink generations:
    NVLink 3 (A100): 600 GB/s per GPU pair
    NVLink 4 (H100): 900 GB/s per GPU pair
    NVLink 5 (B200): 1,800 GB/s per GPU pair
""")

NVLINK = {
    "A100 (NVLink 3)":  600e9,
    "H100 (NVLink 4)":  900e9,
    "B200 (NVLink 5)": 1_800e9,
}

tensor_gb   = 1.0
tensor_bytes = tensor_gb * 1e9
N_GPUS       = 8

print(f"  AllReduce of {tensor_gb:.0f} GB tensor, {N_GPUS} GPUs:\n")
print(f"  {'Cluster':<22}  {'NVLink BW':>12}  {'AllReduce (ms)':>16}")
print(f"  {'─'*22}  {'─'*12}  {'─'*16}")

ar_times = {}
for cluster, bw in NVLINK.items():
    # TODO 4: Compute ring AllReduce time in milliseconds.
    #   t_s = 2 * (N_GPUS - 1) / N_GPUS * tensor_bytes / bw
    #   t_ms = t_s * 1000
    t_ms = None  # YOUR CODE HERE

    assert t_ms is not None, f"compute AllReduce time for {cluster}"
    ar_times[cluster] = t_ms
    print(f"  {cluster:<22}  {bw/1e9:>10.0f} GB/s  {t_ms:>14.2f} ms")

# Verify B200 is fastest, roughly 2x faster than H100
assert ar_times["B200 (NVLink 5)"] < ar_times["H100 (NVLink 4)"], "B200 should be faster"
ratio_h100_b200 = ar_times["H100 (NVLink 4)"] / ar_times["B200 (NVLink 5)"]
print(f"\n  B200 AllReduce is {ratio_h100_b200:.1f}× faster than H100 for {N_GPUS}-GPU ring.")
print("  ✓ Section 4 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Grace Blackwell — NVLink-C2C vs PCIe (Table 20.3)
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: GB200 Grace Blackwell — Eliminating the PCIe Bottleneck ──")
print("""
  The GB200 Superchip connects a Grace CPU to a Blackwell GPU via
  NVLink-C2C at 900 GB/s — 28× faster than PCIe 4.0 x16 (~32 GB/s).

  DataLoader implications (Chapter 7 revisited):
    PCIe systems  → pin_memory + non_blocking are CRITICAL.
    GB200 systems → H2D is no longer the bottleneck; profiling target
                    shifts to CPU preprocessing (py-spy, not idle_pct).
""")

transfers = [
    ("CPU DRAM → GPU VRAM",   32e9,    900e9),
    ("GPU VRAM → CPU DRAM",   32e9,    900e9),
    ("Effective DataLoader BW", 12e9,  400e9),
]

print(f"  {'Transfer Type':<28}  {'PCIe 4.0':>12}  {'NVLink-C2C':>12}  {'Speedup':>8}")
print(f"  {'─'*28}  {'─'*12}  {'─'*12}  {'─'*8}")
for label, pcie_bw, c2c_bw in transfers:
    speedup = c2c_bw / pcie_bw
    print(f"  {label:<28}  {pcie_bw/1e9:>10.0f} GB/s  {c2c_bw/1e9:>10.0f} GB/s  {speedup:>6.0f}×")

print("""
  Key architectural shift:
    PCIe (most GPUs): DataLoader bottleneck → pin_memory, prefetch
    NVLink-C2C (GB200): CPU preprocessing bottleneck → py-spy, codec
""")
print("  ✓ Section 5 passed — Grace Blackwell eliminates PCIe H2D bottleneck")


print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE — Exercise 20.1 done!")
print()
print("  Key takeaways:")
print(f"    B200 FP16 Tensor Core: {fp16_b200:,} TFLOP/s  ({fp16_ratio:.1f}× over H100)")
print(f"    B200 ridge point: {ridge_points['B200 SXM']:.0f} FLOP/byte  (≈ same as H100: {h100_ridge:.0f})")
print(f"    Llama 70B FP16: {llama70_fp16_gb:.0f} GB — single B200 ✓, needs 2× H100")
print(f"    B200 AllReduce (1 GB, 8 GPUs): {ar_times['B200 (NVLink 5)']:.2f} ms via NVLink 5")
print()
print("  Next: 20.2_blackwell_profiling.py")
print("=" * 70)
