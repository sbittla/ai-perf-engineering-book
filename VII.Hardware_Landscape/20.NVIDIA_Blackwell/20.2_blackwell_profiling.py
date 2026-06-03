#!/usr/bin/env python3
"""
20.NVIDIA_Blackwell/20.2_blackwell_profiling.py  —  Chapter 20: NVIDIA Blackwell
=======================================================================
Covers book sections 20.5, 20.6:
  • Consumer Blackwell GPU comparison: RTX 4060 vs 4090 vs RTX 5090 (Table 20.4)
  • Computing the RTX 5090 roofline ridge point
  • Updated ncu metric names for FP8 and FP4 Tensor Core utilization
  • Three profiling adaptations required on Blackwell hardware

Run:
    python VII.Hardware_Landscape/20.NVIDIA_Blackwell/20.2_blackwell_profiling.py

No GPU required — all sections use reference data.
"""

import sys

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

print("=" * 70)
print("  Exercise 20.2 — Blackwell Profiling Guide")
print("=" * 70)

# ─────────────────────────────────────────────────────────────
# SECTION 1: Consumer GPU Comparison (Table 20.4)
# ─────────────────────────────────────────────────────────────
print("\n── Section 1: Consumer GPU Comparison ──")
print("""
  RTX 5090 is the consumer Blackwell flagship. Profiling commands
  (nsys, ncu, torch.profiler) work identically to RTX 4060.
  The key changes are the roofline constants and FP4 support.
""")

# (name, arch, cc, cuda_cores, tensor_gen, vram_gb, mem_type, bw_gbs, fp16_tflops, tdp_w, pcie)
CONSUMER_GPUS = [
    ("RTX 4060",  "Ada Lovelace", "8.9",  3_072, "4th",  8, "GDDR6",  272,  136,  115, "PCIe 4.0 x8"),
    ("RTX 4090",  "Ada Lovelace", "8.9", 16_384, "4th", 24, "GDDR6X",1_008, 660,  450, "PCIe 4.0 x16"),
    ("RTX 5090",  "Blackwell",   "10.0", 21_760, "5th", 32, "GDDR7", 1_792, 838,  575, "PCIe 5.0 x16"),
]

print(f"  {'Spec':<22}  {'RTX 4060':>12}  {'RTX 4090':>12}  {'RTX 5090':>12}")
print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*12}")
specs = [
    ("Architecture",    [g[1] for g in CONSUMER_GPUS]),
    ("Compute Cap.",    [g[2] for g in CONSUMER_GPUS]),
    ("CUDA Cores",      [f"{g[3]:,}" for g in CONSUMER_GPUS]),
    ("Tensor Core Gen", [g[4] for g in CONSUMER_GPUS]),
    ("VRAM (GB)",       [str(g[5]) for g in CONSUMER_GPUS]),
    ("Memory Type",     [g[6] for g in CONSUMER_GPUS]),
    ("BW (GB/s)",       [str(g[7]) for g in CONSUMER_GPUS]),
    ("FP16 (TFLOP/s)",  [str(g[8]) for g in CONSUMER_GPUS]),
    ("TDP (W)",         [str(g[9]) for g in CONSUMER_GPUS]),
    ("PCIe",            [g[10] for g in CONSUMER_GPUS]),
]
for label, vals in specs:
    print(f"  {label:<22}  {vals[0]:>12}  {vals[1]:>12}  {vals[2]:>12}")

# TODO 1: Compute the bandwidth speedup of RTX 5090 over RTX 4060.
rtx5090_bw = 1_792
rtx4060_bw = 272
bw_speedup = rtx5090_bw / rtx4060_bw

assert bw_speedup is not None, "compute bw_speedup"
assert 6.0 < bw_speedup < 7.0, f"BW speedup should be ~6.6×, got {bw_speedup:.1f}"
print(f"\n  RTX 5090 has {bw_speedup:.1f}× the memory bandwidth of RTX 4060.")
print("  Memory-bound kernels complete {:.1f}× faster (same AI per byte).".format(bw_speedup))
print("  ✓ Section 1 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 2: RTX 5090 Ridge Point
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: RTX 5090 Ridge Point vs RTX 4060 ──")
print("""
  The ridge point determines whether a kernel is compute- or memory-bound.
  For consumer GPUs in this book's exercises, we use FP16 Tensor Core
  peak TFLOP/s and the GDDR bandwidth ceiling.
""")

# TODO 2: Compute ridge points for RTX 4060 and RTX 5090.
#   ridge = fp16_tflops * 1000 / bw_gbs   (FLOP/byte)
rtx4060_ridge = 136 * 1000 / 272
rtx5090_ridge = 838 * 1000 / 1792

assert rtx4060_ridge is not None, "compute rtx4060_ridge"
assert rtx5090_ridge is not None, "compute rtx5090_ridge"

print(f"  RTX 4060 ridge point: {rtx4060_ridge:.0f} FLOP/byte")
print(f"  RTX 5090 ridge point: {rtx5090_ridge:.0f} FLOP/byte")
print(f"  Difference: {abs(rtx5090_ridge - rtx4060_ridge):.0f} FLOP/byte "
      f"({'higher' if rtx5090_ridge > rtx4060_ridge else 'lower'} on RTX 5090)")
print("""
  Both cards have similar ridge points (~468–500 FLOP/byte).
  A kernel memory-bound on RTX 4060 is also memory-bound on RTX 5090.
  The speed gain comes from 6.6× more bandwidth, not a roofline shift.
""")
print("  ✓ Section 2 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: ncu Metric Names for FP8 and FP4
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Updated ncu Metric Names for Blackwell ──")
print("""
  Three ncu metrics needed for Blackwell profiling.
  Use these to measure Tensor Core utilization for each precision.
""")

ncu_metrics = [
    (
        "FP16/BF16 (all generations)",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "Chapters 4–6 — unchanged on Blackwell",
    ),
    (
        "FP8 (Hopper/Blackwell)",
        "sm__inst_executed_pipe_tensor_op_imma_fp8.avg.pct_of_peak_sustained_active",
        "New in ncu 2023.x — requires H100 or B200",
    ),
    (
        "FP4 (Blackwell only)",
        "sm__inst_executed_pipe_tensor_op_imma_fp4.avg.pct_of_peak_sustained_active",
        "New in ncu 2024.x — requires B200",
    ),
]

for label, metric, note in ncu_metrics:
    print(f"\n  {label}")
    print(f"  Metric: {metric}")
    print(f"  Note  : {note}")

print("""
  Usage example (FP8 kernel on H100/B200):
    ncu --metrics sm__inst_executed_pipe_tensor_op_imma_fp8.avg.pct_of_peak_sustained_active \\
        --kernel-name '.*gemm.*' python infer_fp8.py

  FP4 also requires FP4 correctness validation:
    import torch
    max_diff = (out_fp4.float() - out_fp16.float()).abs().max()
    assert max_diff < 0.05, f'FP4 overflow: max_diff={max_diff:.4f}'
""")
print("  ✓ Section 3 passed — ncu metrics for FP8 and FP4 noted")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Three Profiling Adaptations for Blackwell
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Three Profiling Adaptations Required on Blackwell ──")
print("""
  The Chapter 5 workflow (nsys → ncu → torch.profiler) works unchanged
  on Blackwell. Three measurement updates are required to interpret
  results correctly.
""")

adaptations = [
    (
        "1. ncu Roofline — New L2 Cache Bandwidth Ceiling",
        [
            "Nsight Compute 2024.x adds an L2 bandwidth roof above the HBM roof.",
            "On Blackwell, L2 bandwidth is ~5× HBM bandwidth.",
            "Kernels below HBM ceiling but above L2 ceiling: optimize L2 access",
            "  patterns (shared memory tiling, register blocking), not HBM coalescing.",
        ],
    ),
    (
        "2. FP8 Tensor Core Metric (Section 3 above)",
        [
            "Use sm__inst_executed_pipe_tensor_op_imma_fp8 instead of the FP16 metric",
            "when profiling FP8-quantized models.",
            "Verify FP8 TC utilization > 80% to confirm Transformer Engine is active.",
        ],
    ),
    (
        "3. FP4 Overflow Detection",
        [
            "FP4 has an extremely limited dynamic range (E2M1 format: max ±6).",
            "Overflow to infinity without proper per-block scaling is a common bug.",
            "Run compute-sanitizer --tool racecheck on FP4 models.",
            "Always compare FP4 output against FP16 baseline (max_diff < 0.05).",
        ],
    ),
]

for title, points in adaptations:
    print(f"\n  {title}")
    for p in points:
        print(f"    {p}")

print("\n  ✓ Section 4 passed — three Blackwell profiling adaptations understood")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Detect Local GPU Generation
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Detect Your Local GPU Generation ──")

if has_torch and torch.cuda.is_available():
    props  = torch.cuda.get_device_properties(0)
    name   = props.name
    major  = props.major
    minor  = props.minor
    cc     = f"{major}.{minor}"

    ARCH_MAP = {
        (7, 0): ("Volta",        "V100"),
        (7, 5): ("Turing",       "RTX 20xx"),
        (8, 0): ("Ampere",       "A100"),
        (8, 6): ("Ampere",       "RTX 30xx"),
        (8, 9): ("Ada Lovelace", "RTX 40xx"),
        (9, 0): ("Hopper",       "H100"),
        (10, 0):("Blackwell",    "B200 / RTX 5090"),
    }
    arch, family = ARCH_MAP.get((major, minor), ("Unknown", "Unknown"))

    is_blackwell = major >= 10
    has_fp8_tc   = major >= 9
    has_fp4_tc   = major >= 10

    print(f"\n  GPU: {name}  (cc {cc})")
    print(f"  Architecture  : {arch}  ({family})")
    print(f"  FP8 Tensor Core support : {'✓' if has_fp8_tc else '✗'}")
    print(f"  FP4 Tensor Core support : {'✓' if has_fp4_tc else '✗'}")
    if is_blackwell:
        print("  → Use FP4/FP8 ncu metrics from Section 3 above.")
    elif has_fp8_tc:
        print("  → Use FP8 ncu metric from Section 3; FP4 metric not applicable.")
    else:
        print("  → Use FP16/BF16 ncu metric; FP8/FP4 not available on this GPU.")
else:
    print("""
  No CUDA GPU detected. Reference identification:
    Compute cap 7.x → Volta / Turing  (use FP16 TC metric)
    Compute cap 8.x → Ampere / Ada    (use FP16 TC metric)
    Compute cap 9.x → Hopper          (use FP8 TC metric)
    Compute cap 10.x→ Blackwell       (use FP4/FP8 TC metrics)
""")

print("  ✓ Section 5 passed")

print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE — Exercise 20.2 done!")
print()
print("  Key takeaways:")
print("    RTX 5090 vs RTX 4060: ~6.6× more bandwidth, similar ridge point.")
print("    FP4 adds ~2× memory efficiency vs FP8 with < 1% perplexity loss.")
print("    Three profiling adaptations: L2 ceiling, FP8 metric, FP4 overflow.")
print()
print("  Next: VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.1_roofline_multi_gpu.py")
print("=" * 70)
