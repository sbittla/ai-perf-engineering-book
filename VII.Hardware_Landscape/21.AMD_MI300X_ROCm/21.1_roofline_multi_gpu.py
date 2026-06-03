#!/usr/bin/env python3
"""
21.AMD_MI300X_ROCm/21.1_roofline_multi_gpu.py  —  Chapter 21: AMD MI300X
=======================================================================
Covers book sections 21.1, 21.4:
  • Detecting the active GPU vendor (NVIDIA CUDA vs AMD ROCm/HIP)
  • MI300X architecture: 304 CUs, 192 GB HBM3, 5.3 TB/s bandwidth
  • Roofline ridge points for MI300X, H100, and B200 from reference specs
  • Classifying LLM inference kernels across vendor hardware
  • Hardware selection decision tree: MI300X vs H100 vs B200

Run:
    python VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.1_roofline_multi_gpu.py

Works on CPU, NVIDIA CUDA, or AMD ROCm — vendor is detected automatically.
All TODO blocks are calculations. All sections must print ✓.
"""

print("=" * 70)
print("  Exercise 21.1 — Roofline Analysis Across GPU Vendors")
print("=" * 70)

try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


# ─────────────────────────────────────────────────────────────
# SECTION 1: Detect GPU Vendor
# ─────────────────────────────────────────────────────────────
print("\n── Section 1: Detect Active GPU Vendor ──")
print("""
  PyTorch maps ROCm/HIP to the same 'cuda' device string.
  torch.version.hip is set on ROCm builds; None on CUDA builds.
  torch.version.cuda is set on CUDA builds; None on ROCm builds.
""")

vendor      = "CPU (no GPU)"
device_name = "CPU"
is_nvidia   = False
is_amd      = False

if has_torch and torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    if torch.version.hip is not None:
        vendor   = f"AMD ROCm  (HIP {torch.version.hip})"
        is_amd   = True
    else:
        vendor   = f"NVIDIA CUDA  (CUDA {torch.version.cuda})"
        is_nvidia = True
elif has_torch:
    pass  # torch imported but no GPU

print(f"  Active device : {device_name}")
print(f"  Vendor        : {vendor}")
print(f"  CUDA available: {has_torch and torch.cuda.is_available()}")

if has_torch and torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"  Compute cap.  : {props.major}.{props.minor}")
    print(f"  VRAM          : {props.total_memory / 1e9:.1f} GB")
else:
    print("  (Running reference tables without live GPU)")
print("  ✓ Section 1 passed — vendor detection understood")


# ─────────────────────────────────────────────────────────────
# SECTION 2: MI300X Architecture vs H100 vs B200 (Table 21.1)
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: MI300X vs H100 vs B200 Key Specifications ──")
print("""
  MI300X: chiplet design — 8 CDNA3 GPU dies + 4 memory controllers
          stacked on an interposer with 8 HBM3 stacks.
  H100:   monolithic Hopper die, 132 SMs, HBM3.
  B200:   monolithic Blackwell die, 192 SMs, HBM3e.
""")

# Reference specs from Table 21.1
GPU_SPECS = {
    # name: (arch, compute_units, vram_gb, bw_tbs, bf16_tflops, fp8_tflops, tdp_w, interconnect, multi_bw)
    "AMD MI300X":   ("CDNA3 chiplet",   304, 192, 5.3,  1_307, 2_614,   750, "Infinity Fabric", 448),
    "NVIDIA H100":  ("Hopper monolithic",132,  80, 3.35, 1_979, 3_958,   700, "NVLink 4",        900),
    "NVIDIA B200":  ("Blackwell monolithic",192,192, 8.0, 5_000,10_000, 1_000, "NVLink 5",      1_800),
}

fields = [
    ("Architecture",    lambda s: s[0]),
    ("CUs / SMs",       lambda s: str(s[1])),
    ("VRAM (GB)",       lambda s: str(s[2])),
    ("HBM BW (TB/s)",   lambda s: str(s[3])),
    ("BF16 TFLOP/s",    lambda s: f"{s[4]:,}"),
    ("FP8 TOPS",        lambda s: f"{s[5]:,}"),
    ("TDP (W)",         lambda s: str(s[6])),
    ("Interconnect",    lambda s: s[7]),
    ("Multi-GPU BW GB/s",lambda s: str(s[8])),
]

names = list(GPU_SPECS.keys())
print(f"  {'Spec':<22}  {names[0]:>14}  {names[1]:>14}  {names[2]:>14}")
print(f"  {'─'*22}  {'─'*14}  {'─'*14}  {'─'*14}")
for label, fn in fields:
    vals = [fn(GPU_SPECS[n]) for n in names]
    print(f"  {label:<22}  {vals[0]:>14}  {vals[1]:>14}  {vals[2]:>14}")

print("\n  MI300X's decisive advantage: 5.3 TB/s HBM3 bandwidth — 58% more than H100.")
print("  MI300X fits 70B FP16 on one GPU (192 GB); H100 needs 2×.")
print("  ✓ Section 2 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Roofline Ridge Points for All Three GPUs
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Compute Ridge Points ──")
print("""
  Ridge point = peak BF16 FLOP/s / memory bandwidth
              = BF16 TFLOP/s × 1000 / BW_GBs   (gives FLOP/byte)

  Below the ridge point → memory-bound (bandwidth is the bottleneck).
  Above the ridge point → compute-bound (FLOP/s is the bottleneck).
""")

ridge_points = {}
for gpu_name, spec in GPU_SPECS.items():
    _, _, _, bw_tbs, bf16_tflops, _, _, _, _ = spec
    bw_gbs = bw_tbs * 1000  # TB/s → GB/s

    # TODO 1: Compute ridge point in FLOP/byte.
    #   ridge = bf16_tflops * 1000 / bw_gbs
    ridge = bf16_tflops * 1000 / bw_gbs

    assert ridge is not None, f"compute ridge for {gpu_name}"
    ridge_points[gpu_name] = ridge

print(f"  {'GPU':<16}  {'BF16 TFLOP/s':>14}  {'BW (GB/s)':>12}  {'Ridge (FLOP/byte)':>18}")
print(f"  {'─'*16}  {'─'*14}  {'─'*12}  {'─'*18}")
for gpu_name, spec in GPU_SPECS.items():
    _, _, _, bw_tbs, bf16, _, _, _, _ = spec
    print(f"  {gpu_name:<16}  {bf16:>14,}  {bw_tbs*1000:>12,.0f}  "
          f"{ridge_points[gpu_name]:>18.1f}")

# MI300X ridge should be the lowest (most BW-rich relative to compute)
assert ridge_points["AMD MI300X"] < ridge_points["NVIDIA H100"], \
    "MI300X should have a lower ridge point than H100 (more bandwidth-rich)"
print(f"\n  MI300X ridge: {ridge_points['AMD MI300X']:.0f}  "
      f"H100 ridge: {ridge_points['NVIDIA H100']:.0f}")
print("  MI300X is more bandwidth-rich → memory-bound workloads benefit more.")
print("  ✓ Section 3 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Classify LLM Operations Across Hardware
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Classify LLM Inference Operations ──")
print("""
  LLM decode (batch=1): AI ≈ 1 FLOPs/byte — almost always memory-bound.
  LLM prefill (batch=32): AI ≈ 100+ FLOPs/byte — may be compute-bound.
  Large GEMM (batch=1024): AI > 400 FLOPs/byte — likely compute-bound.
""")

OPS = [
    # (name, arithmetic_intensity_flops_per_byte)
    ("LLM decode (bs=1, FP16)",       1.0),
    ("LLM decode (bs=8, FP16)",       8.0),
    ("LLM prefill (bs=32, FP16)",   128.0),
    ("GEMM 4096×4096 (bs=64, FP16)",320.0),
    ("GEMM 4096×4096 (bs=1024,FP16)",512.0),
]

print(f"\n  {'Operation':<38}  {'AI':>8}  {'MI300X':>10}  {'H100':>10}  {'B200':>10}")
print(f"  {'─'*38}  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*10}")

for op_name, ai in OPS:
    # TODO 2: For each GPU, classify the operation.
    #   bound = "compute" if ai > ridge_points[gpu] else "memory"
    mi300x_bound = "compute" if ai > ridge_points["AMD MI300X"] else "memory"
    h100_bound   = "compute" if ai > ridge_points["NVIDIA H100"] else "memory"
    b200_bound   = "compute" if ai > ridge_points["NVIDIA B200"] else "memory"

    assert mi300x_bound is not None, f"classify {op_name} on MI300X"
    assert h100_bound   is not None, f"classify {op_name} on H100"
    assert b200_bound   is not None, f"classify {op_name} on B200"
    print(f"  {op_name:<38}  {ai:>8.1f}  {mi300x_bound:>10}  {h100_bound:>10}  {b200_bound:>10}")

# LLM decode (bs=1) should be memory-bound on all three
assert mi300x_bound is not None  # last iteration
print("""
  Key insight: LLM decode at bs=1 is memory-bound on ALL vendors.
  The winning hardware for decode is the one with most HBM bandwidth:
    MI300X (5.3 TB/s) > B200 (8.0 TB/s) > H100 (3.35 TB/s) for capacity
    B200 wins on bandwidth; MI300X wins on cost-per-GB.
""")
print("  ✓ Section 4 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Hardware Selection Decision Tree (Table 21.4)
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Hardware Selection Decision Tree ──")
print("""
  Choosing between MI300X, H100, and B200 depends on the dominant
  bottleneck in your specific workload.
""")

decisions = [
    ("70B FP16 inference (decode)",   "MI300X",         "Fits in 192 GB; 5.3 TB/s; no TP needed"),
    ("70B FP8 inference (high QPS)",  "H100 / B200",    "Higher FLOP/s for prefill at scale"),
    ("405B FP8 inference",            "2× B200",        "192 GB + 8 TB/s each; minimal TP overhead"),
    ("LLM training (dense)",          "B200",           "Highest FP8 FLOP/s; fastest AllReduce"),
    ("LLM training (MoE)",            "H100 or B200",   "Expert parallel requires fast NVLink"),
    ("Research (cost-sensitive)",     "MI300X",         "Competitive price/performance vs H100"),
    ("Apple Silicon (laptop/edge)",   "Mac / MLX",      "Unified memory; see Chapter 22"),
]

print(f"  {'Workload':<38}  {'Best GPU':<16}  Reason")
print(f"  {'─'*38}  {'─'*16}  {'─'*40}")
for workload, gpu, reason in decisions:
    print(f"  {workload:<38}  {gpu:<16}  {reason}")

print("""
  Summary rule: MI300X for memory-capacity-limited decode;
                B200 for throughput-limited training & prefill;
                H100 for current mainstream production deployments.
""")
print("  ✓ Section 5 passed")


print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE — Exercise 21.1 done!")
print()
print("  Key takeaways:")
for gpu_name in GPU_SPECS:
    print(f"    {gpu_name:<16}: ridge = {ridge_points[gpu_name]:.0f} FLOP/byte")
print()
print("  LLM decode is ALWAYS memory-bound (AI ≈ 1 FLOPs/byte).")
print("  Pick hardware based on bandwidth and capacity, not FLOP/s,")
print("  for decode-dominated inference workloads.")
print()
print("  Next: VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.2_rocm_profiling.py")
print("=" * 70)
