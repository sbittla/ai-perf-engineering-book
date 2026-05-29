#!/usr/bin/env python3
"""
22.Custom_Silicon/22.1_hardware_landscape.py  —  Chapter 22: Custom Silicon
=======================================================================
Covers book sections 22.1 – 22.5:
  22.1  Intel Gaudi 3 — Production AI Accelerator
  22.2  AWS Trainium 2 and Inferentia 3
  22.3  Apple Silicon — Unified Memory for Inference
  22.4  Intel AMX — CPU Matrix Acceleration (detects if running on AMX)
  22.5  CXL Memory — Pooling and Disaggregation

Run:
    python VII.Hardware_Landscape/22.Custom_Silicon/22.1_hardware_landscape.py

No GPU required. Section 4 detects Intel AMX on x86 CPUs if available.
"""

import platform
import struct
import sys

print("=" * 70)
print("  Exercise 22.1 — Custom Silicon and CXL Memory")
print("=" * 70)


# ─────────────────────────────────────────────────────────────
# SECTION 1: Intel Gaudi 3 — Production AI Accelerator
# ─────────────────────────────────────────────────────────────
print("\n── Section 1: Intel Gaudi 3 ──")
print("""
  Intel Gaudi 3 (2024) is Intel's production-grade AI accelerator,
  designed as a lower-cost alternative to NVIDIA A100/H100 for LLM
  training and inference at scale. Key differentiators:

  • 128 Tensor Processor Cores (TPC) for mixed-precision GEMM
  • 96 GB HBM2e memory at 3.7 TB/s bandwidth (3 stacks per die)
  • 24× 200 GbE RDMA ports (24 TB/s bisection for cluster communication)
    — Uses Ethernet for inter-node, not InfiniBand; simpler ops/networking
  • Native BF16 and FP8 support; no FP4 as of Gaudi 3
  • Programming: Intel Habana SynapseAI SDK; PyTorch via habana_frameworks
""")

GAUDI3_SPECS = {
    "TPC cores":        128,
    "VRAM (GB)":        96,
    "HBM BW (TB/s)":   3.7,
    "Network":         "24× 200 GbE RDMA  (24 TB/s bisection)",
    "Peak BF16 TFLOP/s":1_835,
    "Peak FP8 TOPS":    3_670,
    "TDP (W)":          600,
    "Programming":     "SynapseAI / habana_frameworks PyTorch",
}

print(f"  {'Specification':<22}  Value")
print(f"  {'─'*22}  {'─'*40}")
for k, v in GAUDI3_SPECS.items():
    print(f"  {k:<22}  {v}")

# TODO 1: Compute Gaudi 3 roofline ridge point.
#   ridge = bf16_tflops * 1000 / bw_gbs
#   (BF16 peak = 1,835 TFLOP/s; HBM BW = 3.7 TB/s = 3,700 GB/s)
gaudi3_ridge = None  # YOUR CODE HERE

assert gaudi3_ridge is not None, "compute gaudi3_ridge"
print(f"\n  Gaudi 3 ridge point: {gaudi3_ridge:.0f} FLOP/byte")
print("  Compare: H100 ≈ 591, MI300X ≈ 247, Gaudi 3 ≈", round(gaudi3_ridge))
print("""
  PyTorch on Gaudi 3:
    import habana_frameworks.torch.core as htcore
    device = 'hpu'                     # Gaudi device string (not 'cuda')
    model  = model.to(device)
    with torch.autocast('hpu', dtype=torch.bfloat16):
        output = model(input.to(device))
    htcore.mark_step()                 # explicit graph step (lazy execution)
""")
print("  ✓ Section 1 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 2: AWS Trainium 2 and Inferentia 3
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: AWS Trainium 2 and Inferentia 3 ──")
print("""
  AWS custom silicon targets the cost-sensitive cloud workload.
  Two chips: Trainium 2 for training, Inferentia 3 for inference.
""")

AWS_CHIPS = {
    "AWS Trainium 2": {
        "Purpose":     "LLM training at scale (trn2 instances)",
        "Peak BF16":   "~85 TFLOP/s per chip (×16 chips per trn2.48xl)",
        "Memory":      "HBM2e, 96 GB per chip",
        "Interconnect":"NeuronLink 2 — 768 GB/s peer BW",
        "SDK":         "AWS Neuron SDK (torch_neuronx)",
        "Key benefit": "~3× lower training cost vs equivalent H100 on AWS",
    },
    "AWS Inferentia 3": {
        "Purpose":     "High-throughput LLM inference (inf3 instances)",
        "Peak INT8":   "~190 TOPS per chip",
        "Memory":      "HBM + on-chip SRAM (compiler-managed)",
        "Interconnect":"NeuronLink — 384 GB/s",
        "SDK":         "AWS Neuron SDK (optimum-neuron)",
        "Key benefit": "Lowest inference cost-per-token on AWS for production",
    },
}

for chip_name, specs in AWS_CHIPS.items():
    print(f"\n  {chip_name}:")
    for k, v in specs.items():
        print(f"    {k:<18}  {v}")

print("""
  PyTorch compilation for Neuron (ahead-of-time, not JIT):
    import torch_neuronx
    traced = torch.jit.trace(model, example_input)
    neuron_model = torch_neuronx.trace(traced, example_input)
    neuron_model.save('model.pt')

  Neuron models must be compiled per batch size and sequence length.
  Dynamic shapes are not supported — set fixed input shapes at compile time.
""")
print("  ✓ Section 2 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Apple Silicon — Unified Memory for Inference
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Apple Silicon — Unified Memory Inference ──")
print("""
  Apple M-series chips use unified memory: CPU and GPU share the same
  physical DRAM. No PCIe H2D copy exists. Memory bandwidth is
  shared but very high (~400 GB/s on M3 Max; ~546 GB/s on M4 Max).

  Key advantage for LLM inference:
  • M3 Max: 128 GB unified memory — fits Llama 3 70B in FP16 (140 GB: too close)
  • M3 Max: 64 GB — fits Llama 3 70B FP8 (70 GB at limit)
  • M4 Max: 128 GB — fits Llama 3 70B FP8 (70 GB with 58 GB for KV)
""")

APPLE_CHIPS = [
    # (chip, year, cpu_cores, gpu_cores, mem_gb_max, bw_gbs, ane_tops)
    ("M2 Ultra",  2023, 24,  60,  192,  800,  31.6),
    ("M3 Max",    2023, 16,  40,  128,  400,  18.0),
    ("M4 Max",    2024, 16,  40,  128,  546,  38.0),
    ("M4 Ultra",  2025, 32,  80,  512, 1_092, 76.0),
]

print(f"  {'Chip':<12}  {'Year':>6}  {'CPU':>5}  {'GPU':>5}  {'Max RAM GB':>10}  "
      f"{'BW GB/s':>10}  {'ANE TOPS':>10}")
print(f"  {'─'*12}  {'─'*6}  {'─'*5}  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*10}")
for chip, yr, cpu, gpu, mem, bw, ane in APPLE_CHIPS:
    print(f"  {chip:<12}  {yr:>6}  {cpu:>5}  {gpu:>5}  {mem:>10}  {bw:>10}  {ane:>10}")

# TODO 2: Llama 3 70B FP16 requires ~140 GB. Does it fit in M4 Max 128 GB config?
#   A float: fits_llama70_fp16_m4max = (140 <= 128)
fits_llama70_fp16_m4max = None  # YOUR CODE HERE → 140 <= 128

assert fits_llama70_fp16_m4max is not None, "compute fits_llama70_fp16_m4max"
assert fits_llama70_fp16_m4max is False, "Llama 70B FP16 (140 GB) exceeds M4 Max 128 GB"
print(f"\n  Llama 3 70B FP16 (140 GB) fits M4 Max 128 GB: {fits_llama70_fp16_m4max}  "
      f"(need FP8 ≈70 GB or M4 Ultra 512 GB)")

# Detect if running on Apple Silicon
is_apple_silicon = (
    platform.system() == "Darwin" and
    platform.machine() in ("arm64", "aarch64")
)
print(f"\n  Running on Apple Silicon: {is_apple_silicon}")
if is_apple_silicon:
    print("  MLX (Apple ML framework) available. Use mps device for PyTorch:")
    print("    device = torch.device('mps') if torch.backends.mps.is_available() else 'cpu'")
    try:
        import torch
        mps_ok = torch.backends.mps.is_available()
        print(f"  torch.backends.mps.is_available() = {mps_ok}")
    except Exception:
        pass
else:
    print("  MLX usage: pip install mlx; device = 'mps' in PyTorch")
print("  ✓ Section 3 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Intel AMX — CPU Matrix Acceleration
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Intel AMX — CPU Matrix Acceleration ──")
print("""
  Intel AMX (Advanced Matrix Extensions) adds 2D tile registers to Xeon
  processors (Sapphire Rapids, Emerald Rapids, Granite Rapids) for
  INT8 and BF16 matrix multiply at ≈10 TFLOP/s BF16 per socket.

  AMX use cases:
  • CPU-side LLM inference when GPU is unavailable or too expensive
  • INT8 quantized inference: AMX-INT8 ≈ 30× faster than scalar INT8
  • Small-batch inference where GPU idle overhead dominates

  PyTorch uses AMX automatically via oneDNN backend (no code changes
  needed). Verify AMX is active: ONEDNN_VERBOSE=1 python infer.py
  should show 'brgemm:amx' in the kernel dispatch log.
""")

# Detect AMX via CPUID (Linux /proc/cpuinfo or Windows)
amx_detected = False
amx_subtypes = []

if platform.system() == "Linux":
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        if "amx_bf16" in cpuinfo:
            amx_subtypes.append("AMX-BF16")
        if "amx_int8" in cpuinfo:
            amx_subtypes.append("AMX-INT8")
        if "amx_tile" in cpuinfo:
            amx_subtypes.append("AMX-TILE")
        amx_detected = len(amx_subtypes) > 0
    except Exception:
        pass
elif platform.system() == "Windows":
    # On Windows, check via Python's platform or try WMI — approximate
    cpu_info = platform.processor()
    # Sapphire Rapids (4th Gen Xeon) and later support AMX
    amx_detected = False  # Cannot reliably detect on Windows without WMI/ctypes
    cpu_info_str = f"  CPU: {cpu_info} (AMX detection limited on Windows)"
    print(cpu_info_str)

print(f"  AMX detected   : {amx_detected}")
if amx_subtypes:
    print(f"  AMX subtypes   : {', '.join(amx_subtypes)}")
elif amx_detected is False:
    print("  (AMX not detected — requires Intel Sapphire Rapids Xeon or newer)")

print("""
  AMX performance reference (Xeon Sapphire Rapids, single socket):
    AMX-BF16 : ~14 TFLOP/s  (vs 1.2 TFLOP/s AVX-512 FP32)
    AMX-INT8  : ~27 TOPS     (vs ~2.5 TOPS AVX-512 INT8)
    Typical LLM decode (7B INT8): ~15 tokens/s on a 96-core Sapphire Rapids

  When to use CPU inference instead of GPU:
    • Batch size = 1, latency-sensitive, GPU startup overhead > serving time
    • Cost-optimised deployment using existing on-prem Xeon fleet
    • Models small enough to fit in L3 cache (< ~96 MB for hot weights)
""")
print("  ✓ Section 4 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 5: CXL Memory — Pooling and Disaggregation
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: CXL Memory — Pooling and Disaggregation ──")
print("""
  CXL (Compute Express Link) is a PCIe 5.0-based cache-coherent
  interconnect that allows hosts to address external memory attached
  via CXL as if it were local DRAM.

  AI infrastructure implications:
  • CXL memory expanders (e.g. Samsung CMS, Micron CXL) add terabytes
    of DRAM to a server without adding CPU sockets.
  • For LLM serving: KV cache can overflow to CXL memory at ~50–100 GB/s
    instead of to NVMe SSD at ~7 GB/s — 10–15× faster KV swap.
  • For model storage: 405B model weights in FP8 (202 GB) can be held
    in CXL memory pool and streamed to GPU on demand.

  CXL memory is cache-coherent — CPU can read/write it with load/store
  instructions. GPU cannot access CXL memory directly; DMA transfers
  via CPU still required.
""")

CXL_SPECS = {
    "Interface":        "PCIe 5.0 (CXL 2.0/3.0)",
    "Max bandwidth":    "~128 GB/s per CXL link (PCIe 5.0 ×16)",
    "Latency vs DRAM":  "~200–400 ns (vs ~80 ns DDR5)",
    "Capacity":         "Up to 1–2 TB per expander",
    "CPU access":       "cache-coherent load/store (NUMA node)",
    "GPU access":       "via CPU DMA only (not direct)",
    "Primary use":      "KV cache overflow, model weight pre-staging",
}

print(f"  {'Property':<24}  Value")
print(f"  {'─'*24}  {'─'*50}")
for k, v in CXL_SPECS.items():
    print(f"  {k:<24}  {v}")

# TODO 3: KV cache bandwidth comparison.
#   CXL BW ≈ 100 GB/s; NVMe SSD BW ≈ 7 GB/s.
#   How many times faster is CXL than NVMe for KV swapping?
cxl_bw_gbs  = 100.0
nvme_bw_gbs = 7.0
cxl_vs_nvme = None  # YOUR CODE HERE → cxl_bw_gbs / nvme_bw_gbs

assert cxl_vs_nvme is not None, "compute cxl_vs_nvme"
print(f"\n  CXL for KV swap: {cxl_vs_nvme:.0f}× faster than NVMe SSD "
      f"({cxl_bw_gbs:.0f} GB/s vs {nvme_bw_gbs:.0f} GB/s)")
print("""
  CXL in practice today (2025):
  • Supported on Intel Sapphire Rapids, AMD EPYC Genoa servers.
  • Linux 6.x exposes CXL memory as NUMA nodes.
  • NUMA binding: numactl --membind=<cxl_node> python serve.py
  • vLLM and SGLang: experimental CXL-backed KV cache offload.
""")
print("  ✓ Section 5 passed")


print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE — Exercise 22.1 done!")
print()
print("  Key takeaways:")
print(f"    Intel Gaudi 3 ridge point : {gaudi3_ridge:.0f} FLOP/byte"
      if gaudi3_ridge else "    Intel Gaudi 3: compute ridge point (see TODO 1)")
print(f"    CXL vs NVMe KV swap       : {cxl_vs_nvme:.0f}× faster"
      if cxl_vs_nvme else "    CXL vs NVMe (see TODO 3)")
print("    AMX-BF16 on Xeon          : ~14 TFLOP/s (useful for CPU inference)")
print("    Apple M4 Max              : 128 GB unified memory, 546 GB/s BW")
print()
print("  This completes Part VII — The 2024–2026 Hardware Landscape.")
print("  Return to Appendix A for environment setup if starting a new platform.")
print("=" * 70)
