#!/usr/bin/env python3
"""
21.AMD_MI300X_ROCm/21.2_rocm_profiling.py  —  Chapter 21: AMD MI300X
=======================================================================
Covers book sections 21.2, 21.3:
  • ROCm 6.x profiling tool mapping vs NVIDIA nsys/ncu (Table 21.2)
  • rocprofv3 command reference for kernel counter collection
  • rocm-smi monitoring commands vs nvidia-smi equivalents
  • PyTorch on ROCm: device detection, BF16, FP8, FSDP/RCCL
  • Known ROCm/CUDA behavioral differences affecting this book's exercises

Run:
    python VII.Hardware_Landscape/21.AMD_MI300X_ROCm/21.2_rocm_profiling.py

No GPU required — all sections are reference material.
"""

print("=" * 70)
print("  Exercise 21.2 — ROCm 6.x Profiling Stack and PyTorch Migration")
print("=" * 70)

try:
    import torch
    has_torch = True
    is_rocm   = has_torch and getattr(torch.version, "hip", None) is not None
except ImportError:
    has_torch = False
    is_rocm   = False


# ─────────────────────────────────────────────────────────────
# SECTION 1: NVIDIA-to-ROCm Profiling Tool Mapping (Table 21.2)
# ─────────────────────────────────────────────────────────────
print("\n── Section 1: Profiling Tool Mapping — NVIDIA → ROCm ──")
print("""
  ROCm 6.x brought the tool chain significantly closer to CUDA parity.
  Most Chapter 5 profiling workflows translate directly.
""")

tools = [
    # (nvidia_tool, purpose, rocm_equiv, parity)
    ("nsys profile",    "System timeline, API trace",        "rocprof --sys-trace",          "Good"),
    ("ncu",             "Per-kernel hardware counters",       "rocprofv3 --pmc",              "Partial"),
    ("torch.profiler",  "PyTorch op attribution",            "torch.profiler (same API)",    "Full"),
    ("nvidia-smi",      "Live GPU utilisation",              "rocm-smi",                     "Full"),
    ("nvitop",          "Rich TUI monitor",                  "amdgpu_top / rocm-smi watch",  "Partial"),
    ("py-spy",          "CPU flamegraph (Python)",           "py-spy (identical)",           "Full"),
    ("perf / eBPF",     "Linux systems profiling",           "perf / eBPF (identical)",      "Full"),
    ("compute-sanitizer","Memory/race error detection",      "rocm-debug-agent / sanitizer", "Partial"),
]

print(f"  {'NVIDIA Tool':<20}  {'Purpose':<36}  {'ROCm Equivalent':<32}  Parity")
print(f"  {'─'*20}  {'─'*36}  {'─'*32}  {'─'*8}")
for nv, purpose, rocm, parity in tools:
    print(f"  {nv:<20}  {purpose:<36}  {rocm:<32}  {parity}")

print("""
  Where parity is 'Partial': rocprofv3 has fewer built-in metric presets
  than ncu; you must specify explicit counter names (see Section 2).
""")
print("  ✓ Section 1 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 2: rocprofv3 Command Reference
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: rocprofv3 — Kernel Counter Collection ──")
print("""
  rocprofv3 is the ROCm equivalent of ncu. Unlike ncu's named metric
  sets (--set basic, --set full), rocprofv3 uses explicit counter names.
""")

rocprofv3_examples = [
    (
        "Basic kernel stats (≈ ncu --set basic)",
        "rocprofv3 --stats python train.py",
    ),
    (
        "HBM memory throughput (≈ ncu dram__throughput)",
        "rocprofv3 --pmc TCC_EA_RDREQ_32B,TCC_EA_WRREQ_32B python infer.py",
    ),
    (
        "Flat memory wavefronts (bandwidth utilisation)",
        "rocprofv3 --pmc TA_FLAT_READ_WAVEFRONTS,TA_FLAT_WRITE_WAVEFRONTS python infer.py",
    ),
    (
        "Export to CSV for analysis",
        "rocprofv3 --stats --output-format csv -o mi300x_profile.csv python infer.py",
    ),
]

for desc, cmd in rocprofv3_examples:
    print(f"\n  # {desc}")
    print(f"  {cmd}")

print("""
  rocm-smi monitoring equivalents (≈ nvidia-smi):
    rocm-smi                           # live utilisation
    watch -n 0.5 rocm-smi             # continuous (≈ watch nvidia-smi)
    rocm-smi --showuse --showmemuse    # GPU + memory utilisation
    rocm-smi --showpower               # power consumption
    rocm-smi --showtemp                # temperature
""")
print("  ✓ Section 2 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 3: PyTorch on ROCm — Installation and Verification
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: PyTorch on ROCm — Installation and Verification ──")
print("""
  PyTorch uses 'cuda' as the device string on ROCm (HIP maps CUDA → AMD).
  torch.version.hip is set on ROCm; torch.version.cuda is None.
""")

print("  Installation:")
print("    pip install torch torchvision torchaudio \\")
print("        --index-url https://download.pytorch.org/whl/rocm6.0")
print()
print("  Verification:")
print("""    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())       # True on ROCm
    print(torch.cuda.get_device_name(0))   # 'AMD Instinct MI300X'
    print(torch.version.hip)               # '6.0.0' on ROCm
    print(torch.version.cuda)              # None on ROCm
""")

if has_torch:
    print(f"  Your environment:")
    print(f"    torch version   : {torch.__version__}")
    print(f"    CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    Device name     : {torch.cuda.get_device_name(0)}")
    hip_ver  = getattr(torch.version, "hip",  None)
    cuda_ver = getattr(torch.version, "cuda", None)
    print(f"    torch.version.hip  : {hip_ver}")
    print(f"    torch.version.cuda : {cuda_ver}")
    print(f"    Running on ROCm : {'YES ✓' if is_rocm else 'NO  (CUDA or CPU)'}")
else:
    print("  (torch not installed — install to see live environment details)")
print("  ✓ Section 3 passed")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Known ROCm/CUDA Differences (affects book exercises)
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Known ROCm Differences Affecting This Book's Exercises ──")
print("""
  Most code from Chapters 1–19 runs unchanged on ROCm.
  Four known differences require attention:
""")

differences = [
    (
        "torch.compile (Chapter 6.3)",
        [
            "Uses Triton-HIP kernels on ROCm — same API, different backend.",
            "max-autotune mode works but first compilation may take longer.",
            "Not all fusion patterns supported on every ROCm version.",
            "Verify compile speedup on your specific ROCm version.",
        ],
    ),
    (
        "FlashAttention (Chapters 11–13)",
        [
            "FlashAttention-2 available for ROCm via flash-attn with ROCm wheels.",
            "FlashAttention-3 (Hopper-specific) has NO ROCm equivalent.",
            "Use composable_kernel ROCm-optimised attention as alternative.",
        ],
    ),
    (
        "FP8 Quantisation (Chapter 6.2)",
        [
            "MI300X supports FP8 E4M3 and E5M2 natively.",
            "PyTorch autocast does NOT auto-select FP8 on ROCm.",
            "Explicit dtype casting required: tensor.to(torch.float8_e4m3fn).",
        ],
    ),
    (
        "Multi-GPU / FSDP (Chapter 13.3)",
        [
            "ROCm uses RCCL (ROCm Collective Communications Library).",
            "RCCL is a drop-in replacement — same Python API as NCCL.",
            "DDP and FSDP work identically on ROCm via RCCL.",
            "AllReduce timing model from Chapter 13 applies with MI300X BW.",
        ],
    ),
]

for title, points in differences:
    print(f"  {title}")
    for p in points:
        print(f"    • {p}")
    print()

print("  ✓ Section 4 passed — ROCm differences noted")


print("\n" + "=" * 70)
print("  ALL SECTIONS COMPLETE — Exercise 21.2 done!")
print()
print("  Key takeaways:")
print("    ROCm profiling stack: rocprofv3 ≈ ncu, rocm-smi ≈ nvidia-smi.")
print("    torch.cuda.is_available() returns True on ROCm — same device API.")
print("    Four exercises to revisit on ROCm: compile, FlashAttention,")
print("    FP8 autocast, and NCCL→RCCL multi-GPU.")
print()
print("  Next: VII.Hardware_Landscape/22.Custom_Silicon/22.1_hardware_landscape.py")
print("=" * 70)
