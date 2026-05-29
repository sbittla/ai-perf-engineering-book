#!/usr/bin/env python3
"""
Appendices/A.Environment_Setup/A.1_environment_check.py  —  Appendix A: Environment Setup

Verify every component required by this book:
  Python version, PyTorch, CUDA, profiling tools, and GPU capabilities.

Run:  python Appendices/A.Environment_Setup/A.1_environment_check.py
"""

import sys
import os
import importlib
import subprocess
import platform
import time

print("=" * 68)
print("  APPENDIX A — Environment Setup Verification")
print("=" * 68)

CUDA_AVAILABLE = False
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    pass

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"
print(f"\n  Platform : {platform.system()} {platform.machine()}")
print(f"  Device   : {DEVICE}")
print()

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Python and PyTorch
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 1 — Python and PyTorch")
print("─" * 68)

print("""
CONCEPT: Version requirements for this book
  • Python 3.10+  — match statements, improved type hints
  • PyTorch 2.0+  — torch.compile requires 2.x
  • CUDA 12.x     — needed for Tensor Core features used in Part II
""")

# TODO: Check Python version — fails if < 3.10
py_major, py_minor = sys.version_info[:2]
print(f"  Python    : {py_major}.{py_minor} ({sys.executable})")
assert (py_major, py_minor) >= (3, 10), (
    f"Python 3.10+ required, found {py_major}.{py_minor}. "
    "Install: sudo apt install python3.10"
)

# TODO: Check PyTorch version — fails if < 2.0
import torch
torch_ver = torch.__version__
print(f"  PyTorch   : {torch_ver}")
torch_major = int(torch_ver.split(".")[0])
assert torch_major >= 2, (
    f"PyTorch 2.0+ required, found {torch_ver}. "
    "Install: pip install torch --index-url https://download.pytorch.org/whl/cu121"
)

# TODO: Report CUDA version visible to PyTorch
cuda_ver = torch.version.cuda if torch.version.cuda else "N/A (CPU build)"
print(f"  CUDA      : {cuda_ver}")
print(f"  GPU count : {torch.cuda.device_count()}")

print("\n✓ Section 1 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — GPU Capabilities
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 2 — GPU Capabilities")
print("─" * 68)

print("""
CONCEPT: GPU spec determines which optimisations apply
  • Compute capability 7.0+ → Tensor Cores (FP16)
  • Compute capability 8.0+ → BF16 Tensor Cores, TF32
  • Compute capability 8.9+ → FP8 (Ada Lovelace, e.g. RTX 4060)
""")

if CUDA_AVAILABLE:
    props = torch.cuda.get_device_properties(0)
    cc_major = props.major
    cc_min   = props.minor
    vram_gb  = props.total_memory / 1e9

    print(f"  GPU name       : {props.name}")
    print(f"  VRAM           : {vram_gb:.1f} GB")
    print(f"  Compute cap    : {cc_major}.{cc_min}")
    print(f"  SM count       : {props.multi_processor_count}")
    print(f"  Tensor Cores   : {'Yes (FP16)' if cc_major >= 7 else 'No'}")
    print(f"  BF16 support   : {torch.cuda.is_bf16_supported()}")
    print(f"  torch.compile  : {'Yes (requires CUDA 12+)' if cc_major >= 7 else 'Limited'}")

    # LLM memory feasibility for this GPU
    print(f"\n  LLM memory feasibility on {vram_gb:.0f}GB VRAM:")
    models = [
        ("GPT-2 (124M, FP16)",      0.25),
        ("Llama-7B (FP16)",          14.0),
        ("Llama-7B (INT8)",           7.0),
        ("Llama-7B (INT4/GPTQ)",      3.5),
        ("Llama-13B (FP16)",         26.0),
        ("Llama-13B (INT4/GPTQ)",     6.5),
    ]
    for name, need_gb in models:
        fits = "✓" if need_gb < vram_gb * 0.9 else "✗"
        print(f"    {fits} {name:<28} needs {need_gb:.1f} GB")
else:
    print("  [CPU mode] GPU capability checks skipped.")
    print("  To enable: install PyTorch with CUDA support.")
    print("  https://pytorch.org/get-started/locally/")

print("\n✓ Section 2 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Required Python Packages
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 3 — Required Python Packages")
print("─" * 68)

print("""
CONCEPT: Book exercise dependencies
  Core packages are mandatory. Optional packages unlock specific features.
""")

def check_package(import_name, pip_name=None, version_attr="__version__"):
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, version_attr, "installed")
        return "✓", str(ver)
    except ImportError:
        return "✗", f"missing — pip install {pip_name or import_name}"

REQUIRED = [
    ("torch",          "torch"),
    ("numpy",          "numpy"),
    ("matplotlib",     "matplotlib"),
    ("tqdm",           "tqdm"),
]
OPTIONAL = [
    ("transformers",   "transformers",  "— HuggingFace models (Ch 10–13)"),
    ("nvitop",         "nvitop",        "— live GPU dashboard"),
    ("py_spy",         "py-spy",        "— CPU flamegraphs (Appendix B)"),
    ("docx",           "python-docx",   "— document generator scripts"),
]

print("  Required packages:")
all_required_ok = True
for imp, pip in REQUIRED:
    status, info = check_package(imp, pip)
    print(f"    {status} {imp:<20} {info}")
    if status == "✗":
        all_required_ok = False

print("\n  Optional packages:")
for imp, pip, note in OPTIONAL:
    status, info = check_package(imp, pip)
    print(f"    {status} {imp:<20} {info}  {note}")

assert all_required_ok, (
    "Some required packages are missing. "
    "Run: pip install -r requirements.txt"
)

print("\n✓ Section 3 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — Profiling Tools
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 4 — Profiling Tools")
print("─" * 68)

print("""
CONCEPT: Profiling tool chain (Parts II and III)
  nsys  → GPU timeline (system level, ms granularity)
  ncu   → kernel hardware counters (nanosecond granularity)
  py-spy → Python CPU flamegraphs (no sudo needed)
  perf  → Linux hardware events (IPC, cache miss rate)
""")

# TODO: Check for external tools — report found/missing without failing
def tool_version(cmd):
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3
        )
        first = (result.stdout or result.stderr).strip().split("\n")[0]
        return "✓", first[:60] if first else "found"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "✗", "not found"

TOOLS = [
    (["nsys", "--version"],          "nsys (Nsight Systems)"),
    (["ncu",  "--version"],          "ncu  (Nsight Compute)"),
    (["perf", "--version"],          "perf"),
    (["py-spy", "--version"],        "py-spy"),
    (["nvitop", "--version"],        "nvitop"),
    (["flamegraph.pl", "--help"],    "flamegraph.pl"),
]

for cmd, label in TOOLS:
    status, info = tool_version(cmd)
    print(f"  {status} {label:<28} {info}")

print("""
  Install missing tools:
    nsys / ncu  : ships with CUDA Toolkit, or apt install nsight-systems
    py-spy      : pip install py-spy
    perf        : sudo apt install linux-tools-generic
    flamegraph  : git clone https://github.com/brendangregg/FlameGraph ~/FlameGraph
""")

print("✓ Section 4 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — Timing Baseline
# ──────────────────────────────────────────────────────────────────────────────
print("─" * 68)
print("Section 5 — Timing Baseline")
print("─" * 68)

print("""
CONCEPT: Establishing a timing reference
  We verify that CUDA events work correctly (or fall back to wall-clock time).
  All exercise files use this pattern for GPU timing.
""")

import torch

# TODO: Run a tiny matmul and time it with CUDA events (or perf_counter on CPU)
SIZE = 1024
a = torch.randn(SIZE, SIZE, device=DEVICE)
b = torch.randn(SIZE, SIZE, device=DEVICE)

def time_matmul(a, b, n=20):
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        for _ in range(n):
            _ = torch.mm(a, b)
        end_evt.record()
        torch.cuda.synchronize()
        return start_evt.elapsed_time(end_evt) / n
    else:
        t0 = time.perf_counter()
        for _ in range(n):
            _ = torch.mm(a, b)
        return (time.perf_counter() - t0) / n * 1000

matmul_ms = time_matmul(a, b)
dtype_bytes = 4  # float32
flops = 2 * SIZE ** 3
bw_gbs = flops / (matmul_ms * 1e-3) / 1e9
print(f"  {SIZE}×{SIZE} matmul : {matmul_ms:.2f} ms  ({bw_gbs:.0f} GFLOPS on {DEVICE})")

if CUDA_AVAILABLE:
    vram_used = torch.cuda.memory_allocated() / 1e6
    print(f"  VRAM used   : {vram_used:.1f} MB")

assert matmul_ms > 0, "Timing returned zero — CUDA events not working"
print("\n✓ Section 5 passed\n")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 68)
print("  ALL SECTIONS PASSED")
print(f"  Environment ready for all book exercises (device={DEVICE})")
print()
print("  Next: Appendices/B.Command_Reference/B.1_profiling_cheatsheet.py")
print("=" * 68)
