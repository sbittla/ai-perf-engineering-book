#!/usr/bin/env python3
"""
V.Workload_Benchmarking/15.Porting_a_Workload/15.1_porting_checklist.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 15: Porting a Workload — Section 1: The Porting Checklist
=======================================================================
Covers book section 15.1:
  • The 6-step porting checklist: profile → port → validate → benchmark
  • NumPy → PyTorch tensor equivalences and common dtype traps
  • Float64 promotion: how Python scalars silently promote to float64
  • Host-to-device (H2D) and device-to-host (D2H) transfer costs
  • Pinned memory: why it speeds up H2D transfers
  • The .contiguous() requirement before calling CUDA kernels

Run:  python V.Workload_Benchmarking/15.Porting_a_Workload/15.1_porting_checklist.py
All sections must print ✓.
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np

print("=" * 60)
print("  Exercise 15.1 — The Porting Checklist")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The 6-step porting checklist
# ─────────────────────────────────────────────────────────────
print("── Section 1: The 6-Step Porting Checklist ──")
print("""
  Porting a workload from CPU to GPU (or from research to production)
  is a common source of silent correctness bugs and unexpected slowdowns.

  THE 6-STEP CHECKLIST:
    □ 1. Profile first  — identify the hotspot (CPU or GPU bottleneck?)
    □ 2. Validate dtype — NumPy defaults float64; PyTorch GPU prefers float32
    □ 3. Validate shape — row-major (C contiguous) vs col-major (Fortran)
    □ 4. Measure H2D    — host-to-device transfer is often the hidden bottleneck
    □ 5. Port hotspot   — only port the bottleneck; keep bookkeeping on CPU
    □ 6. Validate output — assert max absolute difference < 1e-4 (dtype dependent)

  COMMON MISTAKES IN ORDER OF FREQUENCY:
    1. float64 on GPU — 8× slower than float32 on most consumer GPUs
    2. Too many H2D copies — one .to("cuda") per iteration kills throughput
    3. Non-contiguous tensors — .T (transpose) is a view, not contiguous
    4. Python scalars → float64 promotion inside operations
    5. Forgetting model.eval() → different BatchNorm/Dropout behaviour

  RULE: port one op at a time. Validate correctness after each step.
  Never port and optimise simultaneously — you can't tell which change
  caused a correctness regression.
""")
print("  ✓ Section 1 passed — commit the 6-step checklist to memory")


# ─────────────────────────────────────────────────────────────
# SECTION 2: NumPy ↔ PyTorch equivalences and dtype traps
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: NumPy → PyTorch Dtype Traps ──")
print("""
  NumPy defaults to float64. PyTorch defaults to float32 on GPU.
  Mixing them causes silent precision changes AND performance cliffs.

  EQUIVALENCES:
    NumPy                        PyTorch
    ──────────────────────       ──────────────────────────
    np.zeros((n,), dtype=float)  torch.zeros(n)               ← float32 on GPU
    np.array([1.0, 2.0])         torch.tensor([1.0, 2.0])     ← float32 by default
    a @ b                        a @ b  or  torch.mm(a, b)
    np.concatenate([a, b], 0)    torch.cat([a, b], dim=0)
    np.transpose(a)              a.T  (view, not contiguous!)
    np.sum(a, axis=0)            a.sum(dim=0)
    np.maximum(a, 0)             torch.relu(a)  or  a.clamp(min=0)

  THE FLOAT64 TRAP:
    a = np.array([1.0, 2.0])        # float64!
    t = torch.from_numpy(a)          # inherits float64
    t = t.to(DEVICE)                 # float64 on GPU → 8× slower
    t = t.to(torch.float32)          # ← correct fix

  TODO 1: Implement safe_from_numpy() that converts a NumPy array
  to a float32 CUDA tensor regardless of the input dtype.
""")


def safe_from_numpy(arr: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    """
    TODO 1: Convert arr to float32 tensor on `device`.
    Steps: torch.from_numpy(arr) → .float() → .to(device)
    """
    return torch.from_numpy(arr).float().to(device)


# Verify: float64 numpy → float32 CUDA
arr_f64 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
arr_f32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)

t64 = safe_from_numpy(arr_f64)
t32 = safe_from_numpy(arr_f32)

assert t64.dtype == torch.float32, f"Expected float32, got {t64.dtype}"
assert t32.dtype == torch.float32, f"Expected float32, got {t32.dtype}"
assert str(t64.device).startswith(DEVICE.split(":")[0]), f"Expected {DEVICE}, got {t64.device}"

print(f"  float64 numpy → {t64.dtype} on {t64.device}  ✓")
print(f"  float32 numpy → {t32.dtype} on {t32.device}  ✓")

# Demonstrate the trap: Python scalar promotes to float64
scalar = 1.0  # Python float → float64 in NumPy context
trap_tensor = torch.zeros(4) + scalar   # float32 + float64 → float32 (PyTorch handles this)
print(f"\n  Python scalar (1.0) + float32 tensor → {trap_tensor.dtype}  (PyTorch normalises)")
# But in NumPy:
trap_numpy = np.zeros(4) + scalar       # float64!
print(f"  Python scalar (1.0) + np.zeros → {trap_numpy.dtype}  ← NumPy promotes to float64!")
print("  ✓ Section 2 passed — always call .float() after from_numpy()")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Contiguity — .T is a view, not a copy
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Contiguity Traps ──")
print("""
  PyTorch operations that require contiguous memory (e.g. view(), many
  CUDA kernels) will raise RuntimeError if given a non-contiguous tensor.

  COMMON NON-CONTIGUOUS TENSORS:
    t.T                    — transpose: memory layout is unchanged, strides reversed
    t[::2]                 — strided slice
    t.permute(2, 0, 1)     — permute: non-contiguous unless explicitly copied

  DIAGNOSIS:
    t.is_contiguous()      — True if row-major (C order)
    t.contiguous()         — makes a contiguous copy if needed (no-op if already OK)
    t.stride()             — shows memory stride for each dimension

  THE SAFE PATTERN (port checklist step 3):
    t = some_tensor.contiguous()   # always safe before calling CUDA kernel
    t = t.reshape(...)             # reshape requires contiguous
""")

# Demonstrate contiguity
original = torch.randn(4, 6)
transposed = original.T             # non-contiguous view

print(f"  original   is_contiguous: {original.is_contiguous()}")
print(f"  .T         is_contiguous: {transposed.is_contiguous()}")
print(f"  .T.contiguous() is_contiguous: {transposed.contiguous().is_contiguous()}")

# reshape fails on non-contiguous; .view() also fails
try:
    transposed.view(-1)
    print("  view() succeeded (unexpected)")
except RuntimeError:
    print("  view() on .T raised RuntimeError  ← expected behaviour")

# .contiguous().view() always works
flat = transposed.contiguous().view(-1)
assert flat.is_contiguous(), "Should be contiguous after .contiguous()"
assert flat.numel() == 24, "Should have 24 elements"

assert not transposed.is_contiguous(), "Transpose should not be contiguous"
assert transposed.contiguous().is_contiguous(), ".contiguous() should fix it"
print("  ✓ Section 3 passed — always call .contiguous() before reshape/CUDA kernels")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Measuring H2D and D2H transfer costs
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Host-to-Device Transfer Cost ──")
print("""
  The PCIe bus between CPU (host) and GPU (device) is the slowest link
  in the GPU memory hierarchy:
    L1 cache    : ~10,000 GB/s
    HBM (A100)  :  2,000 GB/s
    PCIe 4.0 ×16:     32 GB/s   ← 60× slower than HBM!
    PCIe 3.0 ×16:     16 GB/s

  If your model does input.to("cuda") on every inference call, and
  your input is 100 MB, that is 100/32 ≈ 3 ms added per call.

  PINNED MEMORY (page-locked):
    Normally, CPU memory can be paged out to disk. The DMA engine
    requires the memory to stay in place during transfer.
    Non-pinned: OS must first copy to a pinned staging buffer → 2× the time
    Pinned:     DMA reads directly from the original buffer → ~2× faster H2D

  torch.Tensor.pin_memory() pins an existing CPU tensor.
  DataLoader(pin_memory=True) pins all batch tensors automatically.

  TODO 2: Implement measure_h2d_bandwidth_gbs() that creates a large
  CPU tensor, times the .to(DEVICE) transfer, and returns GB/s.
  Test both pinned and non-pinned versions.
""")


def measure_h2d_bandwidth_gbs(n_elements: int = 32 * 1024 * 1024,
                               pinned: bool = False,
                               warmup: int = 3,
                               iters: int = 10) -> float:
    """
    TODO 2: Measure H2D bandwidth in GB/s.
    Create a float32 CPU tensor. If pinned=True, call .pin_memory().
    Time n_elements .to(DEVICE) transfers.
    Bandwidth = n_elements * 4 bytes / mean_time_s / 1e9
    On CPU-only: just time a clone() to simulate.
    """
    cpu_t = torch.rand(n_elements, dtype=torch.float32)
    if pinned and DEVICE == "cuda":
        cpu_t = cpu_t.pin_memory()

    def transfer():
        if DEVICE == "cuda":
            t = cpu_t.to(DEVICE, non_blocking=False)
            torch.cuda.synchronize()
        else:
            t = cpu_t.clone()

    # Warmup
    for _ in range(warmup):
        transfer()

    t0 = time.perf_counter()
    for _ in range(iters):
        transfer()
    elapsed_s = (time.perf_counter() - t0) / iters

    bytes_transferred = n_elements * 4
    return bytes_transferred / elapsed_s / 1e9


bw_normal = measure_h2d_bandwidth_gbs(pinned=False)
bw_pinned = measure_h2d_bandwidth_gbs(pinned=True)

print(f"  Tensor size: {32 * 1024 * 1024 * 4 / 1e6:.0f} MB")
print(f"  H2D bandwidth (normal) : {bw_normal:.1f} GB/s")
print(f"  H2D bandwidth (pinned) : {bw_pinned:.1f} GB/s")

if DEVICE == "cuda" and bw_pinned > 0 and bw_normal > 0:
    ratio = bw_pinned / bw_normal
    print(f"  Pinned speedup: {ratio:.2f}×  (expected 1.5–2.5× on real GPU)")
else:
    print(f"  (CPU-only: pinned memory has no effect; both measure clone() speed)")

assert bw_normal > 0, "Bandwidth should be positive"
assert bw_pinned > 0, "Bandwidth should be positive"
print("  ✓ Section 4 passed — H2D bandwidth measured; use pin_memory for speed")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Output validation — correctness before speed
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Validating Porting Correctness ──")
print("""
  The porting checklist ends with validation: the GPU result must match
  the CPU reference within numerical tolerance.

  TOLERANCES BY DTYPE:
    float32   : atol=1e-5,  rtol=1e-4   (PyTorch default)
    float16   : atol=1e-2,  rtol=1e-3   (reduced precision)
    bfloat16  : atol=1e-1,  rtol=1e-2   (even lower mantissa bits)
    int8 quant: up to ±1 LSB            (quantisation error)

  COMMON SOURCES OF MISMATCH:
    • Floating-point order of operations (matmul reduction order differs)
    • CUDA fast-math approximations (cos, exp, rsqrt)
    • float64 CPU reference vs float32 GPU result

  VALIDATION PROTOCOL:
    1. Run the operation on CPU in float32
    2. Run the same operation on GPU in float32
    3. max_abs_diff = (cpu_result - gpu_result.cpu()).abs().max()
    4. Assert max_abs_diff < tolerance

  TODO 3: Implement validate_port(cpu_fn, gpu_fn, input_cpu, atol=1e-4)
  that runs both functions, moves the GPU result to CPU, and returns
  (max_abs_diff, passed) where passed = max_abs_diff <= atol.
""")


def validate_port(cpu_fn, gpu_fn, input_cpu: torch.Tensor,
                  atol: float = 1e-4) -> tuple:
    """
    TODO 3: Validate that gpu_fn matches cpu_fn on the given input.
    cpu_result  = cpu_fn(input_cpu)
    gpu_input   = input_cpu.to(DEVICE)
    gpu_result  = gpu_fn(gpu_input).cpu()
    max_abs_diff = (cpu_result - gpu_result).abs().max().item()
    Return (max_abs_diff, max_abs_diff <= atol).
    """
    cpu_result = cpu_fn(input_cpu)
    gpu_input = input_cpu.to(DEVICE)
    gpu_result = gpu_fn(gpu_input).cpu()
    max_abs_diff = (cpu_result - gpu_result).abs().max().item()
    return max_abs_diff, max_abs_diff <= atol


# Build a reference model (CPU) and a deep-copy on GPU (same weights, different device)
torch.manual_seed(42)
ref_model = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 128)
)
ref_model.eval()
gpu_model = copy.deepcopy(ref_model).to(DEVICE)
gpu_model.eval()

test_input = torch.randn(16, 128)  # CPU input

with torch.no_grad():
    diff, passed = validate_port(
        cpu_fn=lambda x: ref_model(x),
        gpu_fn=lambda x: gpu_model(x),
        input_cpu=test_input,
        atol=1e-4,
    )

print(f"  CPU vs GPU max abs diff: {diff:.2e}")
print(f"  Validation passed      : {passed}  (atol=1e-4)")

if not passed:
    print(f"  WARNING: diff {diff:.2e} exceeds tolerance — check dtype and op order")

assert diff < 1e-2, f"Max diff {diff:.2e} is too large — possible dtype error"

print(f"""
  WHEN VALIDATION FAILS:
    1. Check dtypes: cpu_tensor.dtype == gpu_tensor.dtype?
    2. Loosen atol for float16/bfloat16 (expected 1e-2)
    3. Check if fast-math is enabled (CUDA fast-rsqrt can differ by 1e-3)
    4. Reduce batch/sequence length — numerical error accumulates

  PORTING SIGN-OFF CRITERIA:
    □ max_abs_diff < 1e-4 for float32 operations
    □ Throughput on GPU ≥ 2× throughput on CPU (otherwise the port is not worth it)
    □ No .to("cuda") calls inside the hot loop
    □ Input tensors loaded with pin_memory=True
    □ Model in eval() mode (BatchNorm uses running stats, not batch stats)
""")
print("  ✓ Section 5 passed — porting validated; correctness confirmed before benchmarking")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 15.1 complete!")
print("  You have the 6-step porting checklist, dtype trap fixes,")
print("  contiguity rules, H2D bandwidth measurement, and output validation.")
print("  Next: V.Workload_Benchmarking/15.Porting_a_Workload/15.2_bottleneck_shift.py")
print("=" * 60)
