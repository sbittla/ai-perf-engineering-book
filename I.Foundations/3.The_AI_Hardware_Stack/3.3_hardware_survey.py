#!/usr/bin/env python3
"""
3.3_hardware_survey.py  ─  Chapter 3: Reading GPU Specifications
=========================================================================
Covers book section 3.5:
  • Reading GPU specs programmatically (nvidia-smi, torch.cuda APIs)
  • Computing the ridge point for your specific GPU
  • Measuring actual MFU and comparing to theoretical peak
  • Monitoring GPU temperature, power, and clock throttling
  • Multi-GPU topology inspection (NVLink vs PCIe)

This is a capstone exercise for Chapter 3.  It ties together sections
3.1–3.5 by asking you to build a complete hardware report for your GPU.

Run:  python 3.3_hardware_survey.py
"""

import subprocess
import time
import torch

print("=" * 65)
print("  Exercise 03 — GPU Hardware Survey")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: GPU properties via torch.cuda
# ─────────────────────────────────────────────────────────────
print("── Section 1: GPU Properties ──")
print("""
  torch.cuda.get_device_properties() returns an object with all the
  static specs of the GPU.  This is the Python equivalent of
  nvidia-smi --query-gpu=... --format=csv
""")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)

    # TODO 1: Read props.multi_processor_count (number of SMs)
    n_sms = None  # YOUR CODE HERE

    # TODO 2: Read props.total_memory in GB
    vram_gb = None  # YOUR CODE HERE  → props.total_memory / 1e9

    # TODO 3: Read props.major and props.minor (compute capability as a float, e.g. 8.6)
    compute_capability = None  # YOUR CODE HERE  → props.major + props.minor / 10

    assert n_sms              is not None, "read n_sms"
    assert vram_gb            is not None, "read vram_gb"
    assert compute_capability is not None, "read compute_capability"

    print(f"  GPU name                : {props.name}")
    print(f"  Streaming Multiprocessors: {n_sms}")
    print(f"  VRAM                    : {vram_gb:.1f} GB")
    print(f"  CUDA Compute Capability : {compute_capability:.1f}")
    print(f"  Clock rate (MHz)        : {props.clock_rate // 1000}")
    print(f"  Memory clock (MHz)      : {props.memory_clock_rate // 1000}")
    print(f"  L2 cache size           : {props.l2_cache_size / 1e6:.0f} MB")
    print(f"  Max threads per SM      : {props.max_threads_per_multi_processor}")
    print("  ✓ Section 1 passed")

else:
    n_sms = 0
    vram_gb = 0.0
    compute_capability = 0.0
    print("  (CUDA not available — reading CPU info instead)")
    try:
        with open("/proc/cpuinfo") as f:
            lines = f.readlines()
        cpu_name = next((l.split(":")[1].strip() for l in lines if "model name" in l), "unknown")
        n_cores  = sum(1 for l in lines if "processor" in l)
        print(f"  CPU: {cpu_name}")
        print(f"  Cores: {n_cores}")
    except Exception:
        print("  (Could not read /proc/cpuinfo)")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Compute the ridge point for your GPU
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Your GPU's Ridge Point ──")
print("""
  The ridge point separates memory-bound from compute-bound operations.
  ridge_point = peak_FP16_FLOP/s  /  HBM_bandwidth_bytes/s

  Published specs by architecture:
    Ampere (A100 SXM):  312 TFLOP/s FP16,  2,000 GB/s → ridge=156 FLOPs/byte
    Hopper  (H100 SXM): 989 TFLOP/s FP16,  3,350 GB/s → ridge=295 FLOPs/byte
    Ada     (RTX 4090): 165 TFLOP/s FP16,  1,008 GB/s → ridge=164 FLOPs/byte
    Turing  (T4):        65 TFLOP/s FP16,    300 GB/s → ridge=217 FLOPs/byte
""")

# Published specs table (FP16 TC peak FLOP/s, HBM bandwidth bytes/s)
KNOWN_SPECS = {
    "a100": (312e12, 2000e9),
    "a100 80gb": (312e12, 1935e9),
    "h100": (989e12, 3350e9),
    "4090": (165.2e12, 1008e9),
    "4080": (97.5e12, 736e9),
    "4070": (95.9e12, 504e9),
    "4060": (51.5e12, 272e9),
    "3090": (142.6e12, 936e9),
    "3080": (119.4e12, 760e9),
    "3070": (91.1e12, 448e9),
    "t4":   (65.1e12, 300e9),
    "v100": (125.0e12, 900e9),
}

if DEVICE == "cuda":
    gpu_name_lower = props.name.lower()
    peak_flops, mem_bw = None, None
    for key, (pf, mb) in KNOWN_SPECS.items():
        if key in gpu_name_lower:
            peak_flops, mem_bw = pf, mb
            print(f"  Matched spec table: {key}")
            break

    if peak_flops is None:
        print(f"  GPU '{props.name}' not in table — enter specs manually:")
        print(f"  (Using RTX 4060 as fallback)")
        peak_flops, mem_bw = 51.5e12, 272e9

    # TODO 4: Compute the ridge point for your GPU
    ridge_point = None  # YOUR CODE HERE  → peak_flops / mem_bw

    assert ridge_point is not None, "compute ridge_point"
    print(f"\n  Peak FP16 FLOP/s : {peak_flops/1e12:.1f} TFLOP/s")
    print(f"  HBM bandwidth    : {mem_bw/1e9:.0f} GB/s")
    print(f"  Ridge point      : {ridge_point:.1f} FLOPs/byte")
    print()
    print(f"  Operations with AI < {ridge_point:.0f}: memory-bandwidth bound")
    print(f"  Operations with AI > {ridge_point:.0f}: compute bound (Tensor Cores useful)")
    print("  ✓ Section 2 passed")
else:
    peak_flops, mem_bw, ridge_point = 2e12, 50e9, 40.0
    print("  (CPU mode — using approximate CPU specs)")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Measure your GPU's actual MFU
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Measuring MFU (Model FLOPs Utilisation) ──")
print("""
  MFU = observed FLOP/s  /  peak FLOP/s.

  We measure a large FP16 GEMM that is well into compute-bound territory
  (high arithmetic intensity), then compute how close we are to peak.

  On a well-utilised GPU with properly aligned dimensions:
    MFU > 70%  → excellent
    MFU 40–70% → typical optimised code
    MFU < 30%  → investigate (small batch? FP32? wrong dtype?)
""")

M = 4096   # large aligned dimension — should be compute-bound

if DEVICE == "cuda":
    dtype = torch.float16
    A = torch.randn(M, M, device=DEVICE, dtype=dtype)
    B = torch.randn(M, M, device=DEVICE, dtype=dtype)

    # Warmup — critical for accurate FP16 Tensor Core measurements
    for _ in range(20): torch.mm(A, B)
    torch.cuda.synchronize()

    ITERS = 200
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

# TODO 5: Compute FLOPs for one M×M matrix multiply
#   For a square GEMM: FLOPs = 2 * M * M * M
flops_per_call = None  # YOUR CODE HERE

# TODO 6: Compute observed FLOP/s
observed_flops_per_sec = None  # YOUR CODE HERE  → flops_per_call / (avg_ms / 1000)

# TODO 7: Compute MFU as a percentage (0–100)
mfu = None  # YOUR CODE HERE  → observed_flops_per_sec / peak_flops * 100

assert flops_per_call           is not None, "compute flops_per_call"
assert observed_flops_per_sec   is not None, "compute observed_flops_per_sec"
assert mfu                      is not None, "compute mfu"
assert 0 < mfu <= 105,                       f"MFU should be 0–100%, got {mfu:.1f}%"

print(f"  Matrix size     : {M}×{M}  {'FP16' if DEVICE=='cuda' else 'FP32'}")
print(f"  Avg time        : {avg_ms:.3f} ms")
print(f"  FLOPs/call      : {flops_per_call/1e9:.1f} GFLOP")
print(f"  Observed        : {observed_flops_per_sec/1e12:.1f} TFLOP/s")
print(f"  Peak            : {peak_flops/1e12:.1f} TFLOP/s")
print(f"  MFU             : {mfu:.1f}%")
if mfu > 60:
    print(f"  Result: excellent utilisation")
elif mfu > 35:
    print(f"  Result: good — typical for real workloads (not just bare GEMM)")
else:
    print(f"  Result: low — verify FP16, aligned shapes, warmup completed")
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: GPU runtime monitoring
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: GPU Runtime Monitoring ──")
print("""
  GPUs throttle when they exceed their temperature or power limit.
  Throttled clock = lower throughput = misleading benchmarks.

  Always check for thermal throttling before trusting benchmark numbers:
    nvidia-smi dmon -s pcut -d 1   (poll every 1 second)
    nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv
""")

if DEVICE == "cuda":
    # Read current GPU state via nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,"
             "power.draw,power.limit,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            fields = result.stdout.strip().split(", ")
            if len(fields) >= 8:
                temp, clk, max_clk, pwr, pwr_lim, util, mem_used, mem_tot = fields[:8]

                # TODO 8: Compute clock throttle percentage
                #   throttle_pct = (1 - int(clk) / int(max_clk)) * 100
                throttle_pct = None  # YOUR CODE HERE

                assert throttle_pct is not None, "compute throttle_pct"
                print(f"  Temperature      : {temp.strip()} °C")
                print(f"  SM clock         : {clk.strip()} MHz  (max: {max_clk.strip()} MHz)")
                print(f"  Clock throttle   : {throttle_pct:.1f}%  "
                      f"({'OK — not throttling' if throttle_pct < 5 else 'WARNING: throttling detected'})")
                print(f"  Power draw       : {pwr.strip()} W  / {pwr_lim.strip()} W")
                print(f"  GPU utilisation  : {util.strip()}%")
                print(f"  Memory used      : {mem_used.strip()} / {mem_tot.strip()} MB")
        else:
            print("  (nvidia-smi not available or returned an error)")
            throttle_pct = 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  (nvidia-smi not available)")
        throttle_pct = 0.0

    print("  ✓ Section 4 passed")
else:
    print("  (CUDA not available — Section 4 skipped)")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Multi-GPU topology
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Multi-GPU Topology ──")
print("""
  For multi-GPU systems, the interconnect between GPUs determines
  how efficiently distributed training and inference can scale.

  NVLink (NVx in nvidia-smi topo matrix) : 600–900 GB/s
  PCIe   (PIX in nvidia-smi topo matrix) : 32–64 GB/s

  Rule: NVLink allows near-linear DDP scaling; PCIe saturates quickly.
""")

n_gpus = torch.cuda.device_count()
print(f"  GPUs available: {n_gpus}")

if n_gpus > 1:
    # TODO 9: Print the P2P access matrix between all GPU pairs
    print("\n  P2P Access Matrix:")
    print(f"  {'':5}", end="")
    for j in range(n_gpus): print(f"  GPU{j}", end="")
    print()
    for i in range(n_gpus):
        print(f"  GPU{i}", end="")
        for j in range(n_gpus):
            if i == j:
                print("    — ", end="")
            else:
                # TODO 9a: Check if GPU i can access GPU j directly
                can_access = None  # YOUR CODE HERE  → torch.cuda.can_device_access_peer(i, j)
                assert can_access is not None, f"check P2P for ({i},{j})"
                print(f"  {'P2P' if can_access else 'off'}", end="")
        print()

    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "--matrix"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("\n  nvidia-smi topo output:")
            for line in result.stdout.strip().split("\n")[:12]:
                print(f"    {line}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
elif DEVICE == "cuda":
    print(f"  Single GPU system ({props.name})")
    print(f"  Multi-GPU topology only visible on multi-GPU machines.")
    print(f"  In a cloud multi-GPU instance: nvidia-smi topo --matrix")
else:
    print("  (CUDA not available — multi-GPU section skipped)")

print("\n  ✓ Section 5 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Build your hardware summary
# ─────────────────────────────────────────────────────────────
print("\n── Section 6: Your Hardware Summary ──")
print("""
  Summarise the key performance numbers for your GPU.
  This is the reference card you will use throughout Parts II–V.
""")

if DEVICE == "cuda":
    # TODO 10: Fill in the summary — all values computed above
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  Hardware: {props.name:<41}│")
    print(f"  │  SMs: {n_sms:<3}   VRAM: {vram_gb:.0f} GB   CC: {compute_capability:.1f}          │")
    print(f"  │  Peak FP16 : {peak_flops/1e12:>6.1f} TFLOP/s                         │")
    print(f"  │  HBM BW    : {mem_bw/1e9:>6.0f} GB/s                            │")
    print(f"  │  Ridge pt  : {ridge_point:>6.1f} FLOPs/byte                      │")
    print(f"  │  Bare GEMM : {mfu:>5.1f}% MFU                              │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print(f"")
    print(f"  Keep these numbers.  Every profiling result in this book")
    print(f"  should be interpreted relative to these ceilings.")
else:
    print("  (CUDA not available — run on a GPU machine for full summary)")

print("\n  ✓ Section 6 passed")

print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Exercise 03 (Chapter 3) done!")
print()
print("  You have completed all exercises in Part I — Foundations.")
print("  Before continuing to Part II, verify:")
print("    I.Foundations/1.What_Is_AI_Performance_Engineering/1.1_roofline_model.py   ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.1_tensors.py  ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py ✓")
print("    I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py ✓")
print("    I.Foundations/3.The_AI_Hardware_Stack/3.1_cpu_and_memory.py ✓")
print("    I.Foundations/3.The_AI_Hardware_Stack/3.2_gpu_memory_and_compute.py ✓")
print("    I.Foundations/3.The_AI_Hardware_Stack/3.3_hardware_survey.py ✓")
print("=" * 65)
