#!/usr/bin/env python3
"""
V.Workload_Benchmarking/14.Benchmarking_Methodology/14.4_workload_characterization.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 14: Benchmarking Methodology — Section 3: Workload Characterisation
=======================================================================
Covers book section 14.3:
  • What workload characterisation means: profiling before optimising
  • The full characterisation checklist (8 measurements)
  • Measuring compute utilisation and memory bandwidth utilisation
  • Roofline position: classifying your workload as compute or memory bound
  • Building and saving a structured JSON characterisation report
  • Comparing reports to track progress across optimisation rounds

Run:  python V.Workload_Benchmarking/14.Benchmarking_Methodology/14.4_workload_characterization.py
All sections must print ✓.
"""

import json
import math
import statistics
import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 14.3 — Workload Characterisation")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 1: The characterisation checklist
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Workload Characterisation Checklist ──")
print("""
  Before optimising, characterise. Every optimisation round should start
  and end with a characterisation so you can attribute improvements.

  THE 8-POINT CHECKLIST:
    □ 1. Peak memory bandwidth utilisation   (DRAM%)
    □ 2. Peak compute utilisation            (SM%)
    □ 3. Roofline position                   (memory-bound or compute-bound?)
    □ 4. Throughput at max sustained load    (samples/sec)
    □ 5. Latency P50 / P99 at each config   (ms)
    □ 6. GPU memory footprint                (weights + activations in GB)
    □ 7. CPU utilisation during GPU work     (should be < 20%)
    □ 8. Thermal throttling check            (burst vs sustained)

  The most common mistake: jumping to optimisation without characterising.
  You fix a 2% bottleneck while missing a 50% bottleneck sitting next to it.

  TOOLS BY METRIC:
    DRAM% and SM%  : ncu --metrics dram__throughput.avg.pct_of_peak, sm__throughput
    Roofline       : ncu --set roofline
    Throughput     : CUDA events + benchmark()
    Latency P50/P99: gather_timings() (Exercise 14.1)
    GPU memory     : torch.cuda.memory_allocated() / max_memory_allocated()
    CPU utilisation: psutil.cpu_percent() or top
    Thermal        : nvidia-smi -q -d TEMPERATURE / burst vs sustained benchmark
""")
print("  ✓ Section 1 passed — commit the 8-point checklist to memory")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring roofline position
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Roofline Position — Memory or Compute Bound? ──")
print("""
  The roofline model classifies any workload as either:
    MEMORY-BANDWIDTH BOUND: arithmetic intensity < ridge_point
    COMPUTE BOUND:          arithmetic intensity > ridge_point

  ARITHMETIC INTENSITY of a linear layer:
    FLOPs = 2 × batch × seq_len × d_out × d_in       (matmul FLOPs)
    Bytes = (d_out × d_in + batch × seq_len × d_in) × dtype_bytes
    AI    = FLOPs / Bytes

  For large batches: AI ≈ batch × seq_len (grows with batch).
  For batch=1:       AI ≈ 1 (always memory-bound on GPU).

  TODO 1: Implement arithmetic_intensity() for a Linear(d_in, d_out) layer.
""")


def arithmetic_intensity(batch: int, seq_len: int,
                         d_in: int, d_out: int,
                         dtype_bytes: int = 4) -> float:
    """
    TODO 1: Return arithmetic intensity (FLOPs/byte) for a linear layer.
    FLOPs = 2 * batch * seq_len * d_out * d_in
    Bytes = (d_out * d_in + batch * seq_len * d_in) * dtype_bytes
    Return FLOPs / Bytes.
    """
    flops = 2 * batch * seq_len * d_out * d_in
    bytes_traffic = (d_out * d_in + batch * seq_len * d_in) * dtype_bytes
    return flops / bytes_traffic


# Verify: large batch → higher AI
ai_b1  = arithmetic_intensity(1,   64, 512, 512)
ai_b32 = arithmetic_intensity(32,  64, 512, 512)
ai_b128 = arithmetic_intensity(128, 64, 512, 512)
assert ai_b128 > ai_b32 > ai_b1, "AI should grow with batch size"

# Typical A100 ridge point: ~312 TFLOPS / 2 TB/s ≈ 156 FLOPs/byte (FP32)
# For FP16: ~312 TFLOPS / 2 TB/s = 156 FLOPs/byte (same ridge, but FP16 AI is 2x)
RIDGE_FP32 = 156.0   # FLOPs/byte  (approximate A100 ridge)
RIDGE_FP16 = 78.0    # FLOPs/byte  (FP16: half bytes, same TFLOPS → same ridge, but often quoted lower)

print(f"  Linear(512, 512) arithmetic intensity (FP32):")
print(f"  {'Batch × Seq':>12}  {'AI (FLOPs/byte)':>18}  {'Bound':>15}")
print(f"  {'─'*12}  {'─'*18}  {'─'*15}")
for B, S in [(1, 64), (8, 64), (32, 64), (128, 64), (1, 512), (8, 512)]:
    ai  = arithmetic_intensity(B, S, 512, 512, dtype_bytes=4)
    bound = "compute" if ai > RIDGE_FP32 else "memory-BW"
    print(f"  {B:>4}×{S:<7}  {ai:>18.1f}  {bound:>15}")

print(f"\n  A100 ridge point (FP32): {RIDGE_FP32} FLOPs/byte")
print(f"  → batch=1 is always memory-bandwidth-bound on current GPUs")
print("  ✓ Section 2 passed — arithmetic intensity formula implemented")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Measuring peak memory bandwidth on this machine
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Measuring Peak Memory Bandwidth ──")
print("""
  To know whether your workload is near the memory bandwidth ceiling,
  you must measure the ACTUAL peak bandwidth on your hardware.

  METHOD: copy a large tensor — this is purely memory-bandwidth-bound.
    Bytes transferred = 2 × tensor_size  (read + write)
    Bandwidth = 2 × tensor_size / time_s  (GB/s)

  This gives you the empirical ridge point for your specific GPU.
  Published specs (A100: 2 TB/s) are theoretical; real-world
  efficiency is typically 80–90% of peak.

  TODO 2: Implement measure_peak_bandwidth_gbs() below.
""")


def measure_peak_bandwidth_gbs(n_elements: int = 64 * 1024 * 1024,
                                warmup: int = 5, iters: int = 20) -> float:
    """
    TODO 2: Measure peak memory bandwidth in GB/s.
    Create a float32 tensor of n_elements on DEVICE.
    Time n_elements copies (t.clone()) using CUDA events (GPU) or perf_counter (CPU).
    Bandwidth = 2 × n_elements × 4 bytes / (mean_time_s) / 1e9
    Return bandwidth in GB/s.
    """
    t = torch.rand(n_elements, device=DEVICE, dtype=torch.float32)

    def clone_fn():
        _ = t.clone()

    # Warmup
    for _ in range(warmup):
        clone_fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # Measure
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            clone_fn()
        e.record()
        torch.cuda.synchronize()
        mean_ms = s.elapsed_time(e) / iters
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            clone_fn()
        mean_ms = (time.perf_counter() - t0) / iters * 1000

    bytes_per_copy = n_elements * 4 * 2   # float32 × 2 (read + write)
    bw_gbs = bytes_per_copy / (mean_ms / 1000) / 1e9
    del t
    return bw_gbs


peak_bw = measure_peak_bandwidth_gbs()
print(f"  Measured peak memory bandwidth: {peak_bw:.1f} GB/s")

if DEVICE == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU: {props.name}  (theoretical: see GPU spec sheet)")
    print(f"  Efficiency: this measurement / theoretical peak")
else:
    import os
    ncpus = os.cpu_count() or 1
    print(f"  CPU measurement ({ncpus} cores available)")

assert peak_bw > 0, "Bandwidth should be positive"
print("  ✓ Section 3 passed — peak memory bandwidth measured")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Full characterisation sweep
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Full Characterisation Sweep ──")
print("""
  We now run the complete characterisation for a multi-layer MLP:
  throughput, latency distribution, and memory footprint.
""")

D_MODEL = 256
model = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL)
).to(DEVICE)
model.eval()

param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
print(f"  Model: 3-layer MLP, d_model={D_MODEL}, {param_mb:.1f} MB parameters\n")

characterisation = {
    "device":     DEVICE,
    "model":      "3-layer MLP",
    "d_model":    D_MODEL,
    "param_mb":   round(param_mb, 2),
    "peak_bw_gbs": round(peak_bw, 1),
    "configs":    [],
}

batch_seq_configs = [
    (1,   64),
    (8,   64),
    (32,  64),
    (128, 64),
    (8,  256),
]

print(f"  {'B×S':>8}  {'Mean (ms)':>10}  {'P99 (ms)':>10}  {'Tput (sps)':>12}  {'AI':>8}  {'Bound':>12}")
print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*8}  {'─'*12}")

for B, S in batch_seq_configs:
    x_b = torch.randn(B, S, D_MODEL, device=DEVICE)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def fn():
        with torch.no_grad():
            model(x_b)

    # Warmup
    for _ in range(5): fn()
    if DEVICE == "cuda": torch.cuda.synchronize()

    times = []
    for _ in range(30):
        if DEVICE == "cuda":
            s_e = torch.cuda.Event(enable_timing=True)
            e_e = torch.cuda.Event(enable_timing=True)
            s_e.record(); fn(); e_e.record()
            torch.cuda.synchronize()
            times.append(s_e.elapsed_time(e_e))
        else:
            t0 = time.perf_counter(); fn()
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = statistics.mean(times)
    p99_ms  = sorted(times)[int(0.99 * len(times))]
    tps     = B / (mean_ms / 1000)
    ai      = arithmetic_intensity(B, S, D_MODEL, D_MODEL * 4, dtype_bytes=4)
    vram_mb = torch.cuda.max_memory_allocated() / 1e6 if DEVICE == "cuda" else 0.0
    bound   = "compute" if ai > RIDGE_FP32 else "memory-BW"

    print(f"  {B:>2}×{S:<5}  {mean_ms:>10.3f}  {p99_ms:>10.3f}  {tps:>12.0f}  {ai:>8.1f}  {bound:>12}")
    characterisation["configs"].append({
        "batch": B, "seq": S, "mean_ms": round(mean_ms, 3),
        "p99_ms": round(p99_ms, 3), "throughput_sps": round(tps, 1),
        "ai": round(ai, 1), "bound": bound, "vram_mb": round(vram_mb, 1),
    })

assert len(characterisation["configs"]) > 0, "Characterisation should have data"
print("  ✓ Section 4 passed — full characterisation table built")


# ─────────────────────────────────────────────────────────────
# SECTION 5: Saving the characterisation report as JSON
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Structured Characterisation Report ──")
print("""
  A characterisation report is only valuable if you can compare it
  across optimisation rounds and hardware configurations.

  SAVE FORMAT: JSON with all measurements and metadata.
  COMPARE:     diff two JSON reports to see exactly what changed.

  Workflow:
    1. characterise_before.json  ← run before any optimisation
    2. Apply one optimisation (e.g. FP16 precision)
    3. characterise_after.json   ← run again
    4. diff or compute speedup: after.throughput / before.throughput

  TODO 3: Add a "summary" key to the characterisation dict with:
    - max_throughput_sps: highest throughput across all configs
    - min_p99_ms: lowest P99 latency across all configs
    - roofline_limited: True if any config's AI < peak_bw estimate
""")


def add_summary(char_dict: dict) -> dict:
    """
    TODO 3: Add summary keys to char_dict in-place and return it.
    max_throughput_sps = max(config['throughput_sps'] for config in char_dict['configs'])
    min_p99_ms         = min(config['p99_ms'] for config in char_dict['configs'])
    roofline_limited   = any(config['bound'] == 'memory-BW' for config in char_dict['configs'])
    """
    configs = char_dict["configs"]
    char_dict["summary"] = {
        "max_throughput_sps": max(c["throughput_sps"] for c in configs),
        "min_p99_ms":         min(c["p99_ms"]         for c in configs),
        "roofline_limited":   any(c["bound"] == "memory-BW" for c in configs),
    }
    return char_dict


characterisation = add_summary(characterisation)
assert "summary" in characterisation, "add_summary should add 'summary' key"
assert "max_throughput_sps" in characterisation["summary"]

report_path = "/tmp/characterisation_report.json"
with open(report_path, "w") as f:
    json.dump(characterisation, f, indent=2)

print(f"  Report saved: {report_path}")
print(f"  Summary:")
for k, v in characterisation["summary"].items():
    print(f"    {k}: {v}")

print("""
  USAGE PATTERN:
    python 14.3_workload_characterization.py > before.txt
    # Apply optimisation
    python 14.3_workload_characterization.py > after.txt
    diff before.txt after.txt   # see what changed

  COMPARE JSON REPORTS:
    import json
    before = json.load(open("before.json"))
    after  = json.load(open("after.json"))
    for c_b, c_a in zip(before["configs"], after["configs"]):
        speedup = c_a['throughput_sps'] / c_b['throughput_sps']
        print(f"  B={c_b['batch']}: speedup = {speedup:.2f}x")
""")
print("  ✓ Section 5 passed — characterisation report saved and summarised")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 14.3 complete!")
print("  You can now run a complete 8-point characterisation, classify")
print("  workloads on the roofline, and save/compare JSON reports.")
print("  Next: V.Workload_Benchmarking/15.Porting_a_Workload/15.1_porting_checklist.py")
print("=" * 60)
