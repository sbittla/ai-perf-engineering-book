#!/usr/bin/env python3
"""
workload_characterize.py  ─  Phase 5 / Module 11: Full Workload Characterization
==================================================================================

HOW TO RUN
    python workload_characterize.py                   # full characterization
    python workload_characterize.py --quick           # fast sweep only
    python workload_characterize.py --output report.json


"""

import argparse, os, sys, time, json, statistics, subprocess
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model",   default="small", choices=["tiny","small","medium"])
parser.add_argument("--quick",   action="store_true", help="Fewer sweep points")
parser.add_argument("--output",  default="characterization_report.json")
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
report = {"model": args.model, "device": DEVICE, "sections": {}}

def sep(t): print(f"\n{'═'*65}\n  {t}\n{'─'*65}")

def cuda_ms(fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 – Hardware Inventory
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 0 – Hardware Inventory")
hw = {}
if DEVICE == "cuda":
    p = torch.cuda.get_device_properties(0)
    hw = {
        "gpu_name":    p.name,
        "vram_gb":     round(p.total_memory / 1e9, 2),
        "n_sms":       p.multi_processor_count,
        "compute_cap": f"{p.major}.{p.minor}",
        "cuda_version": torch.version.cuda,
    }
    print(f"  GPU:  {p.name}")
    print(f"  VRAM: {p.total_memory/1e9:.1f}GB")
    print(f"  SMs:  {p.multi_processor_count}")
    print(f"  CC:   {p.major}.{p.minor}  CUDA: {torch.version.cuda}")

try:
    r = subprocess.run(["nproc"], capture_output=True, text=True)
    hw["cpu_cores"] = int(r.stdout.strip())
    print(f"  CPU cores: {hw['cpu_cores']}")
except Exception: pass

report["hardware"] = hw

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – Peak Bandwidth & Compute (Roofline anchors)
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 1 – Peak Memory Bandwidth & Compute (Roofline Anchors)")
print("  These establish the theoretical ceilings for your workload.\n")

roofline = {}

if DEVICE == "cuda":
    # Peak memory bandwidth
    n = 128 * 1024 * 1024 // 4
    t = torch.rand(n, device=DEVICE, dtype=torch.float32)
    ms_bw = cuda_ms(lambda: t.clone(), warmup=10, iters=30)
    peak_bw = n * 4 * 2 / (ms_bw / 1000) / 1e9
    roofline["peak_mem_bw_gbs"] = round(peak_bw, 1)
    print(f"  Peak memory bandwidth : {peak_bw:.1f} GB/s")
    del t

    # Peak FP16 compute (large matmul)
    M = 4096
    try:
        A = torch.rand(M, M, device=DEVICE, dtype=torch.float16)
        B = torch.rand(M, M, device=DEVICE, dtype=torch.float16)
        ms_c = cuda_ms(lambda: torch.mm(A, B), warmup=5, iters=20)
        peak_tflops = 2 * M**3 / (ms_c / 1000) / 1e12
        roofline["peak_fp16_tflops"] = round(peak_tflops, 1)
        ridge = peak_tflops * 1e12 / (peak_bw * 1e9)
        roofline["ridge_point_flop_per_byte"] = round(ridge, 1)
        print(f"  Peak FP16 compute     : {peak_tflops:.1f} TFLOP/s")
        print(f"  Ridge point           : {ridge:.1f} FLOP/byte")
        print(f"  → Workloads with AI < {ridge:.0f} are MEMORY-BOUND")
        del A, B
    except RuntimeError:
        print("  (OOM on peak matmul — reduce M)")

report["sections"]["roofline"] = roofline

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Throughput vs Batch Size Curve
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 2 – Throughput vs Batch Size (the T-B Curve)")
print("  Run with: python workload_characterize.py\n")

if not HAS_MODEL:
    print("  (model.py not found — skipping)")
else:
    model = TinyTransformer(args.model, max_seq=65).to(DEVICE).eval()
    batches_cfg = [1,2,4,8,16,32] if args.quick else [1,2,4,8,16,32,64,128]
    tput_curve  = []

    print(f"  {'Batch':>7}  {'Time(ms)':>10}  {'Tok/s':>10}  {'VRAM(GB)':>10}  {'Bound':>12}")
    print(f"  {'─'*7}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}")

    for B in batches_cfg:
        try:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.randint(0, 50257, (B, 64), device=DEVICE)
            ms  = cuda_ms(lambda: model(ids))
            tps = B * 64 / (ms / 1000)
            vram = torch.cuda.max_memory_allocated() / 1e9

            # Rough AI for a transformer block
            D = {"tiny":128,"small":256,"medium":512}[args.model]
            ai = 2 * B * 64 * D / (B * 64 * D * 2 + D * D * 2)   # very rough
            bound = "memory-BW" if ai < roofline.get("ridge_point_flop_per_byte", 50) else "compute"

            tput_curve.append({"batch": B, "ms": round(ms,2),
                               "tps": round(tps,1), "vram_gb": round(vram,3)})
            print(f"  {B:>7}  {ms:>10.2f}  {tps:>10.1f}  {vram:>10.3f}  {bound:>12}")
            del ids
        except RuntimeError:
            print(f"  {B:>7}  OOM")
            break

    report["sections"]["throughput_curve"] = tput_curve
    del model

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – Latency Distribution (P50 / P95 / P99)
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 3 – Latency Percentiles at batch=1 (interactive SLA)")
if HAS_MODEL:
    model = TinyTransformer(args.model, max_seq=65).to(DEVICE).eval()
    ids   = torch.randint(0, 50257, (1, 64), device=DEVICE)
    N_TRIALS = 50

    lats = []
    for _ in range(N_TRIALS + 5):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad(): model(ids)
        e.record()
        torch.cuda.synchronize()
        lats.append(s.elapsed_time(e))
    lats = sorted(lats[5:])

    lat_stats = {
        "p50": round(lats[int(0.50*N_TRIALS)], 3),
        "p95": round(lats[int(0.95*N_TRIALS)], 3),
        "p99": round(lats[int(0.99*N_TRIALS)], 3),
        "max": round(lats[-1], 3),
    }
    for k, v in lat_stats.items():
        print(f"  {k.upper():>5}: {v:.3f} ms")
    report["sections"]["latency_p1"] = lat_stats
    del model

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – Thermal Throttling Check
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 4 – Thermal Throttling Check")
print("""
  Compare burst (first 5s) vs sustained (30s) throughput.
  If sustained < burst × 0.9, the GPU is thermally throttling.
  Fix: improve airflow, reduce power limit, or reduce clock offset.
""")
if DEVICE == "cuda" and HAS_MODEL:
    model = TinyTransformer(args.model, max_seq=65).to(DEVICE).eval()
    ids   = torch.randint(0, 50257, (8, 64), device=DEVICE)

    def run_tput(duration_s):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < duration_s:
            with torch.no_grad(): model(ids)
            torch.cuda.synchronize()
            n += 1
        return n / (time.perf_counter() - t0)

    burst_tps     = run_tput(5)
    sustained_tps = run_tput(20)
    throttle_pct  = (burst_tps - sustained_tps) / burst_tps * 100

    print(f"  Burst (5s):     {burst_tps:.1f} forward passes/sec")
    print(f"  Sustained (20s): {sustained_tps:.1f} forward passes/sec")
    print(f"  Throttle drop:  {throttle_pct:.1f}%  {'← THROTTLING DETECTED' if throttle_pct > 10 else '← OK'}")

    report["sections"]["thermal"] = {
        "burst_tps": round(burst_tps, 1),
        "sustained_tps": round(sustained_tps, 1),
        "throttle_pct": round(throttle_pct, 1),
    }
    del model

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – Save JSON Report
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 5 – Saving Report")
with open(args.output, "w") as f:
    json.dump(report, f, indent=2)
print(f"  ✓ Report saved: {args.output}")
print(f"  Use for: before/after comparison, hardware benchmarking, MLPerf prep")
print(f"""
  NEXT STEPS
    1. Profile hottest kernel: ncu --set roofline python workload_characterize.py --quick
    2. Trace timeline:         nsys profile python workload_characterize.py --quick
    3. Compare hardware:       run on CPU, single GPU, multi-GPU and diff the JSON files
    4. Submit to MLPerf:       fill LoadGen harness with your latency/throughput numbers
""")
