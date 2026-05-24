#!/usr/bin/env python3
"""
workload_port.py  ─  Phase 5 / Module 10: Workload Porting & Bottleneck Shift
==============================================================================

HOW TO RUN
    python workload_port.py                    # all configurations
    python workload_port.py --mode cpu         # CPU only
    python workload_port.py --mode compare     # summary comparison table


"""

import argparse, os, sys, time, json
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--mode",  default="all",
    choices=["all","cpu","gpu_fp32","gpu_fp16","gpu_int8","gpu_compile","compare"])
parser.add_argument("--model", default="small")
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--seq",   type=int, default=64)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

def sep(t): print(f"\n{'═'*65}\n  {t}\n{'─'*65}")

def timed(fn, warmup=3, iters=10):
    for _ in range(warmup): fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters): fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

def cuda_ms(fn, warmup=3, iters=10):
    if not torch.cuda.is_available():
        return timed(fn, warmup, iters)
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

print(f"\n{'='*65}")
print(f"  workload_port.py  ─  Phase 5 Module 10")
print(f"  Model: {args.model}  Batch: {args.batch}  Seq: {args.seq}")
print(f"{'='*65}")

if not HAS_MODEL:
    print("  model.py not found."); exit(0)

results = []

def run_config(label, device, dtype, use_compile=False, use_int8=False):
    """Run one hardware/precision configuration and record results."""
    try:
        torch.cuda.reset_peak_memory_stats() if device == "cuda" else None

        model = TinyTransformer(args.model, max_seq=args.seq+1)
        model = model.to(device)

        if dtype == torch.float16:
            model = model.half()
        elif dtype == torch.bfloat16:
            model = model.to(dtype=torch.bfloat16)
        elif use_int8:
            # Simulate INT8 by quantizing linear layers dynamically
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
        model.eval()

        if use_compile and device == "cuda":
            model = torch.compile(model, mode="default")
            ids_warm = torch.randint(0, 50257, (1, args.seq), device=device)
            with torch.no_grad(): model(ids_warm)
            torch.cuda.synchronize()

        ids = torch.randint(0, 50257, (args.batch, args.seq), device=device)

        if device == "cpu" and use_int8:
            ms = timed(lambda: model(ids), warmup=2, iters=5)
        elif device == "cuda":
            ms = cuda_ms(lambda: model(ids), warmup=3, iters=10)
        else:
            ms = timed(lambda: model(ids), warmup=2, iters=5)

        tps  = args.batch * args.seq / (ms / 1000)
        vram = (torch.cuda.max_memory_allocated() / 1e9
                if device == "cuda" else 0.0)
        param_mb = sum(p.element_size() * p.numel() for p in model.parameters()) / 1e6

        result = {
            "label": label, "device": device,
            "ms": round(ms, 3), "tps": round(tps, 1),
            "vram_gb": round(vram, 3), "param_mb": round(param_mb, 1),
        }
        results.append(result)

        sep(f"{label}")
        print(f"  Time        : {ms:.2f} ms")
        print(f"  Throughput  : {tps:.1f} tok/s")
        if device == "cuda":
            print(f"  VRAM        : {vram:.3f} GB")
        print(f"  Param size  : {param_mb:.1f} MB  ({dtype})")

        # Diagnose bottleneck
        if device == "cpu":
            print(f"""
  BOTTLENECK ANALYSIS (CPU):
    Run: perf stat -e cache-misses,instructions python workload_port.py --mode cpu
    Look for: IPC (instructions per cycle) < 1.0 → stalls
    Look for: cache-miss rate > 5% → memory-bound
    Fix:  vectorise (use PyTorch ops, not Python loops)
          increase batch to improve data reuse
          check NUMA binding: numactl --hardware
    """)
        elif device == "cuda":
            ai = 2 * args.batch * args.seq * 256 / (args.batch * args.seq * 256 * 2 + 256*256*2)
            bound = "memory-BW" if ai < 30 else "compute"
            print(f"""
  BOTTLENECK ANALYSIS (GPU {dtype}):
    Estimated arithmetic intensity: {ai:.1f} FLOP/byte → {bound}
    Verify with ncu:
      ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \\
              sm__throughput.avg.pct_of_peak_sustained_elapsed \\
          python workload_port.py --mode {label.lower().replace(' ','_')}
    """)

        del model, ids
        torch.cuda.empty_cache() if device == "cuda" else None
        return result

    except RuntimeError as e:
        print(f"  {label}: FAILED — {e}")
        return None

# ── Run selected modes ─────────────────────────────────────────────────────────
CUDA = "cuda" if torch.cuda.is_available() else None

mode_map = {
    "cpu":       lambda: run_config("CPU FP32",        "cpu",          torch.float32),
    "gpu_fp32":  lambda: run_config("GPU FP32",        CUDA or "cpu",  torch.float32),
    "gpu_fp16":  lambda: run_config("GPU FP16",        CUDA or "cpu",  torch.float16),
    "gpu_int8":  lambda: run_config("GPU INT8 (dyn)",  "cpu",          torch.float32, use_int8=True),
    "gpu_compile":lambda: run_config("GPU FP16+compile",CUDA or "cpu", torch.float16, use_compile=True),
}

if args.mode == "all" or args.mode == "compare":
    for fn in mode_map.values():
        fn()
else:
    mode_map.get(args.mode, lambda: print("Unknown mode"))()

# ── Comparison Table ──────────────────────────────────────────────────────────
if results:
    sep("COMPARISON TABLE: Bottleneck Shifts Across Hardware")
    baseline = results[0]["tps"]
    print(f"  {'Config':<22}  {'Tok/s':>10}  {'Speedup':>8}  {'VRAM':>8}  {'Bottleneck (expected)':>25}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*25}")
    bottlenecks = {
        "CPU FP32":         "IPC / DRAM latency",
        "GPU FP32":         "compute (large matmul)",
        "GPU FP16":         "memory-BW (decode)",
        "GPU INT8 (dyn)":   "dequantize overhead",
        "GPU FP16+compile": "residual unfused ops",
    }
    for r in results:
        spd  = r["tps"] / baseline
        vram = f"{r['vram_gb']:.2f}GB" if r["vram_gb"] > 0 else "CPU"
        bot  = bottlenecks.get(r["label"], "unknown")
        print(f"  {r['label']:<22}  {r['tps']:>10.1f}  {spd:>7.2f}×  {vram:>8}  {bot:>25}")

    print(f"""
  KEY INSIGHT
    Each configuration has a DIFFERENT limiting bottleneck.
    Applying the wrong optimisation (e.g. compute optimisation on a
    memory-bound workload) yields zero speedup.
    
    Always profile FIRST with ncu/nsys, THEN optimise the actual bottleneck.
    This is the core skill this role demands.
  """)
