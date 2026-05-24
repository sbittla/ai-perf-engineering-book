#!/usr/bin/env python3
"""
results_table.py  —  Shared: Print Before/After Comparison Tables
==================================================================

HOW TO USE:
    1. Run each project's benchmark scripts
    2. Fill in the RESULTS dict below with your actual measured values
    3. python results_table.py

ALTERNATIVE — print a specific project:
    python results_table.py --project 1
    python results_table.py --project 2


"""

import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=int, default=0, help="0=all, 1-4=specific project")
args = parser.parse_args()

# =============================================================================
# Fill in these values with your actual measurements
# =============================================================================
RESULTS = {

    # ── PROJECT 1: LLM Inference Optimization Lab ───────────────────────────
    "project1": {
        "title": "Project 1 — LLM Inference Optimization Lab (gpt2 / Llama-2-7B)",
        "metric": "tok/s (higher=better)",
        "rows": [
            # (Configuration, metric_value, latency_p50_ms, latency_p99_ms, gpu_mem_gb, notes)
            ("HuggingFace generate() FP32",   "??",  "??",  "??",  "??", "← baseline"),
            ("HuggingFace generate() FP16",   "??",  "??",  "??",  "??", "dtype change"),
            ("vLLM + PagedAttention FP16",     "??",  "??",  "??",  "??", ""),
            ("vLLM + Prefix Caching",          "??",  "??",  "??",  "??", "shared prefix"),
            ("torch.compile default",          "??",  "??",  "??",  "??", ""),
            ("torch.compile max-autotune",     "??",  "??",  "??",  "??", ""),
            ("TensorRT FP16 engine",           "??",  "??",  "??",  "??", ""),
        ],
        "cols": ["Configuration", "Tok/s", "P50 (ms)", "P99 (ms)", "VRAM (GB)", "Notes"],
    },

    # ── PROJECT 2: DataLoader I/O Bottleneck Hunt ───────────────────────────
    "project2": {
        "title": "Project 2 — DataLoader I/O Bottleneck Hunt (ResNet18/50)",
        "metric": "GPU Utilisation % (higher=better)",
        "rows": [
            # (Configuration, steps/sec, avg_step_ms, gpu_util_pct, notes)
            ("num_workers=0, no pin_memory",   "??", "??",  "??%",  "← baseline"),
            ("num_workers=4",                  "??", "??",  "??%",  ""),
            ("num_workers=8, pin_memory=True", "??", "??",  "??%",  ""),
            ("+ non_blocking transfers",       "??", "??",  "??%",  ""),
            ("+ GPU augmentation (torchv2)",   "??", "??",  "??%",  ""),
            ("+ NUMA binding (numactl)",        "??", "??",  "??%",  "if NUMA system"),
        ],
        "cols": ["Configuration", "Steps/sec", "Avg step (ms)", "GPU Util %", "Notes"],
    },

    # ── PROJECT 3: KV Cache Memory Pressure ─────────────────────────────────
    "project3": {
        "title": "Project 3 — KV Cache Memory Pressure (vLLM, 20 concurrent users)",
        "metric": "Throughput tok/s under concurrent load",
        "rows": [
            # (Config, tok/s, TTFT_p50_ms, TTFT_p99_ms, peak_vram_pct, notes)
            ("gpu_memory_utilization=0.60",    "??",  "??",  "??",  "60%",   "← constrained"),
            ("gpu_memory_utilization=0.85",    "??",  "??",  "??",  "85%",   ""),
            ("gpu_memory_utilization=0.90",    "??",  "??",  "??",  "90%",   ""),
            ("+ Prefix Caching (shared sys)", "??",  "??",  "??",  "??%",   ""),
            ("+ Chunked Prefill",              "??",  "??",  "??",  "??%",   ""),
            ("+ AWQ INT8 Quantization",        "??",  "??",  "??",  "??%",   "2x memory headroom"),
        ],
        "cols": ["Configuration", "Tok/s", "TTFT P50 (ms)", "TTFT P99 (ms)", "VRAM %", "Notes"],
    },

    # ── PROJECT 4: CPU-to-GPU Pipeline Flamegraph ───────────────────────────
    "project4": {
        "title": "Project 4 — CPU-to-GPU Pipeline Optimisation (ResNet18)",
        "metric": "Steps/second (higher=better)",
        "rows": [
            # (Config, steps/sec, avg_step_ms, cpu_util_pct, notes)
            ("All bottlenecks (baseline)",     "??",  "??",  "??%", "← baseline"),
            ("+ non_blocking transfers",       "??",  "??",  "??%", ""),
            ("+ Remove .item() from loop",     "??",  "??",  "??%", ""),
            ("+ GPU augmentation (torchv2)",   "??",  "??",  "??%", ""),
            ("+ No per-step file I/O",         "??",  "??",  "??%", ""),
            ("All fixes combined",             "??",  "??",  "??%", "← target"),
        ],
        "cols": ["Configuration", "Steps/sec", "Avg step (ms)", "CPU Util", "Notes"],
    },
}

def print_table(project_key: str):
    data  = RESULTS[project_key]
    cols  = data["cols"]
    rows  = data["rows"]
    title = data["title"]
    
    # Calculate column widths
    widths = [len(c) for c in cols]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    
    total_width = sum(widths) + len(cols) * 3 + 1
    
    print(f"\n{'═' * total_width}")
    print(f"  {title}")
    print(f"  Metric: {data['metric']}")
    print(f"{'─' * total_width}")
    
    # Header
    header = " │ ".join(f"{col:<{widths[i]}}" for i, col in enumerate(cols))
    print(f"  {header}")
    print(f"  {'─' * (total_width - 2)}")
    
    # Rows
    baseline_val = None
    for row in rows:
        formatted = " │ ".join(f"{str(cell):<{widths[i]}}" for i, cell in enumerate(row))
        suffix = ""
        
        # Try to calculate speedup vs baseline
        if "baseline" in str(row[-1]) or "← baseline" in str(row[-1]):
            try:
                baseline_val = float(str(row[1]).replace("??", "0"))
            except ValueError:
                baseline_val = None
        elif baseline_val and baseline_val > 0:
            try:
                current_val = float(str(row[1]).replace("??", "0"))
                if current_val > 0:
                    speedup = current_val / baseline_val
                    if speedup != 1.0:
                        suffix = f"  [{speedup:.1f}x]"
            except ValueError:
                pass
        
        print(f"  {formatted}{suffix}")
    
    print(f"{'═' * total_width}")

def print_summary():
    """Print a concise executive summary."""
    print(f"\n{'='*65}")
    print(f"  CAPSTONE PROJECTS — PORTFOLIO SUMMARY")
    print(f"  AI Systems Performance Engineering")
    print(f"{'='*65}")
    print(f"""
  PROJECT 1 — LLM Inference Optimization Lab
  ┌─ Baseline (HuggingFace FP32) : ?? tok/s
  ├─ vLLM PagedAttention FP16    : ?? tok/s  (??.?x improvement)
  └─ TensorRT / torch.compile    : ?? tok/s  (??.?x improvement)
  Tools: nsys timeline, ncu roofline, torch.profiler, nvidia-smi

  PROJECT 2 — DataLoader I/O Bottleneck Hunt
  ┌─ Baseline (num_workers=0)    : ??% GPU utilisation
  └─ Optimised (workers+NUMA)    : ??% GPU utilisation
  Tools: iostat, vmstat, opensnoop, biolatency, nsys

  PROJECT 3 — KV Cache Memory Pressure
  ┌─ Constrained (util=0.60)     : ?? tok/s, TTFT_P99=??ms
  └─ Optimised (prefix+quant)    : ?? tok/s, TTFT_P99=??ms
  Tools: nvidia-smi dmon, torch.cuda.memory_summary, vLLM flags

  PROJECT 4 — CPU-to-GPU Pipeline Flamegraph
  ┌─ Slow (all bottlenecks)      : ?? steps/sec
  └─ Fast (all fixes)            : ?? steps/sec
  Artifact: differential flamegraph diff.svg (blue=removed overhead)
  Tools: py-spy, perf, flamegraph.pl, difffolded.pl
""")
    print(f"  Replace '??' values by running each project's benchmark scripts.")
    print(f"{'='*65}")

# ── Main ──────────────────────────────────────────────────────────────────────
if args.project == 0:
    print_summary()
    for key in ["project1", "project2", "project3", "project4"]:
        print_table(key)
elif 1 <= args.project <= 4:
    print_table(f"project{args.project}")
else:
    print("--project must be 0 (all) or 1-4")
    sys.exit(1)
