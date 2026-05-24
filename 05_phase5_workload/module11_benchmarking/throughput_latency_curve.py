#!/usr/bin/env python3
"""
throughput_latency_curve.py  ─  Phase 5 / Module 11: T-L Tradeoff Curve
=========================================================================

HOW TO RUN
    python throughput_latency_curve.py
    python throughput_latency_curve.py --plot   # saves matplotlib chart


"""

import argparse, os, sys, statistics
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="small")
parser.add_argument("--tokens", type=int, default=50)
parser.add_argument("--plot",   action="store_true")
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_ms(fn, warmup=3, iters=8):
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
print(f"  throughput_latency_curve.py  ─  Phase 5 Module 11")
print(f"  Device: {DEVICE}   Model: {args.model}")
print(f"{'='*65}\n")

if not HAS_MODEL:
    print("  model.py not found. Exiting."); exit(0)

configs = [
    ("FP32 batch=1",    1,  torch.float32, False),
    ("FP16 batch=1",    1,  torch.float16, False),
    ("FP16 batch=4",    4,  torch.float16, False),
    ("FP16 batch=8",    8,  torch.float16, False),
    ("FP16 batch=16",   16, torch.float16, False),
    ("FP16 batch=32",   32, torch.float16, False),
    ("FP16+compile b=1",1,  torch.float16, True),
    ("FP16+compile b=8",8,  torch.float16, True),
]

results = []
print(f"  {'Config':<22}  {'Lat P50(ms)':>12}  {'Lat P99(ms)':>12}  {'Tok/s':>10}  {'VRAM(GB)':>10}")
print(f"  {'─'*22}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")

for label, B, dtype, compile_flag in configs:
    try:
        torch.cuda.reset_peak_memory_stats()
        m = TinyTransformer(args.model, max_seq=args.tokens+33)
        m = m.to(DEVICE, dtype=dtype if dtype != torch.float16 else torch.float32)
        if dtype == torch.float16: m = m.half()
        m.eval()
        if compile_flag:
            m = torch.compile(m, mode="default")
            prompt = torch.randint(0, 50257, (B, 32), device=DEVICE)
            with torch.no_grad(): m.generate(prompt, max_new_tokens=5, temperature=0)

        prompt = torch.randint(0, 50257, (B, 32), device=DEVICE)
        lats = []
        for _ in range(15):
            ms = cuda_ms(
                lambda: m.generate(prompt, max_new_tokens=args.tokens, temperature=0),
                warmup=0, iters=1
            )
            lats.append(ms)
        lats.sort()
        p50 = lats[int(0.50*len(lats))]
        p99 = lats[-1]
        tps = args.tokens / (p50 / 1000) * B
        vram = torch.cuda.max_memory_allocated() / 1e9

        results.append({"label": label, "batch": B, "p50_ms": p50,
                        "p99_ms": p99, "tps": tps, "vram_gb": vram})
        print(f"  {label:<22}  {p50:>12.1f}  {p99:>12.1f}  {tps:>10.1f}  {vram:>10.3f}")
        del m
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  {label:<22}  OOM: {str(e)[:30]}")

# ── Pareto frontier ───────────────────────────────────────────────────────────
print(f"\n  PARETO FRONTIER (max throughput per latency budget)")
print(f"  {'Lat budget (ms)':>18}  {'Best config':>25}  {'Max tok/s':>12}")
print(f"  {'─'*18}  {'─'*25}  {'─'*12}")

for budget in [10, 20, 50, 100, 200, 500]:
    candidates = [r for r in results if r["p99_ms"] <= budget]
    if not candidates: continue
    best = max(candidates, key=lambda r: r["tps"])
    print(f"  {budget:>18}ms  {best['label']:>25}  {best['tps']:>12.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
if args.plot:
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#06080a")
        colors = ["#3fb950","#58a6ff","#f78166","#e3b341","#a5d6ff","#ff9e64","#b9f18d","#ff6b6b"]
        for i, r in enumerate(results):
            ax.scatter(r["p50_ms"], r["tps"], s=120, color=colors[i % len(colors)],
                       label=r["label"], zorder=5)
        ax.set_xlabel("P50 Latency (ms)", color="#888")
        ax.set_ylabel("Throughput (tok/s)", color="#888")
        ax.set_title(f"Throughput-Latency Curve ({args.model})", color="#ccc")
        ax.legend(fontsize=8, framealpha=0.3)
        ax.tick_params(colors="#888")
        plt.tight_layout()
        out = "tl_curve.png"
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
        print(f"\n  ✓ Plot saved: {out}")
    except ImportError:
        print("\n  (matplotlib not installed — skipping plot)")
