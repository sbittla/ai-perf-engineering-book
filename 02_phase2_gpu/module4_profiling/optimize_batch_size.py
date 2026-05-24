#!/usr/bin/env python3
"""
optimize_batch_size.py  ─  Phase 2 / Module 4: Find the Optimal Batch Size
============================================================================

HOW TO RUN
    python optimize_batch_size.py                     # sweep all batch sizes
    python optimize_batch_size.py --task lm           # language model
    python optimize_batch_size.py --task image        # image classifier
    python optimize_batch_size.py --max-batch 256     # custom ceiling

AFTER RUNNING
    nsys profile --stats=true python optimize_batch_size.py --max-batch 32


"""

import argparse, gc
import torch
import torch.nn as nn
from torchvision import models

parser = argparse.ArgumentParser()
parser.add_argument("--task",      default="lm", choices=["lm","image"])
parser.add_argument("--max-batch", type=int, default=128)
parser.add_argument("--seq-len",   type=int, default=64)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ── Build model ───────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  optimize_batch_size.py  ─  Phase 2 Module 4")
print(f"  Task: {args.task}   Device: {DEVICE}")
print(f"{'='*65}\n")

if args.task == "lm":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
    try:
        from model import TinyTransformer
        model = TinyTransformer("medium", max_seq=args.seq_len + 1).to(DEVICE).eval()
        TASK_NAME = "TinyTransformer-medium LM"
    except ImportError:
        # fallback: small embedding + linear
        model = nn.Sequential(
            nn.Embedding(50257, 512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 50257)
        ).to(DEVICE).eval()
        TASK_NAME = "Simple LM fallback"
else:
    model = models.resnet50(weights=None).to(DEVICE).eval()
    TASK_NAME = "ResNet-50 image classifier"

print(f"  Model: {TASK_NAME}")
nparams = sum(p.numel() for p in model.parameters())
print(f"  Params: {nparams/1e6:.1f}M\n")

# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"  {'Batch':>8}  {'Time(ms)':>10}  {'Lat/sample':>12}  "
      f"{'Samples/s':>12}  {'VRAM(GB)':>10}  {'Status':>15}")
print(f"  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*15}")

results = []
batch = 1
while batch <= args.max_batch:
    try:
        torch.cuda.reset_peak_memory_stats()
        if args.task == "lm":
            ids = torch.randint(0, 50257, (batch, args.seq_len), device=DEVICE)
            def forward():
                with torch.no_grad():
                    if hasattr(model, 'generate'):
                        return model(ids)
                    return model(ids)
        else:
            imgs = torch.rand(batch, 3, 224, 224, device=DEVICE, dtype=torch.float16)
            model_fp16 = model.half()
            def forward():
                with torch.no_grad():
                    return model_fp16(imgs)

        ms   = cuda_ms(forward)
        vram = torch.cuda.max_memory_allocated() / 1e9
        tput = batch / (ms / 1000)
        lat  = ms / batch

        marker = ""
        if results and tput < results[-1][2] * 0.95:
            marker = "← throughput peak"

        print(f"  {batch:>8}  {ms:>10.2f}  {lat:>11.3f}ms  "
              f"{tput:>12.1f}  {vram:>10.2f}  {marker:>15}")

        results.append((batch, ms, tput, vram))

    except RuntimeError as oom:
        if "out of memory" in str(oom).lower():
            print(f"  {batch:>8}  {'OOM':>10}  {'—':>12}  {'—':>12}  {'—':>10}  ← OOM")
            torch.cuda.empty_cache(); gc.collect()
            break
        raise

    # Exponential sweep: 1 2 4 8 16 32 64 128 …
    batch = batch * 2 if batch < 32 else batch + 32

# ── Recommendations ──────────────────────────────────────────────────────────
if results:
    best_tput = max(results, key=lambda r: r[2])
    best_lat  = results[0]   # smallest batch

    print(f"""
  ╔══════════════════════════════════════════════════════════╗
  ║  RECOMMENDATIONS                                        ║
  ╠══════════════════════════════════════════════════════════╣
  ║  Max throughput : batch={best_tput[0]:<4}  ({best_tput[2]:.0f} samples/sec)  ║
  ║  Min latency    : batch={best_lat[0]:<4}  ({best_lat[1]:.1f} ms)             ║
  ║                                                         ║
  ║  For serving:                                           ║
  ║    Real-time chat  → batch ≤ 4 (latency SLA)           ║
  ║    Offline batch   → batch = {best_tput[0]:<4} (max throughput)  ║
  ║    Mixed workload  → tune per p95 latency target        ║
  ╚══════════════════════════════════════════════════════════╝
    """)
