#!/usr/bin/env python3
"""
fp16_bf16_bench.py  ─  Phase 2 / Module 5: FP16 vs BF16 vs FP32 Benchmarking
==============================================================================

HOW TO RUN
    python fp16_bf16_bench.py
    python fp16_bf16_bench.py --task matmul
    python fp16_bf16_bench.py --task transformer


"""

import argparse
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="all", choices=["all","matmul","transformer","amp"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

def cuda_ms(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def sep(t): print(f"\n{'═'*60}\n  {t}\n{'─'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1 – Raw matmul throughput per dtype
# ─────────────────────────────────────────────────────────────────────────────
def bench_matmul():
    sep("MATMUL: FP32 vs FP16 vs BF16 Throughput")
    print("""
  Large matmul (2048×2048) isolates dtype effect on tensor core usage.
  FP16 and BF16 activate tensor cores (8× more ops/cycle than FP32 cores).
  """)
    M = 2048
    dtypes = [("FP32",  torch.float32),
              ("FP16",  torch.float16),
              ("BF16",  torch.bfloat16) if HAS_BF16 else None]
    dtypes = [d for d in dtypes if d is not None]

    baseline_ms = None
    print(f"  {'dtype':>6}  {'Time(ms)':>10}  {'TFLOP/s':>10}  {'Speedup':>8}  {'Tensor Core':>12}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}")
    for name, dt in dtypes:
        A = torch.randn(M, M, device=DEVICE, dtype=dt)
        B = torch.randn(M, M, device=DEVICE, dtype=dt)
        ms     = cuda_ms(lambda: torch.mm(A, B))
        tflops = 2*M**3 / (ms/1000) / 1e12
        if baseline_ms is None: baseline_ms = ms
        spd    = baseline_ms / ms
        tc     = "YES" if dt != torch.float32 else "NO"
        print(f"  {name:>6}  {ms:>10.3f}  {tflops:>10.2f}  {spd:>7.2f}x  {tc:>12}")
        del A, B
    print(f"\n  BF16 supported on this GPU: {HAS_BF16}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2 – Transformer forward pass throughput per dtype
# ─────────────────────────────────────────────────────────────────────────────
def bench_transformer():
    sep("TRANSFORMER FORWARD: FP32 vs FP16 vs BF16")
    print("""
  A full transformer layer (attention + FFN) in each dtype.
  Measures real model throughput — more representative than raw matmul.
  """)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
    try:
        from model import TinyTransformer
        has_model = True
    except ImportError:
        has_model = False

    B, T = 8, 64
    dtypes = [("FP32", torch.float32),
              ("FP16", torch.float16),
              ("BF16", torch.bfloat16) if HAS_BF16 else None]
    dtypes = [d for d in dtypes if d is not None]

    baseline_ms = None
    print(f"  {'dtype':>6}  {'Time(ms)':>10}  {'Speedup':>8}  {'Peak VRAM(GB)':>14}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*14}")

    for name, dt in dtypes:
        torch.cuda.reset_peak_memory_stats()
        if has_model:
            mdl = TinyTransformer("small", max_seq=T+1).to(device=DEVICE, dtype=dt).eval()
        else:
            d = 512
            mdl = nn.Sequential(
                nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d)
            ).to(device=DEVICE, dtype=dt).eval()

        ids = torch.randint(0, 50257, (B, T), device=DEVICE)
        def fwd():
            with torch.no_grad():
                if has_model:
                    return mdl(ids)
                return mdl(torch.rand(B, T, 512, device=DEVICE, dtype=dt))

        ms   = cuda_ms(fwd)
        vram = torch.cuda.max_memory_allocated() / 1e9
        if baseline_ms is None: baseline_ms = ms
        spd  = baseline_ms / ms
        print(f"  {name:>6}  {ms:>10.3f}  {spd:>7.2f}x  {vram:>14.3f}")
        del mdl

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3 – Automatic Mixed Precision (AMP) training
# ─────────────────────────────────────────────────────────────────────────────
def bench_amp():
    sep("AMP (Automatic Mixed Precision) Training Step")
    print("""
  AMP automates the FP32↔FP16 switch:
    • Forward pass:  FP16 ops (fast tensor cores)
    • Loss compute:  FP32 accumulation (stable)
    • Backward:      FP16 gradients × GradScaler (prevent underflow)
    • Weight update: FP32 master weights (precision)

  torch.autocast handles the cast automatically based on op type.
  GradScaler multiplies loss by a large factor before backward,
  then divides gradients after unscaling (prevents FP16 underflow).
    """)
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
    try:
        from model import TinyTransformer
        model = TinyTransformer("small", max_seq=65).to(DEVICE)
    except ImportError:
        model = nn.Linear(512, 512).to(DEVICE)

    opt    = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler()        # manages loss scale for FP16
    ids    = torch.randint(0, 50257, (8, 64), device=DEVICE)
    crit   = nn.CrossEntropyLoss()

    # FP32 baseline step
    def step_fp32():
        opt.zero_grad(set_to_none=True)
        if hasattr(model, 'forward') and hasattr(model, 'lm_head'):
            logits, _ = model(ids)
            loss = crit(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
        else:
            x = torch.rand(8, 512, device=DEVICE)
            loss = model(x).mean()
        loss.backward()
        opt.step()

    # AMP step (FP16 forward, FP32 master weights)
    def step_amp():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            if hasattr(model, 'lm_head'):
                logits, _ = model(ids)
                loss = crit(logits.reshape(-1, logits.size(-1)), ids.reshape(-1))
            else:
                x = torch.rand(8, 512, device=DEVICE)
                loss = model(x).mean()
        # scaler.scale multiplies loss to prevent FP16 gradient underflow
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                    # restore true gradient scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)                        # only steps if no inf/nan grads
        scaler.update()                         # adjust scale factor for next step

    ms_fp32 = cuda_ms(step_fp32, warmup=3, iters=10)
    ms_amp  = cuda_ms(step_amp,  warmup=3, iters=10)

    print(f"  FP32 training step : {ms_fp32:.2f} ms")
    print(f"  AMP  training step : {ms_amp:.2f} ms  ({ms_fp32/ms_amp:.2f}× speedup)")
    print(f"  GradScaler current scale: {scaler.get_scale()}")
    print("""
  RULE OF THUMB
    Use AMP for all training unless you see NaN losses.
    If you see NaN: reduce scale init, or switch to BF16 (no scaling needed).
    AMP command:
        scaler = torch.cuda.amp.GradScaler()
        with torch.autocast('cuda', dtype=torch.float16): ...
    """)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  fp16_bf16_bench.py  ─  Phase 2 Module 5")
    print(f"  Device: {DEVICE}   BF16 support: {HAS_BF16}")
    print(f"{'='*60}")

    d = {"matmul": bench_matmul, "transformer": bench_transformer, "amp": bench_amp}
    if args.task == "all":
        for fn in d.values(): fn()
    else:
        d[args.task]()
