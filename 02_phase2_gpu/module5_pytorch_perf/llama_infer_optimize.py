#!/usr/bin/env python3
"""
llama_infer_optimize.py  ─  Phase 2 / Module 5: Llama Inference Optimization Ladder
=====================================================================================

HOW TO RUN
    python llama_infer_optimize.py
    python llama_infer_optimize.py --model medium   # larger model
    python llama_infer_optimize.py --tokens 200     # more tokens


"""

import argparse, os, sys, time, statistics
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="small",
                    choices=["tiny","small","medium"])
parser.add_argument("--tokens", type=int, default=100)
parser.add_argument("--batch",  type=int, default=1)
args = parser.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
try:
    from model import TinyTransformer
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_BF16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()

def cuda_event_ms(fn, warmup=3, iters=5):
    """CUDA-event timing over iters runs, returns avg ms."""
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

def tps_from_ms(ms, n_new_tokens):
    return n_new_tokens / (ms / 1000)

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  llama_infer_optimize.py  ─  Phase 2 Module 5")
print(f"  Model: {args.model}   Tokens: {args.tokens}   Device: {DEVICE}")
print(f"{'='*65}\n")

if not HAS_MODEL:
    print("  TinyTransformer not found. Ensure model.py is accessible.")
    exit(1)

prompt = torch.randint(0, 50257, (args.batch, 32), device=DEVICE)
results = []

def run_step(label, model, prompt, tokens, note=""):
    """Run generate, measure tok/s, print result."""
    model.eval()
    # Warmup
    with torch.no_grad():
        model.generate(prompt, max_new_tokens=10, temperature=0)
    torch.cuda.synchronize()

    # Timed
    ms = cuda_event_ms(
        lambda: model.generate(prompt, max_new_tokens=tokens, temperature=0),
        warmup=2, iters=3
    )
    tps  = tps_from_ms(ms, tokens)
    vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0

    baseline_tps = results[0][1] if results else tps
    speedup = tps / baseline_tps

    marker = "← baseline" if not results else f"{speedup:.2f}×"
    print(f"  {label:<28}  {tps:>8.1f} tok/s  {ms:>8.1f}ms  "
          f"{vram:.2f}GB  {marker}")
    if note:
        print(f"  {'':28}  {note}")

    results.append((label, tps, ms, vram, speedup))
    return model

print(f"  {'Step':<28}  {'Tok/s':>8}  {'Time(ms)':>9}  {'VRAM':>5}  {'Speedup':>8}")
print(f"  {'─'*28}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*8}")

# ── STEP 0: Baseline – FP32 eager ─────────────────────────────────────────────
model = TinyTransformer(args.model, max_seq=args.tokens + 33).to(DEVICE)
run_step("0: FP32 eager", model, prompt, args.tokens,
         note="No tensor cores. Pure FP32 CUDA cores.")

# ── STEP 1: FP16 ──────────────────────────────────────────────────────────────
del model; torch.cuda.empty_cache()
model = TinyTransformer(args.model, max_seq=args.tokens + 33).to(DEVICE).half()
run_step("1: FP16 (.half())", model, prompt, args.tokens,
         note="Tensor cores active. ~2x memory savings.")

# ── STEP 2: BF16 (if supported) ───────────────────────────────────────────────
if HAS_BF16:
    del model; torch.cuda.empty_cache()
    model = TinyTransformer(args.model, max_seq=args.tokens+33).to(DEVICE, dtype=torch.bfloat16)
    run_step("2: BF16 (.bfloat16())", model, prompt, args.tokens,
             note="Same speed as FP16, wider range (no overflow).")
else:
    print(f"  {'2: BF16':<28}  {'SKIPPED':>8}  (GPU doesn't support BF16)")
    model = TinyTransformer(args.model, max_seq=args.tokens+33).to(DEVICE).half()

# ── STEP 3: torch.compile (default mode) ──────────────────────────────────────
print(f"\n  Compiling model (torch.compile default) … ", end="", flush=True)
t_compile = time.perf_counter()
compiled_model = torch.compile(model, mode="default", fullgraph=False)
# Trigger compilation with one forward pass
with torch.no_grad():
    compiled_model.generate(prompt, max_new_tokens=5, temperature=0)
torch.cuda.synchronize()
print(f"done ({time.perf_counter()-t_compile:.1f}s)\n")

run_step("3: +torch.compile(default)", compiled_model, prompt, args.tokens,
         note="Kernel fusion. First call triggers JIT compilation.")

# ── STEP 4: torch.compile (max-autotune) ──────────────────────────────────────
print(f"\n  Recompiling with max-autotune … (may take 60s+) ", end="", flush=True)
del compiled_model; torch.cuda.empty_cache()
model_at = TinyTransformer(args.model, max_seq=args.tokens+33).to(DEVICE).half()
compiled_at = torch.compile(model_at, mode="max-autotune", fullgraph=False)
with torch.no_grad():
    compiled_at.generate(prompt, max_new_tokens=5, temperature=0)
torch.cuda.synchronize()
print("done\n")

run_step("4: +max-autotune", compiled_at, prompt, args.tokens,
         note="Benchmarks kernel implementations. Best for fixed shapes.")

# ── STEP 5: Pinned prompt + non_blocking transfer ─────────────────────────────
# (Shows transfer optimisation when prompt comes from CPU)
cpu_prompt = torch.randint(0, 50257, (args.batch, 32))
pin_prompt  = cpu_prompt.pin_memory()

def gen_with_transfer():
    gpu_p = pin_prompt.to(DEVICE, non_blocking=True)
    return compiled_at.generate(gpu_p, max_new_tokens=args.tokens, temperature=0)

ms_transfer = cuda_event_ms(gen_with_transfer, warmup=2, iters=3)
tps_t = tps_from_ms(ms_transfer, args.tokens)
speedup_t = tps_t / results[0][1]
print(f"  {'5: +pin_memory transfer':<28}  {tps_t:>8.1f} tok/s  "
      f"{ms_transfer:>8.1f}ms  —     {speedup_t:.2f}×")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  OPTIMISATION SUMMARY")
print(f"{'='*65}")
baseline_tps = results[0][1]
for label, tps, ms, vram, _ in results:
    bar_width = int(tps / baseline_tps * 30)
    bar = "█" * min(bar_width, 40)
    print(f"  {label:<28}  {bar:<40}  {tps:.0f} tok/s  ({tps/baseline_tps:.2f}×)")

print(f"""
  NEXT STEPS TO PUSH FURTHER
    • Use vLLM (PagedAttention + continuous batching) for serving
    • Add INT8/INT4 quantization (see quantization_bench.py)
    • Profile the remaining bottleneck: nsys profile python llama_infer_optimize.py
    • ncu on the attention kernel: is it still memory-bound?
""")
