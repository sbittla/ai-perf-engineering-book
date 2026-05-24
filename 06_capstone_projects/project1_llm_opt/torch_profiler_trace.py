#!/usr/bin/env python3
"""
torch_profiler_trace.py  —  Project 1, Step 4: PyTorch Built-in Profiler
==========================================================================

HOW TO RUN:
    python torch_profiler_trace.py --model gpt2 --tokens 50
    
    Then open trace: 
    1. Chrome trace: open chrome://tracing → load reports/chrome_trace.json
    2. TensorBoard: tensorboard --logdir reports/tb_trace

NVTX MARKERS (bonus):
    Adding torch.cuda.nvtx.range_push/pop() labels sections so they
    appear as named bands in Nsight Systems timeline — very useful.


"""

import argparse
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="gpt2")
parser.add_argument("--tokens", type=int, default=50)
parser.add_argument("--outdir", default="reports")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16
).to(device)
model.eval()

prompt = "The key challenge in machine learning is"
inputs  = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)

# ── WARMUP — run once outside profiler ───────────────────────────────────────
# CRITICAL: Always warm up before profiling.
# First run triggers CUDA JIT compilation (cuBLAS auto-tuning, Triton compilation).
# These one-time costs would inflate profiling results if included.
print("Warming up (1 run outside profiler)...")
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
torch.cuda.synchronize()
print("Warmup done.\n")

# ── PROFILE with torch.profiler ──────────────────────────────────────────────
print("Running profiler (3 steps)...")

# [See book for detailed explanation]
schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)

# tensorboard_trace_handler saves a Chrome-compatible .pt.trace.json file that
# can be opened in both TensorBoard and chrome://tracing.  The trace can only
# be exported once, so we use this single handler instead of also calling
# export_chrome_trace() manually after the context exits.
tb_dir = os.path.join(args.outdir, "tb_trace")
chrome_path = os.path.join(args.outdir, "chrome_trace.json")

with profile(
    activities=[
        ProfilerActivity.CPU,   # Record CPU ops: Python, C++ ATen ops
        ProfilerActivity.CUDA,  # Record CUDA kernels on the GPU
    ],
    schedule=schedule,
    on_trace_ready=tensorboard_trace_handler(tb_dir),
    record_shapes=True,    # Record tensor shapes — helps identify large ops
    profile_memory=True,   # Track memory allocation/deallocation per op
    with_stack=True,       # Include Python call stack (larger file, more context)
    with_flops=True,       # Estimate FLOPs for matmul and conv ops
) as prof:
    
    for step in range(5):  # wait(1) + warmup(1) + active(3) = 5 total
        
        # NVTX range markers: these appear as coloured bands in Nsight Systems.
        # Even without nsys, they add structure to the TensorBoard trace.
        torch.cuda.nvtx.range_push(f"step_{step}_generate")
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        torch.cuda.nvtx.range_pop()
        
        # prof.step() tells the profiler to advance the schedule
        # IMPORTANT: must be called at the end of each profiling step
        prof.step()

# ── PRINT TABLE — top ops by CUDA time ───────────────────────────────────────
print("\n" + "="*70)
print("  TOP 20 OPS BY CUDA TIME")
print("="*70)

# key_averages() aggregates repeated calls to the same op
# sort_by='cuda_time_total' puts the biggest GPU time consumers first
# row_limit=20 shows top 20 rows
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# ── PRINT TABLE — top ops by CPU time ────────────────────────────────────────
print("\n" + "="*70)
print("  TOP 10 OPS BY CPU TIME (these may be blocking GPU)")
print("="*70)
print(prof.key_averages().table(
    sort_by="cpu_time_total",
    row_limit=10
))

# ── PRINT TABLE — memory allocations ────────────────────────────────────────
print("\n" + "="*70)
print("  TOP 10 OPS BY MEMORY ALLOCATION")
print("="*70)
print(prof.key_averages().table(
    sort_by="self_cuda_memory_usage",
    row_limit=10
))

print(f"\n✓ Traces saved: {tb_dir}/  (*.pt.trace.json)")
print(f"  → TensorBoard : tensorboard --logdir {tb_dir}")
print(f"                  then http://localhost:6006 → PyTorch Profiler tab")
print(f"  → Chrome      : open chrome://tracing → Load any *.pt.trace.json file")

# ── Quick torch.utils.bottleneck hint ────────────────────────────────────────
print("\n" + "="*70)
print("  TIP: For a quick one-command bottleneck report, run:")
print(f"  python -m torch.utils.bottleneck baseline_inference.py --model {args.model}")
print("  This combines cProfile (Python) + autograd profiler (CUDA) automatically.")
print("="*70)
