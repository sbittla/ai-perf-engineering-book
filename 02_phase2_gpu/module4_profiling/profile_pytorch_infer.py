#!/usr/bin/env python3
"""
profile_pytorch_infer.py  ─  Phase 2 / Module 4: Profile PyTorch Inference
============================================================================

HOW TO RUN
    # Run all profiling modes:
    python profile_pytorch_infer.py

    # Then wrap the whole thing with nsys for the system timeline:
    nsys profile --trace=cuda,nvtx python profile_pytorch_infer.py

    # Quick bottleneck report:
    python -m torch.utils.bottleneck profile_pytorch_infer.py


"""

import os, sys, argparse
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="gpt2",  help="HF model name")
parser.add_argument("--seq",    type=int, default=64)
parser.add_argument("--batch",  type=int, default=4)
parser.add_argument("--outdir", default="prof_output")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {args.model} …")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(DEVICE).eval()
except ImportError:
    print("transformers not installed; using TinyTransformer fallback")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'capstone2_phase2to5', 'shared'))
    from model import TinyTransformer
    model = TinyTransformer("small").to(DEVICE).eval()
    tokenizer = None

# Build input
if tokenizer:
    text = "The performance of large language models depends on" * 3
    enc  = tokenizer(text, return_tensors="pt",
                     max_length=args.seq, truncation=True)
    ids  = enc["input_ids"].to(DEVICE).expand(args.batch, -1)
else:
    ids = torch.randint(0, 50257, (args.batch, args.seq), device=DEVICE)

print(f"Input shape: {ids.shape}   device: {DEVICE}\n")

# ── Warmup ─────────────────────────────────────────────────────────────────────
print("Warmup …")
with torch.no_grad():
    for _ in range(3):
        if tokenizer:
            _ = model.generate(ids, max_new_tokens=10, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        else:
            _ = model(ids)
torch.cuda.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – torch.profiler table (sorted by CUDA time)
# ─────────────────────────────────────────────────────────────────────────────
print("="*60)
print("  SECTION 1 – torch.profiler:  top ops by CUDA time")
print("="*60)
print("""
  WHAT TO LOOK FOR
    Self CUDA %  – fraction of total GPU time this op owns (excludes callees)
    CUDA total   – wall time on GPU including all sub-ops
    Calls        – how many times this op fired
    Input shapes – helps identify which layer the op belongs to

  SLOW OP PATTERNS
    aten::copy_  with large size  → H2D transfer (consider pin_memory)
    aten::linear with small batch → memory-BW bound (need bigger batch)
    aten::softmax                 → fuse with FlashAttention
""")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with torch.no_grad():
        if tokenizer:
            out = model.generate(ids, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        else:
            out, _ = model(ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – Export Chrome trace
# ─────────────────────────────────────────────────────────────────────────────
chrome_path = os.path.join(args.outdir, "chrome_trace.json")
prof.export_chrome_trace(chrome_path)
print(f"\n✓ Chrome trace → {chrome_path}")
print(  "  Open: chrome://tracing  →  Load  →  select the file")
print(  "  Zoom into GPU rows to see individual kernel durations\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – torch.profiler with schedule + TensorBoard
# ─────────────────────────────────────────────────────────────────────────────
print("="*60)
print("  SECTION 3 – Schedule-based profiling (skip/warmup/active)")
print("="*60)
print("""
  schedule(wait=1, warmup=1, active=3):
    step 0: profiler INACTIVE  (skip startup overhead)
    step 1: profiler WARMING   (profiler running, results discarded)
    steps 2-4: profiler ACTIVE (these steps are captured)

  This gives clean data without first-call JIT cost contaminating results.
""")

tb_dir = os.path.join(args.outdir, "tb_trace")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=tensorboard_trace_handler(tb_dir),
    record_shapes=True,
    with_stack=True,
) as prof2:
    for step in range(5):
        # Add NVTX marker: visible as named band in nsys timeline
        torch.cuda.nvtx.range_push(f"infer_step_{step}")
        with torch.no_grad():
            if tokenizer:
                _ = model.generate(ids, max_new_tokens=10, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
            else:
                _ = model(ids)
        torch.cuda.nvtx.range_pop()
        prof2.step()

print(f"✓ TensorBoard trace → {tb_dir}/")
print(f"  Run: tensorboard --logdir {tb_dir}")
print(f"  Open: http://localhost:6006  →  PyTorch Profiler tab\n")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – GPU memory timeline
# ─────────────────────────────────────────────────────────────────────────────
print("="*60)
print("  SECTION 4 – GPU Memory After Inference")
print("="*60)
if DEVICE == "cuda":
    print(f"  Allocated : {torch.cuda.memory_allocated()/1e9:.3f} GB")
    print(f"  Reserved  : {torch.cuda.memory_reserved()/1e9:.3f} GB")
    print(f"  Peak alloc: {torch.cuda.max_memory_allocated()/1e9:.3f} GB")

print(f"""
  DIAGNOSE MEMORY LEAKS
    If 'allocated' grows step-over-step, you have a leak:
      - Storing tensors with .item() but keeping the tensor alive
      - Keeping references to loss / activations outside of with torch.no_grad()
      - KV cache not being freed between requests

  COMMANDS
    # Full allocator dump (very verbose):
    print(torch.cuda.memory_summary())

    # Watch live with nvidia-smi:
    watch -n 0.5 nvidia-smi
""")

print("✓  profile_pytorch_infer.py complete.")
print(f"   Outputs in: {args.outdir}/")
