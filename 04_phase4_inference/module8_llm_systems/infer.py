#!/usr/bin/env python3
"""
infer.py  —  Phase 4, Module 8: LLM Inference Script (Profiling Target)
========================================================================

HOW TO RUN:
    # Basic inference:
    python infer.py --model small --prompt "The future of AI is"

    # Benchmark throughput and latency:
    python infer.py --model small --bench --batch 8 --tokens 100

    # Profile with nsys:
    nsys profile --stats=true --trace=cuda,nvtx --output=reports/infer \\
        python infer.py --model small --tokens 50 --nvtx

    # Profile with ncu (attention kernels only):
    ncu --kernel-name ".*attention.*|.*mm.*" --set full \\
        python infer.py --model tiny --tokens 5


"""

import argparse
import sys
import os
import time
import torch
import torch.nn.functional as F
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
from model import TinyTransformer, get_model

parser = argparse.ArgumentParser(description="LLM Inference — profiling target")
parser.add_argument("--model",     default="small",
                    choices=["tiny","small","medium","large"],
                    help="TinyTransformer size")
parser.add_argument("--prompt",    default="The key insight about transformer models is that",
                    help="Text prompt (tokenised as random IDs for this demo)")
parser.add_argument("--tokens",    type=int, default=100, help="New tokens to generate")
parser.add_argument("--batch",     type=int, default=1,   help="Batch size")
parser.add_argument("--dtype",     default="float16", choices=["float32","float16","bfloat16"])
parser.add_argument("--bench",     action="store_true",  help="Run benchmark (multiple runs + stats)")
parser.add_argument("--runs",      type=int, default=5,  help="Benchmark runs")
parser.add_argument("--nvtx",      action="store_true",  help="Add NVTX markers for nsys")
parser.add_argument("--hf",        action="store_true",  help="Use HuggingFace model instead")
parser.add_argument("--hf-model",  default="gpt2",       help="HuggingFace model name")
args = parser.parse_args()

device    = "cuda" if torch.cuda.is_available() else "cpu"
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
dtype     = dtype_map[args.dtype]

print(f"\n{'='*60}")
print(f"  infer.py — LLM Inference Profiling Target")
print(f"{'='*60}")
print(f"  Device  : {device}")
print(f"  Model   : {args.model}")
print(f"  Dtype   : {args.dtype}")
print(f"  Batch   : {args.batch}")
print(f"  Tokens  : {args.tokens}")
print(f"{'='*60}\n")

# ── Load model ────────────────────────────────────────────────────────────────
if args.hf:
    # Use real HuggingFace model (GPT-2 etc.) for more realistic profiling
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading HuggingFace model: {args.hf_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=dtype, device_map="auto"
    )
    model.eval()
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    input_ids = input_ids.expand(args.batch, -1)   # repeat for batch

else:
    # Use our TinyTransformer for controlled experiments
    model = TinyTransformer(size=args.model, max_seq=512)
    model = model.to(device=device)
    model.eval()

    # Simulate tokenised prompt (random IDs)
    seq_len = 32   # fixed prompt length for benchmarking consistency
    input_ids = torch.randint(0, 50257, (args.batch, seq_len), device=device)

print(f"  Model loaded. Prompt shape: {input_ids.shape}")
if not args.hf:
    mem = TinyTransformer.estimate_memory_gb(args.model, batch=args.batch, seq=args.tokens)
    print(f"  Estimated inference memory: {mem['total_infer_gb']:.3f} GB")


# ── Single inference run (returns detailed timing) ───────────────────────────
def run_inference_once(input_ids):
    """
    Run one complete inference: prefill + decode N tokens.
    Returns dict with: prefill_ms, decode_ms, total_ms, n_tokens
    """
    if args.nvtx:
        torch.cuda.nvtx.range_push("inference_total")

    # ── PREFILL PHASE ─────────────────────────────────────────────────────────
    # Measure time to process the entire input prompt (first forward pass).
    # This populates the KV cache and gives us the first output token.
    prefill_start = torch.cuda.Event(enable_timing=True)
    prefill_end   = torch.cuda.Event(enable_timing=True)

    if args.nvtx:
        torch.cuda.nvtx.range_push("prefill")

    prefill_start.record()

    with torch.no_grad():
        if args.hf:
            # HuggingFace: generate 1 token = prefill only
            prefill_out = model.generate(
                input_ids, max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            # Our model: forward pass returns logits + KV caches
            logits, kv_caches = model(input_ids)

    prefill_end.record()

    if args.nvtx:
        torch.cuda.nvtx.range_pop()  # prefill

    # ── DECODE PHASE ─────────────────────────────────────────────────────────
    # Generate tokens one by one using the KV cache.
    # Each step: one forward pass on a single new token.
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_end   = torch.cuda.Event(enable_timing=True)

    if args.nvtx:
        torch.cuda.nvtx.range_push("decode")

    decode_start.record()

    with torch.no_grad():
        if args.hf:
            output = model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            output = model.generate(
                input_ids,
                max_new_tokens=args.tokens,
                temperature=0.0,
            )

    decode_end.record()

    if args.nvtx:
        torch.cuda.nvtx.range_pop()   # decode
        torch.cuda.nvtx.range_pop()   # inference_total

    torch.cuda.synchronize()

    prefill_ms = prefill_start.elapsed_time(prefill_end)
    decode_ms  = decode_start.elapsed_time(decode_end)
    n_new      = output.shape[1] - input_ids.shape[1]

    return {
        "prefill_ms": prefill_ms,
        "decode_ms":  decode_ms,
        "total_ms":   prefill_ms + decode_ms,
        "n_tokens":   n_new,
        "ttft_ms":    prefill_ms,    # Time to first token = prefill time
        "tps":        n_new / (decode_ms / 1000) if decode_ms > 0 else 0,
    }


# ── Warmup ────────────────────────────────────────────────────────────────────
print(f"\nWarming up (2 runs, excluded from stats)...")
for _ in range(2):
    r = run_inference_once(input_ids)
print(f"  Warmup complete. First run: {r['total_ms']:.1f}ms, {r['tps']:.1f} tok/s\n")

# ── Benchmark ─────────────────────────────────────────────────────────────────
n_runs = args.runs if args.bench else 1
results = []

print(f"Running {'benchmark' if args.bench else 'inference'} ({n_runs} run(s))...")
print(f"\n  {'Run':>5}  {'TTFT (ms)':>10}  {'Decode (ms)':>12}  {'Total (ms)':>11}  {'Tok/s':>8}")
print(f"  {'─'*5}  {'─'*10}  {'─'*12}  {'─'*11}  {'─'*8}")

for i in range(n_runs):
    r = run_inference_once(input_ids)
    results.append(r)
    print(f"  {i:>5}  {r['ttft_ms']:>10.1f}  {r['decode_ms']:>12.1f}  {r['total_ms']:>11.1f}  {r['tps']:>8.1f}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  INFERENCE RESULTS")
print(f"{'='*60}")

ttft_list  = [r['ttft_ms']  for r in results]
total_list = [r['total_ms'] for r in results]
tps_list   = [r['tps']      for r in results]

def p(vals, pct):
    sorted_v = sorted(vals)
    return sorted_v[int(pct * len(sorted_v))]

print(f"  TTFT  avg     : {statistics.mean(ttft_list):.1f}ms")
print(f"  TTFT  P99     : {p(ttft_list, 0.99):.1f}ms")
print(f"  Total avg     : {statistics.mean(total_list):.1f}ms")
print(f"  Total P99     : {p(total_list, 0.99):.1f}ms")
print(f"  Tok/s avg     : {statistics.mean(tps_list):.1f}")
print(f"  Tok/s min     : {min(tps_list):.1f}")

if device == "cuda":
    print(f"  Peak VRAM     : {torch.cuda.max_memory_allocated()/1e9:.3f}GB")

print(f"\n  PREFILL vs DECODE breakdown (last run):")
r = results[-1]
print(f"    Prefill (TTFT)  : {r['prefill_ms']:.1f}ms  ({r['prefill_ms']/r['total_ms']*100:.0f}% of total)")
print(f"    Decode ({r['n_tokens']} toks): {r['decode_ms']:.1f}ms  ({r['decode_ms']/r['total_ms']*100:.0f}% of total)")
print(f"    Per-token       : {r['decode_ms']/max(r['n_tokens'],1):.2f}ms")

print(f"""
  WHAT THIS REVEALS:
    Prefill fraction high → long prompt is the bottleneck
    Per-token time high   → memory-bandwidth bound (check: nvidia-smi + ncu)
    Total latency high    → need batching, quantization, or faster GPU
""")
print(f"  Next: nsys profile --trace=cuda,nvtx python infer.py --nvtx")
print(f"  Then: open nsys-ui reports/infer.nsys-rep to see prefill vs decode")
