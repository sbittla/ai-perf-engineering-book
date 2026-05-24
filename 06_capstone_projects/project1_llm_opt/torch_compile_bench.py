#!/usr/bin/env python3
"""
torch_compile_bench.py  —  Project 1, Step 8: torch.compile Optimisation
==========================================================================

HOW TO RUN:
    python torch_compile_bench.py --model gpt2 --tokens 100
    python torch_compile_bench.py --model gpt2 --tokens 100 --mode max-autotune


"""

import argparse
import time
import torch
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM

parser = argparse.ArgumentParser()
parser.add_argument("--model",  default="gpt2")
parser.add_argument("--tokens", type=int, default=100)
parser.add_argument("--runs",   type=int, default=5)
parser.add_argument("--mode",   default="all",
                    help="Compilation mode: all | default | reduce-overhead | max-autotune")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_name):
    """Load a fresh copy of the model. We reload for each mode to ensure
    we're not measuring carry-over effects from previous compilations."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()
    return tokenizer, model

def benchmark_mode(model_name, compile_mode, num_runs, max_new_tokens):
    """
    Run inference in a given compile mode and return timing stats.
    Returns dict with: mode, avg_tps, p50_lat, p99_lat, compile_time_s
    """
    print(f"\n  Loading model for mode='{compile_mode}'...")
    tokenizer, model = load_model(model_name)
    
    prompt = "The transformer architecture was introduced in"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    
    compile_time_s = 0.0
    
    if compile_mode != "eager":
        print(f"  Compiling with mode='{compile_mode}'...")
        print(f"  (This may take 1–5 minutes for max-autotune)")
        
        compile_start = time.perf_counter()
        
        # torch.compile returns a compiled callable with the same interface.
        # The actual compilation is DEFERRED — it happens on the first forward pass.
        # fullgraph=True forces the entire model to compile as one graph.
        # If fullgraph fails (graph breaks), set it to False.
        model = torch.compile(
            model,
            mode=compile_mode,       # Compilation strategy (see docstring)
            fullgraph=False,         # True = error on graph breaks; False = partial compile
            dynamic=False,           # False = optimise for fixed shapes (faster)
        )
        
        # TRIGGER COMPILATION by running a forward pass
        # This is where the actual JIT compilation happens
        print(f"  Triggering compilation (first forward pass)...")
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=10, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        
        compile_time_s = time.perf_counter() - compile_start
        print(f"  Compilation took: {compile_time_s:.1f}s")
        
        # WARM UP COMPILED MODEL (2 more runs)
        # After compilation, the first few runs may still be slower due to
        # CUDA graph capture and autotuning
        for _ in range(2):
            with torch.no_grad():
                _ = model.generate(input_ids, max_new_tokens=20, do_sample=False,
                                   pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        print(f"  Warmup complete.")
    
    else:
        # Eager mode warmup
        with torch.no_grad():
            _ = model.generate(input_ids, max_new_tokens=20, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
    
    # ── Timed benchmark runs ──────────────────────────────────────────────────
    latencies_ms = []
    
    for run in range(num_runs):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        end_evt.record()
        torch.cuda.synchronize()
        
        lat_ms = start_evt.elapsed_time(end_evt)
        latencies_ms.append(lat_ms)
    
    new_tokens = output.shape[1] - input_ids.shape[1]
    avg_tps = new_tokens / (statistics.mean(latencies_ms) / 1000)
    sorted_lats = sorted(latencies_ms)
    
    # Clean up GPU memory before loading next model
    del model
    torch.cuda.empty_cache()
    
    return {
        "mode":         compile_mode,
        "avg_tps":      avg_tps,
        "p50_lat_ms":   sorted_lats[len(sorted_lats)//2],
        "p99_lat_ms":   sorted_lats[-1],
        "compile_s":    compile_time_s,
        "mem_peak_gb":  torch.cuda.max_memory_allocated(0) / 1e9,
    }

# ── Run selected modes ────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  torch.compile Benchmark — {args.model}")
print(f"{'='*65}")

modes_to_run = (
    ["eager", "default", "reduce-overhead", "max-autotune"]
    if args.mode == "all"
    else ["eager", args.mode]
)

results = []
for mode in modes_to_run:
    print(f"\n── Mode: {mode} {'─'*(40-len(mode))}")
    r = benchmark_mode(args.model, mode, args.runs, args.tokens)
    results.append(r)
    print(f"  Avg throughput : {r['avg_tps']:.1f} tok/s")
    print(f"  P50 latency    : {r['p50_lat_ms']:.1f} ms")
    print(f"  P99 latency    : {r['p99_lat_ms']:.1f} ms")
    if r['compile_s'] > 0:
        print(f"  Compile time   : {r['compile_s']:.1f} s")

# ── Comparison table ──────────────────────────────────────────────────────────
print(f"\n\n{'='*65}")
print(f"  COMPARISON TABLE")
print(f"{'='*65}")
print(f"  {'Mode':<22} {'Tok/s':>8} {'P50 (ms)':>10} {'P99 (ms)':>10} {'Speedup':>9} {'Compile(s)':>11}")
print(f"  {'-'*22} {'-'*8} {'-'*10} {'-'*10} {'-'*9} {'-'*11}")

baseline_tps = results[0]["avg_tps"]
for r in results:
    speedup = r["avg_tps"] / baseline_tps
    compile_str = f"{r['compile_s']:.0f}s" if r['compile_s'] > 0 else "—"
    marker = " ← baseline" if r["mode"] == "eager" else ""
    print(f"  {r['mode']:<22} {r['avg_tps']:>8.1f} {r['p50_lat_ms']:>10.1f} {r['p99_lat_ms']:>10.1f} {speedup:>8.2f}x {compile_str:>11}{marker}")

print(f"\n  HOW TO INTERPRET:")
print(f"  Speedup >1.0x = faster than eager baseline")
print(f"  max-autotune often gives 1.2–2.0x on transformer models")
print(f"  reduce-overhead gives best results for fixed batch/sequence shapes")
