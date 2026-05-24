#!/usr/bin/env python3
"""
baseline_inference.py  —  Project 1, Step 1: Establish the Baseline
======================================================================

HOW TO RUN:
    # With a small model for quick testing:
    python baseline_inference.py --model gpt2 --tokens 100 --runs 5

    # With Llama-2-7B (requires HuggingFace access token):
    python baseline_inference.py --model meta-llama/Llama-2-7b-hf --tokens 200 --runs 3

    # Profile with nsys at the same time (see profile_nsys.sh):
    nsys profile --stats=true python baseline_inference.py --model gpt2


"""

import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="LLM inference baseline benchmark")
parser.add_argument("--model",   default="gpt2",         help="HuggingFace model ID")
parser.add_argument("--prompt",  default="The history of machine learning began when",
                    help="Input prompt text")
parser.add_argument("--tokens",  type=int, default=100,  help="Number of new tokens to generate")
parser.add_argument("--runs",    type=int, default=5,    help="Number of benchmark runs (first is warmup)")
parser.add_argument("--dtype",   default="float16",      help="Model dtype: float16 or float32")
args = parser.parse_args()

# ── Device setup ─────────────────────────────────────────────────────────────
# Always verify CUDA is available before assuming it will work
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: CUDA not available. Running on CPU — timings will not represent GPU performance.")

print(f"\n{'='*60}")
print(f"  Baseline Inference Benchmark")
print(f"{'='*60}")
print(f"  Model  : {args.model}")
print(f"  Device : {device}")
if device == "cuda":
    # torch.cuda.get_device_properties gives detailed GPU specs
    props = torch.cuda.get_device_properties(0)
    print(f"  GPU    : {props.name}")
    print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
print(f"  Dtype  : {args.dtype}")
print(f"  Tokens : {args.tokens} new tokens per run")
print(f"  Runs   : {args.runs} (first = warmup, excluded from stats)")
print(f"{'='*60}\n")

# ── Load model and tokenizer ──────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Ensure the tokenizer has a pad token (required for batch inference later)
# Many causal LM tokenizers don't set this by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model with dtype={args.dtype}...")

# torch_dtype controls numerical precision:
#   float32 — full precision, highest memory usage (~28GB for 7B)
#   float16 — half precision, ~2x memory reduction, slight accuracy loss
#   bfloat16 — brain float, better for training, similar memory to float16
dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
torch_dtype = dtype_map.get(args.dtype, torch.float16)

# Load model then move to device. device_map="auto" requires the accelerate
# package; using .to(device) is equivalent on single-GPU setups.
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,  # load weights directly to GPU, skip CPU staging
).to(device)
model.eval()  # eval mode disables dropout and batch norm tracking

# ── Snapshot GPU memory AFTER model load ─────────────────────────────────────
if device == "cuda":
    # memory_allocated() = tensors PyTorch owns
    # memory_reserved()  = total GPU memory the allocator holds (may be larger)
    mem_model_gb = torch.cuda.memory_allocated(0) / 1e9
    mem_reserved_gb = torch.cuda.memory_reserved(0) / 1e9
    print(f"\nMemory after model load:")
    print(f"  Allocated : {mem_model_gb:.2f} GB")
    print(f"  Reserved  : {mem_reserved_gb:.2f} GB")

# ── Tokenise the prompt ───────────────────────────────────────────────────────
print(f"\nPrompt: \"{args.prompt}\"")
inputs = tokenizer(args.prompt, return_tensors="pt")

# Move input tensors to the same device as the model
# non_blocking=True means the CPU doesn't wait for the transfer to complete
# before continuing — lets the CPU do other work while data moves to GPU
input_ids = inputs["input_ids"].to(device, non_blocking=True)
prompt_tokens = input_ids.shape[1]
print(f"Prompt token count: {prompt_tokens}")

# ── Helper: time one generation run ──────────────────────────────────────────
def run_inference(input_ids, max_new_tokens):
    """
    Run one forward pass and return (output_ids, ttft_ms, total_ms).
    
    TTFT = Time to First Token = prefill latency.
    The prefill processes all input tokens in parallel (like a forward pass).
    The decode generates one token at a time (autoregressive).
    These have different bottlenecks:
        Prefill  → compute-bound (lots of tokens processed at once)
        Decode   → memory-bandwidth-bound (KV cache reads dominate)
    """
    # Create CUDA events to timestamp GPU operations precisely
    # enable_timing=True allows elapsed_time() to work
    event_start      = torch.cuda.Event(enable_timing=True)
    event_first_token = torch.cuda.Event(enable_timing=True)
    event_end        = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():  # no_grad disables gradient tracking — not needed for inference
        
        # Record start time on the GPU timeline
        event_start.record()
        
        # ── PREFILL PHASE ────────────────────────────────────────────────────
        # Run just the prefill (forward pass on all input tokens) to measure TTFT.
        # We do this by generating exactly 1 new token.
        # This populates the KV cache with keys/values for all input tokens.
        prefill_output = model.generate(
            input_ids,
            max_new_tokens=1,       # Generate only 1 token = just the prefill
            do_sample=False,        # Greedy decoding — deterministic, no randomness
            pad_token_id=tokenizer.pad_token_id,
        )
        event_first_token.record()  # Timestamp after first token = end of prefill
        
        # ── DECODE PHASE ─────────────────────────────────────────────────────
        # Now generate the remaining tokens.
        # Each step reads the KV cache, computes attention, samples one token.
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        event_end.record()
        
        # synchronize() blocks the CPU until the GPU has finished ALL queued work.
        # Without this, elapsed_time() would return 0 or incorrect values because
        # the GPU is still running asynchronously.
        torch.cuda.synchronize()

    # elapsed_time() returns milliseconds between two recorded GPU events
    ttft_ms  = event_start.elapsed_time(event_first_token)
    total_ms = event_start.elapsed_time(event_end)
    
    return output, ttft_ms, total_ms

# ── Benchmark loop ────────────────────────────────────────────────────────────
print(f"\nRunning {args.runs} inference passes...")
print(f"{'Run':<6} {'TTFT (ms)':<12} {'Total (ms)':<14} {'Tok/s':<10} {'Note'}")
print("-" * 55)

ttft_list  = []
total_list = []
tps_list   = []

for i in range(args.runs):
    output, ttft_ms, total_ms = run_inference(input_ids, args.tokens)
    
    # Tokens generated = output length - input length
    # (generate() returns input + new tokens concatenated)
    new_tokens = output.shape[1] - prompt_tokens
    
    # Tokens per second = number of new tokens / time in seconds
    tps = new_tokens / (total_ms / 1000)
    
    note = "← warmup (excluded)" if i == 0 else ""
    print(f"{i:<6} {ttft_ms:<12.1f} {total_ms:<14.1f} {tps:<10.1f} {note}")
    
    # Exclude warmup run from statistics.
    # REASON: First run includes JIT compilation, kernel loading, and caching
    # that won't repeat on subsequent calls. Warmup gives a pessimistic result.
    if i > 0:
        ttft_list.append(ttft_ms)
        total_list.append(total_ms)
        tps_list.append(tps)

# ── Summary statistics ────────────────────────────────────────────────────────
if tps_list:
    import statistics
    print(f"\n{'='*55}")
    print(f"  BASELINE RESULTS (excluding warmup)")
    print(f"{'='*55}")
    print(f"  Avg TTFT (prefill latency) : {statistics.mean(ttft_list):.1f} ms")
    print(f"  Avg total latency          : {statistics.mean(total_list):.1f} ms")
    print(f"  Avg throughput             : {statistics.mean(tps_list):.1f} tok/s")
    print(f"  Min throughput             : {min(tps_list):.1f} tok/s")
    print(f"  Max throughput             : {max(tps_list):.1f} tok/s")
    
    if device == "cuda":
        mem_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"  Peak GPU memory (inference): {mem_peak_gb:.2f} GB")

# ── Decode and print a sample output ─────────────────────────────────────────
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\nSample output (first run):")
print(f"  {output_text[:200]}...")

print(f"\n✓ Baseline captured. Record these numbers for comparison.")
print(f"  Next: run 'bash profile_nsys.sh' to see the GPU timeline.")
