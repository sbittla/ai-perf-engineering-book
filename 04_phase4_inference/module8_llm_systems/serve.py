#!/usr/bin/env python3
"""
serve.py  —  Phase 4, Module 8: Long-Running Inference Server
=============================================================

HOW TO USE:

    Terminal 1 — Start the server:
        python serve.py --model small

    Terminal 2 — Attach nsys AFTER warmup (snapshot steady state):
        nsys profile --delay=5 --duration=10 \\
            --trace=cuda,osrt --output=reports/serve_snapshot \\
            python serve.py --model small
        # The --delay=5 waits 5 seconds (past warmup) before recording

    Terminal 2 — Attach py-spy to running server:
        py-spy record --pid $(pgrep -f serve.py) --output serve_flame.svg --duration 10

    Terminal 2 — strace to see syscalls during steady-state:
        sudo strace -p $(pgrep -f serve.py) -c -e trace=all &
        sleep 10 && sudo kill %1

    Terminal 2 — perf to see CPU profile during inference:
        sudo perf record -g -p $(pgrep -f serve.py) sleep 10
        perf report --stdio | head -30

    Terminal 3 — Monitor GPU:
        watch -n 0.5 nvidia-smi


"""

import argparse
import sys
import os
import time
import signal
import torch
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared', 'models'))
from model import TinyTransformer

parser = argparse.ArgumentParser(description="Steady-state inference server for live profiling")
parser.add_argument("--model",       default="small", choices=["tiny","small","medium","large"])
parser.add_argument("--batch",       type=int, default=4,   help="Batch size per request")
parser.add_argument("--min-tokens",  type=int, default=20,  help="Min output tokens per request")
parser.add_argument("--max-tokens",  type=int, default=100, help="Max output tokens per request")
parser.add_argument("--sleep-ms",    type=float, default=0, help="ms to sleep between requests (simulates rate limiting)")
parser.add_argument("--max-requests",type=int, default=0,   help="Stop after N requests (0=run forever)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*60}")
print(f"  serve.py — Steady-State Inference Server")
print(f"{'='*60}")
print(f"  Model  : {args.model}")
print(f"  Device : {device}")
print(f"  Batch  : {args.batch}")
print(f"\n  Server running. Attach profilers now:")
print(f"  nsys: nsys profile --delay=5 --duration=10 python serve.py")
print(f"  py-spy: py-spy record --pid $(pgrep -f serve.py) --duration 10 -o serve.svg")
print(f"  nvidia-smi: watch -n 0.5 nvidia-smi")
print(f"\n  Press Ctrl+C to stop.")
print(f"{'='*60}\n")

# ── Load model ────────────────────────────────────────────────────────────────
model = TinyTransformer(size=args.model, max_seq=512)
model = model.to(device)
model.eval()

# ── Stats tracking ────────────────────────────────────────────────────────────
total_requests   = 0
total_tokens     = 0
start_wall       = time.perf_counter()
last_log_time    = start_wall
LOG_INTERVAL_S   = 5.0    # Print stats every 5 seconds

# ── Graceful shutdown ─────────────────────────────────────────────────────────
running = True
def shutdown(sig, frame):
    global running
    running = False
    print(f"\n\nShutting down after {total_requests} requests, {total_tokens} tokens...")
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ── Request simulation ────────────────────────────────────────────────────────
def simulate_request():
    """
    Simulate one inference request.
    - Random prompt length (16–64 tokens) to vary prefill cost
    - Random output length (min_tokens to max_tokens) to vary decode time
    This produces a realistic mix of short and long requests.
    """
    # Variable prompt lengths simulate real chat traffic
    prompt_len  = random.randint(16, 64)
    output_len  = random.randint(args.min_tokens, args.max_tokens)

    prompt = torch.randint(0, 50257, (args.batch, prompt_len), device=device)

    with torch.no_grad():
        output = model.generate(
            prompt,
            max_new_tokens=output_len,
            temperature=0.8,
        )

    return output.shape[1] - prompt_len   # number of new tokens generated

# ── Warmup ────────────────────────────────────────────────────────────────────
print("  Warming up (5 requests)...", flush=True)
for _ in range(5):
    simulate_request()
if device == "cuda":
    torch.cuda.synchronize()
print("  Warmup complete. Steady-state serving started.\n", flush=True)

# ── Main serving loop ─────────────────────────────────────────────────────────
while running:
    t0 = time.perf_counter()

    new_tokens = simulate_request()

    elapsed_ms = (time.perf_counter() - t0) * 1000
    total_requests += 1
    total_tokens   += new_tokens

    # Optional rate limiting (simulates real request arrival rate)
    if args.sleep_ms > 0:
        time.sleep(args.sleep_ms / 1000.0)

    # Periodic stats logging
    now = time.perf_counter()
    if now - last_log_time >= LOG_INTERVAL_S:
        wall_elapsed   = now - start_wall
        rps  = total_requests / wall_elapsed
        tps  = total_tokens   / wall_elapsed
        gpu_util = "?"
        gpu_mem  = "?"

        if device == "cuda":
            try:
                import subprocess
                r = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1
                )
                parts = r.stdout.strip().split(",")
                gpu_util = f"{parts[0].strip()}%"
                gpu_mem  = f"{int(parts[1].strip())/1024:.1f}GB"
            except Exception:
                pass

        print(f"  t={wall_elapsed:6.0f}s  req={total_requests:5d}  "
              f"tok={total_tokens:7d}  RPS={rps:.1f}  TPS={tps:.0f}  "
              f"GPU={gpu_util}  VRAM={gpu_mem}",
              flush=True)

        last_log_time = now

    if args.max_requests > 0 and total_requests >= args.max_requests:
        break

# ── Final summary ─────────────────────────────────────────────────────────────
wall_total = time.perf_counter() - start_wall
print(f"\n{'='*60}")
print(f"  Server stopped after {wall_total:.1f}s")
print(f"  Total requests : {total_requests}")
print(f"  Total tokens   : {total_tokens}")
print(f"  Avg RPS        : {total_requests/wall_total:.1f} requests/sec")
print(f"  Avg TPS        : {total_tokens/wall_total:.0f} tokens/sec")
print(f"{'='*60}")
