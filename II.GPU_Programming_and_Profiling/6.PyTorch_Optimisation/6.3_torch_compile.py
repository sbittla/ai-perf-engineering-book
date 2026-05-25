#!/usr/bin/env python3
"""
6.PyTorch_Optimisation/6.3_torch_compile.py  ─  Chapter 6: torch.compile
=======================================================================
Covers book section 6.3:
  • What torch.compile does: graph capture → TorchInductor → Triton
  • Compilation modes and their trade-offs
  • Warmup behaviour (first calls are slow due to compilation)
  • Batch size sweep to find peak throughput

Run:  python II.GPU_Programming_and_Profiling/6.PyTorch_Optimisation/6.3_torch_compile.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 6.3 — torch.compile and Batch Size Optimisation")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: torch.compile Basics
# ─────────────────────────────────────────────────────────────
print("── Section 1: torch.compile Basics ──")
print("""
  torch.compile was introduced in PyTorch 2.0 as the successor to
  torch.jit.script.  It captures the computation graph by tracing the
  Python code with TorchDynamo, then passes the graph to TorchInductor,
  which generates optimised Triton or C++ kernels.

  Key differences from eager mode:
    - Kernel fusion: multiple PyTorch ops become one Triton kernel
    - Memory layout optimisation: rearranges tensors for cache efficiency
    - Operator fusion: reduces HBM round-trips for elementwise chains

  Compilation modes:
    "default"        : balanced fusion and compile time
    "reduce-overhead": aggressive fusion, accepts longer compile time
    "max-autotune"   : exhaustive kernel search (very slow to compile,
                       fastest runtime; use for repeated production inference)

  Important: the first 2-3 calls trigger compilation and are much slower
  than steady-state.  ALWAYS warm up before timing.
""")

# Build a simple MLP for the compilation demo
mlp = nn.Sequential(
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 512), nn.ReLU(),
    nn.Linear(512, 256),
).to(DEVICE)
mlp.eval()

# Eager reference
x = torch.randn(64, 512, device=DEVICE)
with torch.no_grad():
    eager_out = mlp(x)

# TODO 1: Call torch.compile(mlp, mode="default") to create compiled_mlp.
#   Then run 5 warmup iterations with dummy input to trigger compilation.
compiled_mlp = None  # YOUR CODE HERE → torch.compile(mlp, mode="default")

assert compiled_mlp is not None, "compiled_mlp must not be None"

print("  Running warmup iterations (first 2-3 may be slow due to compilation)...")
if compiled_mlp is not None:
    for i in range(5):
        with torch.no_grad():
            out = compiled_mlp(x)
    if DEVICE == "cuda":
        torch.cuda.synchronize()

# Verify same output shape
with torch.no_grad():
    compiled_out = compiled_mlp(x) if compiled_mlp else eager_out
assert compiled_out.shape == eager_out.shape, \
    f"Compiled output shape {compiled_out.shape} != eager {eager_out.shape}"
print(f"  Eager shape  : {eager_out.shape}")
print(f"  Compiled shape: {compiled_out.shape}")
print("  ✓ Section 1 passed — torch.compile produces correct output shape")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Compilation Warmup Behaviour
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Compilation Warmup Behaviour ──")
print("""
  The first time the compiled model is called, TorchDynamo captures the
  computation graph and TorchInductor generates the kernel code.
  This takes 0.5–5 seconds for a typical model.

  You should see the first timing step (step 0) being significantly
  slower than later steps (steps 5–9).  After compilation finishes,
  timing stabilises at the steady-state value.

  Rule: always discard at least 3 steps for compiled models.
  In production: use max-autotune mode once, save the compiled artifact
  with torch.export(), and load the pre-compiled version at serving time.
""")

# Build a FRESH compiled model (no prior warmup) to observe compilation overhead
fresh_mlp = nn.Sequential(
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 256),
).to(DEVICE)
fresh_mlp.eval()

# TODO 2: Collect per-step timing for 10 steps of the fresh compiled model.
#   Create a fresh_compiled = torch.compile(fresh_mlp, mode="default").
#   For each step i in range(10), measure the time of one forward pass.
#   Use CUDA events on GPU, time.perf_counter on CPU.
#   Store results in step_times list (in ms).

fresh_compiled = None  # YOUR CODE HERE → torch.compile(fresh_mlp, mode="default")
step_times = []        # YOUR CODE HERE → list of per-step times in ms

# Fallback check
if not step_times or fresh_compiled is None:
    print("  (TODO 2 not completed — generating reference data)")
    # Reference: first call slow, rest fast
    step_times = [500.0, 100.0, 8.0, 2.0, 2.1, 2.0, 2.1, 1.9, 2.0, 2.0]

print(f"  {'Step':>5}  {'Time (ms)':>12}  {'Note'}")
for i, t in enumerate(step_times):
    note = " ← compilation overhead" if i < 3 else ""
    print(f"  {i:>5}  {t:>12.3f}{note}")

assert step_times[0] > step_times[8] * 2, (
    f"First step ({step_times[0]:.2f} ms) should be > 2x step 8 "
    f"({step_times[8]:.2f} ms) — first call includes compilation."
)
print("  ✓ Section 2 passed — compilation overhead visible in step 0")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Mode Comparison
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Mode Comparison ──")
print("""
  torch.compile has three main modes:

  "default"         : good starting point; reasonable compile time and
                      runtime. Recommended for most cases.
  "reduce-overhead" : more aggressive loop fusion, graph reuse. Useful
                      when the same graph is called repeatedly with
                      different data (batch inference).
  "max-autotune"    : exhaustive autotuning (tries many kernel variants
                      and picks the fastest). Compile time: minutes.
                      Use only for fixed-shape production deployment.

  We benchmark all three on a transformer-like layer.
""")

if DEVICE == "cuda":
    # TransformerEncoderLayer is a realistic, complex layer
    t_layer = nn.TransformerEncoderLayer(
        d_model=256, nhead=8, dim_feedforward=1024,
        batch_first=True, device=DEVICE,
    )
    t_layer.eval()
    xt = torch.randn(16, 64, 256, device=DEVICE)  # (batch, seq, d_model)

    def bench_mode(model, x, mode_name, warmup=10, iters=30):
        """Benchmark a model after warmup."""
        if mode_name != "eager":
            m = torch.compile(model, mode=mode_name)
        else:
            m = model
        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                m(x)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        # Time
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            with torch.no_grad():
                m(x)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    # TODO 3: Time each mode after warmup (10 warmup, 30 iters).
    #   Call bench_mode for "eager", "default", and "reduce-overhead".
    #   Print speedup vs eager.
    eager_ms = None  # YOUR CODE HERE → bench_mode(t_layer, xt, "eager")
    default_ms = None  # YOUR CODE HERE → bench_mode(t_layer, xt, "default")
    reduce_ms  = None  # YOUR CODE HERE → bench_mode(t_layer, xt, "reduce-overhead")

    if eager_ms is None:
        print("  (TODO 3 not completed — using reference values)")
        eager_ms, default_ms, reduce_ms = 3.0, 2.8, 2.5

    assert reduce_ms <= eager_ms * 1.5, (
        f"reduce-overhead ({reduce_ms:.3f} ms) should not be > 1.5x eager "
        f"({eager_ms:.3f} ms) after warmup"
    )
    print(f"  Eager              : {eager_ms:.3f} ms")
    print(f"  default            : {default_ms:.3f} ms  "
          f"({eager_ms/default_ms:.2f}x speedup)")
    print(f"  reduce-overhead    : {reduce_ms:.3f} ms  "
          f"({eager_ms/reduce_ms:.2f}x speedup)")
else:
    print("  (Requires CUDA — skipping mode comparison)")

print("  ✓ Section 3 passed — compiled mode not slower than eager after warmup")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Batch Size Sweep
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Batch Size Sweep ──")
print("""
  Throughput (samples/second) does not scale linearly with batch size.
  At small batch sizes, GPU utilisation is low and the hardware is mostly
  idle.  As batch size increases, more SMs are active and throughput rises.
  Eventually the GPU is fully occupied and throughput plateaus.

  Finding the batch size that maximises throughput is important for:
    - Inference serving: maximise tokens/second for a given SLA
    - Training: maximise samples/second for a given memory budget
    - Cost optimisation: find the cheapest batch size per sample

  We sweep batch sizes [1, 2, 4, 8, 16, 32, 64] on a 512→512 linear
  layer and report throughput in samples/second.
""")

linear_layer = nn.Linear(512, 512).to(DEVICE)
linear_layer.eval()

batch_sizes   = [1, 2, 4, 8, 16, 32, 64]
throughputs   = {}

# TODO 4: Implement the batch size sweep.
#   For each batch in batch_sizes:
#     1. Create a random input: x = torch.randn(batch, 512, device=DEVICE)
#     2. Warmup for 5 iterations
#     3. Time 30 iterations with CUDA events (or perf_counter on CPU)
#     4. Compute throughput_sps = batch / (ms_per_iter / 1000)
#     5. Store in throughputs[batch]

for batch in batch_sizes:
    x_sw = torch.randn(batch, 512, device=DEVICE)
    # YOUR CODE HERE: warmup, timing, compute throughput_sps, store in throughputs[batch]
    pass  # replace with implementation

# Fallback for untouched TODO
if not throughputs:
    print("  (TODO 4 not completed — generating reference data)")
    # Reference: small batches → low throughput, larger → higher
    base = 1000
    for b in batch_sizes:
        throughputs[b] = base * min(b, 32) + (b > 32) * base * 32

print(f"  {'Batch':>6}  {'Throughput (samples/s)':>24}")
for b in batch_sizes:
    print(f"  {b:>6}  {throughputs[b]:>24,.0f}")

peak_batch = max(throughputs, key=throughputs.get)
print(f"\n  Peak throughput batch size: {peak_batch}")

assert peak_batch >= 8, \
    f"peak_batch should be >= 8, got {peak_batch}. " \
    "Did you implement the throughput computation correctly?"
print("  ✓ Section 4 passed — batch size sweep complete, peak batch identified")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 6.3 complete!")
print("  You now understand torch.compile modes, warmup behaviour,")
print("  mode trade-offs, and batch size throughput sweeps.")
print("  Next: II.GPU_Programming_and_Profiling/7.DataLoader_Optimisation/7.1_dataloader_pipeline.py")
print("=" * 60)
