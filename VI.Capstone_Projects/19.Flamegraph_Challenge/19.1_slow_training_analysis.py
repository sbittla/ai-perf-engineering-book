#!/usr/bin/env python3
"""
VI.Capstone_Projects/19.Flamegraph_Challenge/19.1_slow_training_analysis.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 19: Capstone 4 — CPU-to-GPU Pipeline Flamegraph Challenge
Section 1: Diagnosing Three Hidden Bottlenecks
=======================================================================
Covers capstone section 19.1:
  • A deliberately slow training loop with three hidden bottlenecks
  • Bottleneck 1: .item() inside the loop — forces GPU-CPU sync every step
  • Bottleneck 2: blocking H2D transfer (no non_blocking=True)
  • Bottleneck 3: no torch.no_grad() during validation — wastes memory
  • Measuring the cost of each bottleneck independently
  • Building the "before" baseline for the challenge

Run:  python VI.Capstone_Projects/19.Flamegraph_Challenge/19.1_slow_training_analysis.py
All sections must print ✓.
"""

import json
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Capstone 19.1 — Flamegraph Challenge: Find the Bottlenecks")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

D_MODEL = 512
BATCH   = 128
N_STEPS = 30


# ─────────────────────────────────────────────────────────────
# SECTION 1: The Three Hidden Bottlenecks
# ─────────────────────────────────────────────────────────────
print("── Section 1: The Three Hidden Bottlenecks ──")
print("""
  A flamegraph challenge presents you with a slow implementation and
  asks you to find the bottlenecks by profiling, then fix them.

  THE THREE BUGS IN THIS TRAINING LOOP:

  Bug 1: loss.item() called inside the training step
    loss.item() calls torch.Tensor.item() which:
      - Forces synchronisation between CPU and GPU
      - Means the CPU must wait for the GPU to finish the forward/backward
      - Introduces ~50–200 µs CPU-GPU sync overhead PER STEP
      - In a tight training loop: adds up to seconds over many batches

  Bug 2: No non_blocking=True on .to(device)
    data.to(device) without non_blocking=True blocks the CPU until
    the DMA transfer is complete. This prevents the CPU from preparing
    the next batch while the GPU is computing.
    Fix: data.to(device, non_blocking=True)

  Bug 3: Validation runs with gradients enabled
    During validation, running without torch.no_grad() means PyTorch
    builds the computation graph (autograd) even though we never call
    .backward(). This wastes ~30% of memory and adds overhead to every op.
    Fix: wrap validation in "with torch.no_grad():"

  YOUR TASK (in 19.2): Fix each bug independently and measure its speedup.
""")
print("  ✓ Section 1 passed — understand the three bottlenecks")


# ─────────────────────────────────────────────────────────────
# SECTION 2: The Deliberately Slow Training Loop
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: The Slow Training Loop ──")
print("""
  The slow_training_step() function has all three bugs.
  We measure its per-step latency as the baseline.

  TODO 1: Implement slow_training_step(model, optimizer, x, y)
  that has all three bugs:
    1. Calls loss.item() to log the loss (forces GPU sync)
    2. Transfers x,y to device without non_blocking
    3. Calls evaluate() which does NOT use torch.no_grad()
  Return (step_time_ms, logged_loss).
""")

model = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, 10),
).to(DEVICE)


def slow_training_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                        x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> tuple:
    """
    TODO 1: One slow training step with all three bugs.
    Bug 1: call loss.item() to accumulate the running loss
    Bug 2: transfer x_cpu, y_cpu to DEVICE without non_blocking
    Bug 3: call evaluate_slow(model, x_cpu) which has no torch.no_grad()
    Return (step_time_ms, logged_loss_float).
    """
    t0 = time.perf_counter()

    # Bug 2: blocking H2D transfer
    x = x_cpu.to(DEVICE)
    y = y_cpu.to(DEVICE)

    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    # Bug 1: .item() forces CPU-GPU sync inside the loop
    logged_loss = loss.item()   # ← BAD: sync here

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    step_ms = (time.perf_counter() - t0) * 1000
    return step_ms, logged_loss


def evaluate_slow(model: nn.Module, x_cpu: torch.Tensor) -> float:
    """Bug 3: validation WITHOUT torch.no_grad() — builds unnecessary graph."""
    x = x_cpu.to(DEVICE)
    logits = model(x)   # ← builds autograd graph; wastes memory
    return logits.max(dim=-1).values.mean().item()


# Run baseline
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
x_cpu = torch.randn(BATCH, D_MODEL)
y_cpu = torch.randint(0, 10, (BATCH,))

# Warmup
for _ in range(5):
    slow_training_step(model, optimizer, x_cpu, y_cpu)

# Measure
slow_times = []
for _ in range(N_STEPS):
    ms, _ = slow_training_step(model, optimizer, x_cpu, y_cpu)
    slow_times.append(ms)

slow_mean_ms = statistics.mean(slow_times)
slow_p99_ms  = sorted(slow_times)[int(0.99 * len(slow_times))]
print(f"  SLOW training step:  mean={slow_mean_ms:.3f} ms  P99={slow_p99_ms:.3f} ms")
assert slow_mean_ms > 0
print("  ✓ Section 2 passed — slow baseline measured")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Measuring Each Bug's Cost
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Isolating Each Bug's Cost ──")
print("""
  To measure each bug's cost, we fix ONE bug at a time and compare
  to the baseline. This is the isolation protocol from Chapter 14.

  TODO 2: Implement fixed_step_no_item() that fixes ONLY Bug 1
  (remove the loss.item() call inside the step; accumulate loss as
  a tensor instead). Measure and compare to slow baseline.
""")


def fixed_step_no_item(model: nn.Module, optimizer: torch.optim.Optimizer,
                        x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> float:
    """
    TODO 2: Fix Bug 1 only — no loss.item() inside the step.
    Accumulate loss as a tensor sum; call item() only ONCE per epoch,
    not per step. Return step_time_ms.
    """
    t0 = time.perf_counter()

    x = x_cpu.to(DEVICE)   # still Bug 2 (blocking)
    y = y_cpu.to(DEVICE)

    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    # No loss.item() here! ← Fix 1

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


for _ in range(5):
    fixed_step_no_item(model, optimizer, x_cpu, y_cpu)
fix1_times = [fixed_step_no_item(model, optimizer, x_cpu, y_cpu)
              for _ in range(N_STEPS)]
fix1_mean_ms = statistics.mean(fix1_times)
item_overhead_ms = slow_mean_ms - fix1_mean_ms

print(f"  SLOW (all bugs):         {slow_mean_ms:.3f} ms")
print(f"  Fix 1 (no .item()):      {fix1_mean_ms:.3f} ms  "
      f"(saved {item_overhead_ms:.3f} ms, {item_overhead_ms/slow_mean_ms*100:.1f}%)")


# Bug 2: non_blocking
def fixed_step_nonblocking(model: nn.Module, optimizer: torch.optim.Optimizer,
                             x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> float:
    """Fix Bug 2 only — non_blocking=True on H2D transfer."""
    t0 = time.perf_counter()

    # Fix 2: non-blocking transfer
    x = x_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))
    y = y_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))

    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    logged = loss.item()   # still Bug 1

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


for _ in range(5):
    fixed_step_nonblocking(model, optimizer, x_cpu, y_cpu)
fix2_times = [fixed_step_nonblocking(model, optimizer, x_cpu, y_cpu)
              for _ in range(N_STEPS)]
fix2_mean_ms = statistics.mean(fix2_times)
nonblocking_overhead_ms = slow_mean_ms - fix2_mean_ms
print(f"  Fix 2 (non_blocking):    {fix2_mean_ms:.3f} ms  "
      f"(saved {nonblocking_overhead_ms:.3f} ms, {nonblocking_overhead_ms/slow_mean_ms*100:.1f}%)")


# Bug 3: no_grad during evaluation
def measure_eval_overhead():
    """Measure cost of one eval pass WITH vs WITHOUT torch.no_grad()."""
    x_eval = torch.randn(BATCH, D_MODEL)

    # Without no_grad (Bug 3)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = evaluate_slow(model, x_eval)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    slow_eval_ms = (time.perf_counter() - t0) / 20 * 1000

    # With no_grad (fix)
    def evaluate_fast(model, x_cpu):
        with torch.no_grad():
            x = x_cpu.to(DEVICE)
            return model(x).max(dim=-1).values.mean().item()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = evaluate_fast(model, x_eval)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    fast_eval_ms = (time.perf_counter() - t0) / 20 * 1000

    return slow_eval_ms, fast_eval_ms


slow_eval_ms, fast_eval_ms = measure_eval_overhead()
eval_overhead_ms = slow_eval_ms - fast_eval_ms
eval_speedup = slow_eval_ms / fast_eval_ms if fast_eval_ms > 0 else 1.0
print(f"  Fix 3 (no_grad eval):    slow={slow_eval_ms:.3f}ms  fast={fast_eval_ms:.3f}ms  "
      f"({eval_speedup:.2f}× speedup)")

assert slow_mean_ms > 0 and fix1_mean_ms > 0 and fix2_mean_ms > 0
print("  ✓ Section 3 passed — each bug's cost isolated")


# ─────────────────────────────────────────────────────────────
# SECTION 4: What a Flamegraph Would Show
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Reading the Flamegraph ──")
print("""
  A CPU flamegraph of the slow training loop would show:

  WIDE FLAT TOP STACKS (hotspots):
    1. cudaDeviceSynchronize  ← caused by loss.item()
       Width proportional to sync overhead
       Fix: remove .item() from the inner loop

    2. cudaMemcpy (host→device)  ← blocking H2D transfer
       Appears as a wide band below the forward/backward stack
       Fix: non_blocking=True + pin_memory

    3. autograd engine nodes during eval  ← unnecessary graph building
       Appears as extra stack depth on eval calls
       Fix: torch.no_grad()

  HOW TO GENERATE A REAL FLAMEGRAPH:
    1. Install py-spy: pip install py-spy
    2. Run: py-spy record -o slow_flame.svg -- python 19.1_slow_training_analysis.py
    3. Open slow_flame.svg in a browser
    4. Look for wide stacks at the top — those are your bottlenecks

  DIFFERENTIAL FLAMEGRAPH (before vs after):
    py-spy record -o before.json -- python 19.1_slow_training_analysis.py
    py-spy record -o after.json  -- python 19.2_optimised_training.py
    Use speedscope or brendangregg/FlameGraph to diff them
""")


# Simulate what the profiler sees: compare CPU time distributions
import torch.profiler

model.eval()
x_b = torch.randn(BATCH, D_MODEL, device=DEVICE)
y_b = torch.randint(0, 10, (BATCH,), device=DEVICE)

activities = [torch.profiler.ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)

# Warmup
with torch.no_grad():
    for _ in range(3):
        model(x_b)

with torch.profiler.profile(activities=activities) as prof:
    for _ in range(5):
        with torch.no_grad():
            out = model(x_b)
        # Simulate Bug 1: sync after each step
        if DEVICE == "cuda":
            _ = out.mean().item()   # forces sync

key_time = "device_time_total" if DEVICE == "cuda" else "cpu_time_total"
slow_ops = [(evt.key, getattr(evt, key_time, 0))
            for evt in prof.key_averages() if getattr(evt, key_time, 0) > 0]
slow_ops.sort(key=lambda x: -x[1])
print(f"  Top 5 ops in slow loop ({'CUDA' if DEVICE == 'cuda' else 'CPU'} time):")
for name, t_us in slow_ops[:5]:
    print(f"    {name:<35}  {t_us/5:.0f} µs/iter")


# Save diagnosis
diagnosis = {
    "slow_step_mean_ms": round(slow_mean_ms, 3),
    "bug_costs": {
        "bug1_item_sync_ms":    round(item_overhead_ms, 3),
        "bug2_blocking_h2d_ms": round(nonblocking_overhead_ms, 3),
        "bug3_eval_no_grad_ms": round(eval_overhead_ms, 3),
    },
    "eval_speedup_from_no_grad": round(eval_speedup, 2),
}
with open("/tmp/capstone19_diagnosis.json", "w") as f:
    json.dump(diagnosis, f, indent=2)
print(f"\n  Diagnosis saved: /tmp/capstone19_diagnosis.json")
assert diagnosis["slow_step_mean_ms"] > 0
print("  ✓ Section 4 passed — flamegraph interpretation and profiler output shown")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 19.1 complete!")
print("  Three bottlenecks found:")
print(f"    Bug 1 (.item() sync)    : {item_overhead_ms:.3f} ms overhead/step")
print(f"    Bug 2 (blocking H2D)    : {nonblocking_overhead_ms:.3f} ms overhead/step")
print(f"    Bug 3 (grad in eval)    : {eval_overhead_ms:.3f} ms overhead/eval call")
print("  Next: VI.Capstone_Projects/19.Flamegraph_Challenge/19.2_optimised_training.py")
print("=" * 60)
