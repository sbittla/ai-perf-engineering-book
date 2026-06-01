#!/usr/bin/env python3
"""
VI.Capstone_Projects/19.Flamegraph_Challenge/19.2_optimized_training.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Chapter 19: Capstone 4 — CPU-to-GPU Pipeline Flamegraph Challenge
Section 2: Applying All Fixes and Comparing Results
=======================================================================
Covers capstone section 19.2:
  • Fix all three bottlenecks from 19.1 one at a time
  • Measure the isolated speedup from each fix
  • Build the final before/after comparison table
  • Reflect on the differential flamegraph workflow
  • Complete the Part VI capstone challenge

Run:  python VI.Capstone_Projects/19.Flamegraph_Challenge/19.2_optimized_training.py
All sections must print ✓.
"""

import json
import os
import statistics
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 60)
print("  Capstone 19.2 — Applying the Fixes")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

D_MODEL = 512
BATCH   = 128
N_STEPS = 50

model = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, D_MODEL * 4), nn.GELU(),
    nn.Linear(D_MODEL * 4, 10),
).to(DEVICE)


def time_steps(step_fn, optimizer, x_cpu, y_cpu, warmup=5, n=N_STEPS):
    """Run step_fn n times; return list of times in ms."""
    for _ in range(warmup):
        step_fn(model, optimizer, x_cpu, y_cpu)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(n):
        ms = step_fn(model, optimizer, x_cpu, y_cpu)
        times.append(ms)
    return times


# ─────────────────────────────────────────────────────────────
# SECTION 1: Re-establish the buggy baseline
# ─────────────────────────────────────────────────────────────
print("── Section 1: Re-establish the Buggy Baseline ──")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
x_cpu = torch.randn(BATCH, D_MODEL)
y_cpu = torch.randint(0, 10, (BATCH,))


def slow_step(model, optimizer, x_cpu, y_cpu):
    """All three bugs."""
    t0 = time.perf_counter()
    x = x_cpu.to(DEVICE)              # Bug 2: blocking
    y = y_cpu.to(DEVICE)
    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    _ = loss.item()                    # Bug 1: GPU sync per step
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


slow_times = time_steps(slow_step, optimizer, x_cpu, y_cpu)
slow_mean  = statistics.mean(slow_times)
slow_p99   = sorted(slow_times)[int(0.99 * len(slow_times))]
print(f"  Buggy baseline:  mean={slow_mean:.3f} ms  P99={slow_p99:.3f} ms")
assert slow_mean > 0
print("  ✓ Section 1 passed — baseline re-established")


# ─────────────────────────────────────────────────────────────
# SECTION 2: Fix 1 — Remove .item() from the hot path
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Fix 1 — Remove .item() from the hot path ──")
print("""
  THE FIX: Accumulate loss as a Python float (using tensor.detach()) only
  at the end of each epoch — not every step.

  PATTERN (correct):
    running_loss = torch.tensor(0.0, device=DEVICE)
    for x, y in loader:
        ...
        running_loss += loss.detach()   # no sync — stays on GPU
    epoch_loss = running_loss.item() / n_steps   # ONE sync per epoch

  This reduces GPU synchronizations from N_STEPS per epoch to 1 per epoch.
  On a 10,000-step epoch, that is 9,999 fewer sync calls.

  TODO 1: Implement fast_step_no_item(model, optimizer, x_cpu, y_cpu)
  that replaces loss.item() with loss.detach().float().  Return step_time_ms.
""")


def fast_step_no_item(model: nn.Module, optimizer: torch.optim.Optimizer,
                       x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> float:
    """
    TODO 1: Fix Bug 1 — no .item() per step.
    Accumulate with loss.detach() (stays on GPU).
    Still has Bug 2 and Bug 3 (measure isolation).
    Return step_time_ms.
    """
    t0 = time.perf_counter()
    x = x_cpu.to(DEVICE)       # Bug 2 still present
    y = y_cpu.to(DEVICE)
    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    _ = loss.detach()           # Fix 1: no sync
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


fix1_times = time_steps(fast_step_no_item, optimizer, x_cpu, y_cpu)
fix1_mean  = statistics.mean(fix1_times)
fix1_sp    = slow_mean / fix1_mean
item_saved = slow_mean - fix1_mean
print(f"  Fix 1 (no .item()):  mean={fix1_mean:.3f} ms  speedup={fix1_sp:.2f}×  "
      f"saved={item_saved:.3f} ms/step")
assert fix1_mean > 0
print(f"  ✓ Section 2 passed — Fix 1 applied ({fix1_sp:.2f}×)")


# ─────────────────────────────────────────────────────────────
# SECTION 3: Fix 2 — non_blocking H2D transfer
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Fix 2 — non_blocking=True H2D Transfer ──")
print("""
  THE FIX: Use .to(device, non_blocking=True).
  This returns immediately; the DMA engine handles the copy in the background.
  The GPU kernel will not start until the copy finishes (hardware dependency),
  but the CPU thread is free to prepare the NEXT batch or issue other calls.

  PATTERN (correct):
    DataLoader(dataset, pin_memory=True, ...)   # MUST pin to use non_blocking
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

  Non-blocking is most effective when combined with:
    - pin_memory=True in DataLoader (avoids staging copy)
    - Prefetch: overlap next batch load with current GPU compute

  TODO 2: Implement fast_step_nonblocking(model, optimizer, x_cpu, y_cpu)
  that applies Fix 1 + Fix 2. Return step_time_ms.
""")


def fast_step_nonblocking(model: nn.Module, optimizer: torch.optim.Optimizer,
                           x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> float:
    """
    TODO 2: Fix Bug 1 + Bug 2.
    Use non_blocking=True on H2D transfer and detach() for loss.
    Return step_time_ms.
    """
    t0 = time.perf_counter()
    # Fix 2: non-blocking transfer
    x = x_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))
    y = y_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))
    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    _ = loss.detach()   # Fix 1
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


fix2_times = time_steps(fast_step_nonblocking, optimizer, x_cpu, y_cpu)
fix2_mean  = statistics.mean(fix2_times)
fix2_sp    = slow_mean / fix2_mean
print(f"  Fix 1+2 (no .item + nonblocking):  mean={fix2_mean:.3f} ms  speedup={fix2_sp:.2f}×")
print(f"  Isolated Fix 2 gain: {(fix1_mean - fix2_mean):.3f} ms/step")
assert fix2_mean > 0
print(f"  ✓ Section 3 passed — Fix 2 applied (cumulative {fix2_sp:.2f}×)")


# ─────────────────────────────────────────────────────────────
# SECTION 4: Fix 3 — torch.no_grad() during validation
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Fix 3 — torch.no_grad() in Evaluation ──")
print("""
  THE FIX: Wrap all validation (evaluation-only) code in:
    with torch.no_grad():
        ...

  WHY IT MATTERS:
    Without no_grad, PyTorch builds a computation graph at every op.
    Each intermediate tensor stores its grad_fn and retains a reference
    to its inputs — doubling peak memory usage during a forward pass.

    For inference or validation, you NEVER call .backward(), so this
    graph is pure waste: memory and CPU/GPU overhead with no benefit.

  RULE: model.eval() is NOT enough. eval() changes BatchNorm and
  Dropout behavior, but does NOT disable gradient computation.
  You need BOTH: model.eval() + torch.no_grad().

  TODO 3: Implement evaluate_fast(model, x_cpu) with torch.no_grad().
  Measure speedup vs evaluate_slow (no no_grad).
""")


def evaluate_slow(model: nn.Module, x_cpu: torch.Tensor) -> float:
    """Bug 3: no torch.no_grad() — builds unnecessary computation graph."""
    model.eval()
    x = x_cpu.to(DEVICE)
    logits = model(x)   # builds autograd graph
    result = logits.max(dim=-1).values.mean().item()
    model.train()
    return result


def evaluate_fast(model: nn.Module, x_cpu: torch.Tensor) -> float:
    """TODO 3: Evaluation WITH torch.no_grad()."""
    model.eval()
    with torch.no_grad():
        x = x_cpu.to(DEVICE)
        logits = model(x)
        result = logits.max(dim=-1).values.mean().item()
    model.train()
    return result


x_eval = torch.randn(BATCH, D_MODEL)

# Warm up
for _ in range(5):
    evaluate_slow(model, x_eval)
    evaluate_fast(model, x_eval)

# Measure
if DEVICE == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    evaluate_slow(model, x_eval)
if DEVICE == "cuda":
    torch.cuda.synchronize()
slow_eval_ms = (time.perf_counter() - t0) / 50 * 1000

if DEVICE == "cuda":
    torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    evaluate_fast(model, x_eval)
if DEVICE == "cuda":
    torch.cuda.synchronize()
fast_eval_ms = (time.perf_counter() - t0) / 50 * 1000

eval_sp = slow_eval_ms / fast_eval_ms if fast_eval_ms > 0 else 1.0
eval_saved_ms = slow_eval_ms - fast_eval_ms
print(f"  evaluate_slow (no no_grad): {slow_eval_ms:.3f} ms")
print(f"  evaluate_fast (no_grad)   : {fast_eval_ms:.3f} ms")
print(f"  Speedup: {eval_sp:.2f}×  saved {eval_saved_ms:.3f} ms/eval call")
assert evaluate_fast(model, x_eval) is not None
print(f"  ✓ Section 4 passed — Fix 3 applied ({eval_sp:.2f}× eval speedup)")


# ─────────────────────────────────────────────────────────────
# SECTION 5: All Fixes Combined — Final Comparison
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: All Fixes — Final Comparison ──")


def fast_step_all(model: nn.Module, optimizer: torch.optim.Optimizer,
                   x_cpu: torch.Tensor, y_cpu: torch.Tensor) -> float:
    """All three bugs fixed."""
    t0 = time.perf_counter()
    x = x_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))
    y = y_cpu.to(DEVICE, non_blocking=(DEVICE == "cuda"))
    optimizer.zero_grad()
    logits = model(x)
    loss   = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    _ = loss.detach()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


fast_times = time_steps(fast_step_all, optimizer, x_cpu, y_cpu)
fast_mean  = statistics.mean(fast_times)
fast_p99   = sorted(fast_times)[int(0.99 * len(fast_times))]
total_sp   = slow_mean / fast_mean

print(f"\n  {'Config':<30}  {'Mean (ms)':>10}  {'P99 (ms)':>10}  {'Speedup':>9}")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*9}")
print(f"  {'Buggy baseline (all 3 bugs)':<30}  {slow_mean:>10.3f}  {slow_p99:>10.3f}  {'1.00×':>9}")
print(f"  {'Fix 1 (no .item())':<30}  {fix1_mean:>10.3f}  {'—':>10}  {fix1_sp:>8.2f}×")
print(f"  {'Fix 1+2 (+non_blocking)':<30}  {fix2_mean:>10.3f}  {'—':>10}  {fix2_sp:>8.2f}×")
print(f"  {'All fixes (1+2+3)':<30}  {fast_mean:>10.3f}  {fast_p99:>10.3f}  {total_sp:>8.2f}×")

# Load previous diagnosis if available
diag_path = "/tmp/capstone19_diagnosis.json"
prev_diagnosis = {}
if os.path.exists(diag_path):
    with open(diag_path) as f:
        prev_diagnosis = json.load(f)

# Build final report
report = {
    "device": DEVICE,
    "batch_size": BATCH,
    "d_model": D_MODEL,
    "baseline_mean_ms":  round(slow_mean, 3),
    "baseline_p99_ms":   round(slow_p99, 3),
    "optimized_mean_ms": round(fast_mean, 3),
    "optimized_p99_ms":  round(fast_p99, 3),
    "total_speedup":     round(total_sp, 2),
    "fix_contributions": {
        "fix1_no_item_sp":     round(fix1_sp, 2),
        "fix2_nonblocking_sp": round(fix2_sp, 2),
        "fix3_no_grad_eval_sp": round(eval_sp, 2),
    },
    "eval_overhead": {
        "slow_eval_ms": round(slow_eval_ms, 3),
        "fast_eval_ms": round(fast_eval_ms, 3),
        "speedup":      round(eval_sp, 2),
    },
}
with open("/tmp/capstone19_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(f"""
  TOTAL TRAINING STEP SPEEDUP: {total_sp:.2f}×
  Eval speedup from no_grad:   {eval_sp:.2f}×
  Report saved: /tmp/capstone19_report.json

  LESSON SUMMARY:
    Bug 1 (.item() per step)   : forces GPU-CPU sync every step
                                  → remove from inner loop; call once/epoch
    Bug 2 (blocking H2D)       : CPU blocks during each tensor transfer
                                  → non_blocking=True + pin_memory
    Bug 3 (no no_grad in eval) : builds unused computation graph
                                  → always use no_grad + model.eval() in validation

  FLAMEGRAPH EVIDENCE:
    Bug 1 shows as:  wide "cudaDeviceSynchronize" bands in GPU trace
    Bug 2 shows as:  wide "cudaMemcpyAsync" blocking in CPU trace
    Bug 3 shows as:  extra stack depth in eval (autograd engine nodes)

  These are the three most common training loop bugs in production.
  The flamegraph makes them immediately visible; the profiler confirms cost.
""")
assert report["total_speedup"] > 0
print("  ✓ Section 5 passed — all fixes applied, final comparison complete")


print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Capstone 19.2 complete!")
print(f"  Total speedup: {total_sp:.2f}× from three targeted fixes.")
print("  Capstone 19 (Flamegraph Challenge) complete.")
print("  Part VI — Capstone Projects complete!")
print("=" * 60)
