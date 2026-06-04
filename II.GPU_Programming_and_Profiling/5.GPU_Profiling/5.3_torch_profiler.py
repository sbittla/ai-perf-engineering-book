#!/usr/bin/env python3
"""
5.GPU_Profiling/5.3_torch_profiler.py  ─  Chapter 5: torch.profiler Deep Dive
=======================================================================
Covers book section 5.3:
  • Basic profiler setup for a multi-layer model
  • The profiler schedule (wait/warmup/active)
  • Chrome trace export for visual inspection
  • Reading Self CUDA % to find the real bottleneck

Run:  python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.3_torch_profiler.py
All sections must print ✓.
"""

import os
import tempfile
import torch
import torch.nn as nn
from torch.profiler import (
    profile,
    ProfilerActivity,
    schedule,
    tensorboard_trace_handler,
)

print("=" * 60)
print("  Exercise 5.3 — torch.profiler Deep Dive")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Basic Profiling
# ─────────────────────────────────────────────────────────────
print("── Section 1: Basic Profiling ──")
print("""
  torch.profiler wraps your code and records the time spent in every
  PyTorch operator on both CPU and GPU.  The key output is the operator
  table sorted by cuda_time_total (or cpu_time_total on CPU-only machines).

  The operator at the top of the table is your first optimization target.
  Look for the Self CUDA column — this is the time spent ONLY in this op,
  not in its callees.  Ops with high total time but low self time are
  wrappers; the real work is in their children.

  activities list:
    CPU-only: [ProfilerActivity.CPU]
    GPU:      [ProfilerActivity.CPU, ProfilerActivity.CUDA]
""")

# Build a 4-layer MLP
model = nn.Sequential(
    nn.Linear(512, 1024), nn.ReLU(),
    nn.Linear(1024, 1024), nn.ReLU(),
    nn.Linear(1024, 512), nn.ReLU(),
    nn.Linear(512, 128),
).to(DEVICE)
model.eval()
x = torch.randn(64, 512, device=DEVICE)

# TODO 1: Build the activities list.
#   Always include ProfilerActivity.CPU.
#   If DEVICE == "cuda", also add ProfilerActivity.CUDA.
activities = [ProfilerActivity.CPU]
if DEVICE == "cuda":
    activities.append(ProfilerActivity.CUDA)

assert activities is not None, "activities list must not be None"
assert len(activities) >= 1,   "activities must have at least one entry"

with profile(activities=activities, record_shapes=True) as prof:
    with torch.no_grad():
        model(x)

sort_key = "cuda_time_total" if DEVICE == "cuda" else "cpu_time_total"
print(f"\n  Top 10 ops by {sort_key}:")
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
print("  ✓ Section 1 passed — profiler ran and printed operator table")

# ─────────────────────────────────────────────────────────────
# SECTION 2: The Profiler Schedule
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: The Profiler Schedule ──")
print("""
  Profiling every step of a long training run is expensive and produces
  enormous trace files.  The profiler schedule lets you:
    wait    N: skip N steps at the start (let things warm up)
    warmup  N: run the profiler but discard these steps
    active  N: record these N steps
    repeat  N: run the whole cycle N times (0 = run once)

  schedule(wait=1, warmup=1, active=3, repeat=1) means:
    Step 0: skip
    Step 1: profiler on but discarded (warmup)
    Steps 2-4: profiled and recorded

  This is the recommended pattern for training loops — it avoids the
  profiler overhead on all but the steps you care about.
""")

TOTAL_STEPS = 5  # 1 wait + 1 warmup + 3 active

# TODO 2: Implement the 5-step loop.
#   Use profile(..., schedule=schedule(wait=1, warmup=1, active=3, repeat=1))
#   Inside the loop: run the model forward pass, then call prof.step().
events_list = []

sched = schedule(wait=1, warmup=1, active=3, repeat=1)
with profile(
    activities=activities,
    schedule=sched,
    record_shapes=False,
) as prof:
    for step in range(TOTAL_STEPS):
        # Forward pass with torch.no_grad() and model(x)
        with torch.no_grad():
            model(x)
        # Advance the profiler schedule
        prof.step()

# Collect events for assertion
if DEVICE == "cuda":
    events_list = prof.events()

if DEVICE == "cuda":
    assert len(events_list) > 0, \
        "prof.events() should not be empty after profiling with schedule"
    print(f"  Profiler collected {len(events_list)} events over {TOTAL_STEPS} steps")
else:
    print("  (CPU fallback: events() may be empty — skipping count assertion)")
print("  ✓ Section 2 passed — schedule-based profiling ran correctly")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Chrome Trace Export
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Chrome Trace Export ──")
print("""
  torch.profiler can export its data in the Chrome Trace Event format,
  readable in:
    chrome://tracing  (built into Chrome/Chromium)
    Perfetto UI       (https://ui.perfetto.dev)
    TensorBoard       (pip install tensorboard)

  The JSON file contains every operator with its start time, duration,
  thread ID, and (on GPU) CUDA stream ID.  It is the most detailed view
  of your model's execution timeline.

  To open in Chrome: navigate to chrome://tracing, click Load, select the JSON.
  To open in TensorBoard: use prof.export_chrome_trace() then:
    tensorboard --logdir /tmp/tb_logs
""")

TRACE_PATH = os.path.join(tempfile.gettempdir(), "pt2_chrome_trace.json")

# Run a fresh profile to get a trace we can export
with profile(activities=activities, record_shapes=True) as prof2:
    for _ in range(3):
        with torch.no_grad():
            model(x)

# TODO 3: Export the Chrome trace to TRACE_PATH.
#   Call prof2.export_chrome_trace(TRACE_PATH)
prof2.export_chrome_trace(TRACE_PATH)

assert os.path.exists(TRACE_PATH), \
    f"Chrome trace file not found at {TRACE_PATH}. Did you call export_chrome_trace()?"
file_size_kb = os.path.getsize(TRACE_PATH) / 1024
print(f"  Chrome trace exported: {TRACE_PATH} ({file_size_kb:.1f} KB)")
print(f"  Open with: chrome://tracing → Load → select {TRACE_PATH}")
print("  ✓ Section 3 passed — Chrome trace exported")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Reading Self CUDA %
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Reading Self CUDA % ──")
print("""
  The profiler reports two time columns per operator:
    CUDA total   : time including all ops called BY this op (inclusive)
    Self CUDA    : time spent ONLY in this specific op (exclusive)

  Example:
    aten::linear   CUDA total = 5ms   Self CUDA = 0.01ms
      aten::mm     CUDA total = 4ms   Self CUDA = 4ms
      aten::addmm  CUDA total = 0.9ms Self CUDA = 0.9ms

  aten::linear's total is 5ms but its self is nearly zero — the real work
  is in aten::mm.  optimizing aten::linear as a whole is impossible; you
  must look at its children.

  Always sort by self_cuda_time_total to find the true bottleneck kernel.
""")

# Build the sorted table
with profile(
    activities=activities,
    record_shapes=True,
) as prof3:
    with torch.no_grad():
        for _ in range(10):
            model(x)

# TODO 4: From prof3.key_averages(), find the entry with the highest
#   self_cuda_time_total (or self_cpu_time_total on CPU).
#   Store it in top_op.
#
#   Hint:
#     avgs = prof3.key_averages()
#     attr = "self_cuda_time_total" if DEVICE == "cuda" else "self_cpu_time_total"
#     top_op = max(avgs, key=lambda e: getattr(e, attr, 0))
avgs = prof3.key_averages()
# PyTorch 2.6+ renamed the per-event CUDA timing attributes to the
# device-agnostic "*_device_time_total"; older releases used "*_cuda_time_total".
attr = "self_device_time_total" if DEVICE == "cuda" else "self_cpu_time_total"
top_op = max(avgs, key=lambda e: getattr(e, attr, 0))

assert top_op is not None, "top_op must not be None — check your key_averages() call"

if DEVICE == "cuda":
    self_ms = top_op.self_device_time_total / 1000  # microseconds → ms
    total_ms = top_op.device_time_total / 1000
    print(f"  Top op by self CUDA time: {top_op.key}")
    print(f"    Self CUDA time  : {self_ms:.3f} ms")
    print(f"    Total CUDA time : {total_ms:.3f} ms")
    print(f"    Self %          : {100*self_ms/total_ms:.1f}%" if total_ms > 0 else "")
else:
    print(f"  Top op by self CPU time: {top_op.key}")
    self_ms = top_op.self_cpu_time_total / 1000
    print(f"    Self CPU time: {self_ms:.3f} ms")

sort_by_self = "self_cuda_time_total" if DEVICE == "cuda" else "self_cpu_time_total"
print(f"\n  Full table sorted by {sort_by_self}:")
print(prof3.key_averages().table(sort_by=sort_by_self, row_limit=8))
print("  ✓ Section 4 passed — Self CUDA time analysis complete")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 5.3 complete!")
print("  You now know how to use the profiler schedule, export Chrome")
print("  traces, and interpret Self CUDA time to find true bottlenecks.")
print("  Next: II.GPU_Programming_and_Profiling/6.PyTorch_Optimization/6.1_precision_and_amp.py")
print("=" * 60)
