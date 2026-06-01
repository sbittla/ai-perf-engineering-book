#!/usr/bin/env python3
"""
5.GPU_Profiling/5.1_nsys_profiling.py  ─  Chapter 5: Nsight Systems Profiling
=======================================================================
Covers book section 5.1:
  • NVTX markers for annotating the nsys timeline
  • Measuring CPU-GPU overlap to detect idle GPU time
  • Computing idle_fraction from wall time vs GPU time
  • Annotating a full training step with NVTX phases

Run:
  python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py
  nsys profile --trace=cuda,nvtx python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py

All sections must print ✓.
"""

import time
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 5.1 — Nsight Systems Profiling")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# HOW TO USE nsys
# ─────────────────────────────────────────────────────────────
print("""
  ── How to use Nsight Systems ──

  Nsight Systems (nsys) captures a full system timeline showing:
    - CPU thread activity and kernel launches
    - CUDA kernel execution on the GPU
    - Memory copy operations (HtoD, DtoH, DtoD)
    - NVTX annotation ranges (your custom labels)

  To profile this script:
    nsys profile --trace=cuda,nvtx \\
        -o /tmp/5.1_nsys_output \\
        python II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.1_nsys_profiling.py

  Then open the .nsys-rep file in the Nsight Systems GUI, or generate
  a text report:
    nsys stats /tmp/5.1_nsys_output.nsys-rep

  Look for: NVTX ranges appearing as colored bands above the GPU kernels.
  If the GPU row shows gaps between your NVTX ranges, the GPU was idle.
""")

# ─────────────────────────────────────────────────────────────
# SECTION 1: NVTX Markers
# ─────────────────────────────────────────────────────────────
print("── Section 1: NVTX Markers ──")
print("""
  NVTX (NVIDIA Tools Extension) adds named ranges to the profiler timeline.
  Range push/pop pairs appear as colored bands labelled with your string.

  Pattern:
    torch.cuda.nvtx.range_push("phase_name")
    ... do work ...
    torch.cuda.nvtx.range_pop()

  On CPU-only machines, nvtx calls are silently no-ops — safe to run.
""")

# Build a 3-layer MLP
mlp = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 10),
).to(DEVICE)

x = torch.randn(64, 256, device=DEVICE)

# Phase: data_prep (already done for you — study the pattern)
torch.cuda.nvtx.range_push("data_prep")
x_ready = x.clone()
torch.cuda.nvtx.range_pop()

# TODO 1: Add NVTX range_push("forward") before the model call
#         and range_pop() after it.
#   YOUR CODE HERE → torch.cuda.nvtx.range_push("forward")
with torch.no_grad():
    output = mlp(x_ready)
#   YOUR CODE HERE → torch.cuda.nvtx.range_pop()

# Phase: loss_backward
torch.cuda.nvtx.range_push("loss_backward")
target = torch.randint(0, 10, (64,), device=DEVICE)
loss_fn = nn.CrossEntropyLoss()
mlp.train()
out2 = mlp(x_ready)
loss = loss_fn(out2, target)
loss.backward()
torch.cuda.nvtx.range_pop()

assert output.shape == (64, 10), \
    f"Expected output shape (64, 10), got {output.shape}"
print(f"  MLP output shape: {output.shape}  ✓")
print("  ✓ Section 1 passed — NVTX markers added; run under nsys to visualise")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Measuring CPU-GPU Overlap
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Measuring CPU-GPU Overlap ──")
print("""
  In a real training loop, each step has:
    1. CPU time: DataLoader (disk I/O, augmentations, batching)
    2. GPU time: forward pass, loss, backward

  If the DataLoader is slow, the GPU idles waiting for the next batch.
  We simulate this with time.sleep(delay) inside a loop.

  We measure:
    wall_ms      = total wall-clock time for N steps (perf_counter)
    gpu_ms_total = total GPU compute time (sum of CUDA event timings)

  idle_fraction = (wall_ms - gpu_ms_total) / wall_ms
""")

mlp.eval()
STEPS = 10
SIMULATED_IO_DELAY = 0.002  # 2ms simulated DataLoader delay per step

gpu_ms_total = 0.0
wall_start = time.perf_counter()

# TODO 2: Implement the 10-step loop.
#   Each step:
#     - time.sleep(SIMULATED_IO_DELAY)  [simulates DataLoader]
#     - Use CUDA events to time the GPU forward pass
#     - Accumulate gpu time into gpu_ms_total
#   Use time.perf_counter() for wall_start / wall_end.

if DEVICE == "cuda":
    for step in range(STEPS):
        time.sleep(SIMULATED_IO_DELAY)  # simulate I/O

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        # YOUR CODE HERE: s.record(); forward pass; e.record(); synchronize; accumulate
        pass  # remove this and implement the above

wall_end = time.perf_counter()
wall_ms  = (wall_end - wall_start) * 1000

assert gpu_ms_total >= 0, "gpu_ms_total must be >= 0"
print(f"  Wall time total  : {wall_ms:.1f} ms")
print(f"  GPU compute total: {gpu_ms_total:.1f} ms")
print("  ✓ Section 2 passed — CPU-GPU timeline measurements collected")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Detecting GPU Idle Time
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Detecting GPU Idle Time ──")
print("""
  idle_fraction measures how much of the total wall time the GPU was NOT
  computing.  A high idle fraction (>20%) means the GPU is starved —
  typically due to:
    - Slow DataLoader (disk/network I/O bound)
    - CPU-side preprocessing bottleneck
    - synchronization barriers (e.g. loss.item() in the inner loop)

  A low idle fraction (<10%) means the GPU is the bottleneck —
  time to profile kernels with ncu (Exercise 5.2).
""")

# TODO 3: Compute idle_fraction = (wall_ms - gpu_ms_total) / wall_ms.
#   Handle the edge case where wall_ms = 0 (set idle_fraction = 0.0).
idle_fraction = None  # YOUR CODE HERE

if idle_fraction is None:
    idle_fraction = 0.0  # fallback for CPU-only

assert 0 <= idle_fraction <= 1, f"idle_fraction must be in [0,1], got {idle_fraction}"
print(f"  Idle fraction: {idle_fraction:.2%}")
if idle_fraction > 0.20:
    print("  GPU is idle >20% of the time → likely I/O bound.")
    print("  Fix: increase DataLoader num_workers, add pin_memory=True,")
    print("       or cache preprocessed data to a fast disk/RAM.")
else:
    print("  GPU is busy >80% of the time → DataLoader is keeping up.")
    print("  If still slow, profile the kernels with ncu.")

print("  ✓ Section 3 passed — idle_fraction computed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Annotating a Full Training Step
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Annotating a Full Training Step ──")
print("""
  A production training step has six clearly distinct phases that should
  appear as separate NVTX ranges in the nsys timeline:

    zero_grad     — clear gradient buffers
    data_to_gpu   — CPU → GPU tensor transfer
    forward       — model forward pass
    loss          — loss function computation
    backward      — gradient computation
    optimizer_step — weight update

  Annotating each phase lets you immediately see in nsys which phase
  is slow, whether the GPU is idle between phases, and whether the
  PCIe transfer overlaps with compute.
""")

# Build a small model for the full training step demo
train_model = nn.Sequential(
    nn.Linear(128, 256), nn.ReLU(),
    nn.Linear(256, 10),
).to(DEVICE)
optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Create a batch on CPU (simulating DataLoader output)
xb_cpu = torch.randn(32, 128)
yb_cpu = torch.randint(0, 10, (32,))

# TODO 4: Add NVTX range_push / range_pop around EACH of the six phases below.
#   The "zero_grad" phase is already done for you as a template.

# Phase 1: zero_grad
torch.cuda.nvtx.range_push("zero_grad")
optimizer.zero_grad(set_to_none=True)
torch.cuda.nvtx.range_pop()

# Phase 2: data_to_gpu
# YOUR CODE HERE → torch.cuda.nvtx.range_push("data_to_gpu")
xb = xb_cpu.to(DEVICE)
yb = yb_cpu.to(DEVICE)
# YOUR CODE HERE → torch.cuda.nvtx.range_pop()

# Phase 3: forward
# YOUR CODE HERE → torch.cuda.nvtx.range_push("forward")
pred = train_model(xb)
# YOUR CODE HERE → torch.cuda.nvtx.range_pop()

# Phase 4: loss
# YOUR CODE HERE → torch.cuda.nvtx.range_push("loss")
loss = criterion(pred, yb)
# YOUR CODE HERE → torch.cuda.nvtx.range_pop()

# Phase 5: backward
# YOUR CODE HERE → torch.cuda.nvtx.range_push("backward")
loss.backward()
# YOUR CODE HERE → torch.cuda.nvtx.range_pop()

# Phase 6: optimizer_step
# YOUR CODE HERE → torch.cuda.nvtx.range_push("optimizer_step")
optimizer.step()
# YOUR CODE HERE → torch.cuda.nvtx.range_pop()

assert loss.item() > 0, f"loss should be > 0, got {loss.item()}"
print(f"  Training step completed.  loss = {loss.item():.4f}")
print("  NVTX annotations added — run under nsys to see the six phases.")

print("  ✓ Section 4 passed — full training step annotated with NVTX")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 5.1 complete!")
print("  You now know how to annotate code with NVTX, measure CPU-GPU")
print("  overlap, compute idle_fraction, and annotate a training step.")
print("  Next: II.GPU_Programming_and_Profiling/5.GPU_Profiling/5.2_ncu_profiling.py")
print("=" * 60)
