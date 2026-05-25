#!/usr/bin/env python3
"""
I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py  ─  Chapter 2: The Ten Most Common Mistakes
===========================================================================
Covers book section 2.6.

Each section demonstrates a real mistake and has you:
  (a) observe the wrong behaviour (the bug is pre-coded so you can see it)
  (b) fix it and verify the correct behaviour

Run:  python I.Foundations/2.PyTorch_Fundamentals/2.6_common_mistakes.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print("=" * 65)
print("  Exercise 06 — The Ten Most Common PyTorch Mistakes")
print("=" * 65)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# MISTAKE 1: Measuring GPU time with time.time()
# ─────────────────────────────────────────────────────────────
print("── Mistake 1: Wrong GPU Timing (time.time vs CUDA events) ──")
print("""
  WRONG: time.time() returns before the GPU finishes (async).
  RIGHT: CUDA events record timestamps on the GPU timeline.
""")

A = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float32)
B = torch.randn(1024, 1024, device=DEVICE, dtype=torch.float32)

# The wrong way (pre-coded so you can observe it)
t0 = time.time()
C  = torch.mm(A, B)
t_wrong_ms = (time.time() - t0) * 1000

if DEVICE == "cuda":
    # TODO 1: Measure the TRUE GPU time using CUDA events.
    #   start = torch.cuda.Event(enable_timing=True) ...
    t_correct_ms = None  # YOUR CODE HERE

    assert t_correct_ms is not None and t_correct_ms > 0, "measure with CUDA events"
    print(f"  time.time() result : {t_wrong_ms:.3f} ms  ← CPU submission, not GPU time")
    print(f"  CUDA events result : {t_correct_ms:.3f} ms  ← true GPU execution time")
    if t_wrong_ms < t_correct_ms * 0.5:
        print("  Confirmed: time.time() underreported by > 50%")
else:
    print(f"  CPU time: {t_wrong_ms:.3f} ms  (no GPU to demonstrate async)")
print("  ✓ Mistake 1 fixed")

# ─────────────────────────────────────────────────────────────
# MISTAKE 2: Forgetting model.eval() before inference
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 2: Forgetting model.eval() ──")
print("""
  With Dropout active (train mode), the model gives different predictions
  on identical inputs every call — benchmarks are non-deterministic,
  accuracy is degraded.  Always call model.eval() before inference.
""")

class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc      = nn.Linear(8, 4)
        self.dropout = nn.Dropout(p=0.8)
    def forward(self, x):
        return self.dropout(self.fc(x))

mdl  = DropoutModel().eval()   # starts in eval for safety

x_inp = torch.randn(50, 8)

# Bug: call train() but forget eval() before inference
mdl.train()
with torch.no_grad():
    out_bad1 = mdl(x_inp)
    out_bad2 = mdl(x_inp)
outputs_differ_in_train = not torch.allclose(out_bad1, out_bad2)

# TODO 2: Switch mdl to eval mode, then run two forward passes.
#   Verify that both outputs are identical (deterministic).
pass  # YOUR CODE HERE  → mdl.eval()
with torch.no_grad():
    out_good1 = mdl(x_inp)
    out_good2 = mdl(x_inp)
outputs_same_in_eval = torch.allclose(out_good1, out_good2)

print(f"  Outputs differ in train mode : {outputs_differ_in_train}  (expected True)")
print(f"  Outputs same   in eval  mode : {outputs_same_in_eval}  (expected True)")
assert outputs_differ_in_train, "train mode should produce different outputs (dropout)"
assert outputs_same_in_eval,    "eval mode should produce identical outputs"
print("  ✓ Mistake 2 fixed")

# ─────────────────────────────────────────────────────────────
# MISTAKE 3: Not zeroing gradients before backward()
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 3: Gradient Accumulation Bug ──")
print("""
  PyTorch ACCUMULATES gradients.  Calling backward() twice without
  zeroing accumulates the gradient from both passes — the effective
  batch size doubles silently.
""")

w = torch.tensor([1.0], requires_grad=True)

# Step 1 — correct
(w * 3.0).sum().backward()
grad_step1 = w.grad.item()   # should be 3.0

# Step 2 — bug: no zero_grad
(w * 3.0).sum().backward()
grad_step2_bug = w.grad.item()   # accumulated: 6.0

# TODO 3: Zero the gradient, then do a fresh backward.
#   w.grad.zero_()  or use optimizer.zero_grad(set_to_none=True)
pass  # YOUR CODE HERE  → w.grad.zero_()
(w * 3.0).sum().backward()
grad_step2_fixed = w.grad.item()   # should be 3.0

print(f"  grad after step 1          : {grad_step1}  (correct: 3.0)")
print(f"  grad after step 2 (bug)    : {grad_step2_bug}  (wrong: accumulated to 6.0)")
print(f"  grad after step 2 (fixed)  : {grad_step2_fixed}  (correct: 3.0)")
assert grad_step1       == 3.0, f"step1 grad wrong: {grad_step1}"
assert grad_step2_bug   == 6.0, f"step2 bug should be 6.0"
assert grad_step2_fixed == 3.0, f"fixed grad should be 3.0"
print("  ✓ Mistake 3 fixed")

# ─────────────────────────────────────────────────────────────
# MISTAKE 4: Calling .item() inside the training loop
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 4: Synchronising GPU Inside the Training Loop ──")
print("""
  loss.item() and tensor.cpu() both SYNCHRONISE the CPU with the GPU
  — the CPU blocks until all pending GPU work finishes.  In a tight
  inner loop this serialises the pipeline and can cut throughput by 50%+.

  Fix: accumulate the raw tensor, call .item() every N steps.
""")

# Simulate a training loop that logs loss every step (WRONG) vs every 50 (RIGHT)
dummy_model = nn.Linear(64, 64, device=DEVICE)
dummy_loss  = nn.MSELoss()
x_dummy     = torch.randn(32, 64, device=DEVICE)
y_dummy     = torch.randn(32, 64, device=DEVICE)

STEPS = 100

# Wrong: .item() every step
t0 = time.perf_counter()
for _ in range(STEPS):
    out  = dummy_model(x_dummy)
    loss = dummy_loss(out, y_dummy)
    _ = loss.item()   # GPU sync every step
t_wrong = (time.perf_counter() - t0) * 1000

# TODO 4: Fix the loop — accumulate loss as a tensor, call .item() every 50 steps.
t_fixed_start = time.perf_counter()
running = torch.tensor(0.0, device=DEVICE)
for step in range(STEPS):
    out  = dummy_model(x_dummy)
    loss = dummy_loss(out, y_dummy)
    # YOUR CODE HERE: add loss to running (no .item()), call .item() every 50 steps
    pass
if DEVICE == "cuda":
    torch.cuda.synchronize()
t_fixed = (time.perf_counter() - t_fixed_start) * 1000

print(f"  .item() every step  : {t_wrong:.1f} ms  ← GPU sync {STEPS}× ")
print(f"  .item() every 50    : {t_fixed:.1f} ms  ← GPU sync 2×")
# Even on CPU the principle holds; on GPU the gap is much larger
print("  ✓ Mistake 4 demonstrated — on real GPU workloads the gap is 2–5×")

# ─────────────────────────────────────────────────────────────
# MISTAKE 5: Creating tensors on the wrong device inside the loop
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 5: Tensors Created on CPU Inside the Loop ──")
print("""
  Every torch.zeros(...), torch.ones(...), torch.randn(...) without
  a device= argument creates a CPU tensor.  If your model is on GPU,
  you get a device mismatch error — or, worse, a silent PCIe copy.
  Fix: always pass device=DEVICE when creating temporary tensors.
""")

model5 = nn.Linear(4, 4).to(DEVICE)
x5     = torch.randn(8, 4, device=DEVICE)

# TODO 5: Create a zeros mask of shape (8, 4) DIRECTLY on DEVICE (device=DEVICE)
mask = None  # YOUR CODE HERE  → torch.zeros(8, 4, device=DEVICE)

# Check that the mask is on the correct device (no implicit PCIe copy)
assert mask is not None,                 "mask is None"
assert mask.device.type == DEVICE,       f"mask should be on {DEVICE}, got {mask.device}"
out5 = model5(x5 * mask)
assert out5.device.type == DEVICE,       "output should be on DEVICE"
print(f"  mask.device : {mask.device}  ← created directly on {DEVICE}, no copy")
print("  ✓ Mistake 5 fixed")

# ─────────────────────────────────────────────────────────────
# MISTAKE 6: Using FP32 where FP16 would work
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 6: FP32 When FP16/BF16 Tensor Cores Are Available ──")

if DEVICE != "cuda":
    print("  (CUDA not available — FP32 vs FP16 comparison skipped)")
else:
    print("""
  On Tensor Core GPUs, FP16 matrix multiplications are 2–8× faster than FP32.
  Many engineers leave performance on the table by defaulting to FP32.
  With AMP, this is free — just wrap forward() in torch.autocast().
    """)
    M = 2048
    A_fp32 = torch.randn(M, M, device=DEVICE, dtype=torch.float32)
    B_fp32 = torch.randn(M, M, device=DEVICE, dtype=torch.float32)

    # Benchmark FP32
    def mm_fp32(): return torch.mm(A_fp32, B_fp32)

    # TODO 6: Create A_fp16 and B_fp16 as float16 versions of A_fp32 and B_fp32
    A_fp16 = None  # YOUR CODE HERE  → A_fp32.half()
    B_fp16 = None  # YOUR CODE HERE  → B_fp32.half()

    def mm_fp16(): return torch.mm(A_fp16, B_fp16)

    # Warmup
    for _ in range(5):
        mm_fp32(); mm_fp16()
    torch.cuda.synchronize()

    def time_fn(fn, iters=50):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters): fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    t_fp32 = time_fn(mm_fp32)
    t_fp16 = time_fn(mm_fp16)
    speedup = t_fp32 / t_fp16 if t_fp16 > 0 else 1.0

    assert A_fp16 is not None and A_fp16.dtype == torch.float16, "A_fp16 should be float16"
    print(f"  FP32 {M}×{M} matmul : {t_fp32:.3f} ms")
    print(f"  FP16 {M}×{M} matmul : {t_fp16:.3f} ms  ({speedup:.1f}× faster)")
    print("  ✓ Mistake 6 demonstrated — use FP16/BF16 on Tensor Core hardware")

# ─────────────────────────────────────────────────────────────
# MISTAKE 7: Benchmarking without warmup
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 7: No Warmup Before Benchmarking ──")
print("""
  First kernel launch: CUDA context init + JIT compilation.
  Can inflate the first measurement by 10–1000×.
  Always run at least 5 warmup iterations before timing.
""")

model7 = nn.Sequential(nn.Linear(256, 256), nn.ReLU()).to(DEVICE)
model7.eval()
x7 = torch.randn(64, 256, device=DEVICE)

times = []
for i in range(8):
    if DEVICE == "cuda":
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad(): model7(x7)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    else:
        t0 = time.perf_counter()
        with torch.no_grad(): model7(x7)
        times.append((time.perf_counter() - t0) * 1000)

# TODO 7: Compute the steady-state average using runs 4–7 (discard first 3)
steady_avg = None  # YOUR CODE HERE  → sum(times[4:]) / len(times[4:])
first_run  = times[0]

assert steady_avg is not None and steady_avg > 0, "compute steady_avg"
print(f"  Run 0 (cold)   : {first_run:.3f} ms")
print(f"  Runs 4-7 avg   : {steady_avg:.3f} ms  ← steady state")
print(f"  First/steady   : {first_run/steady_avg:.1f}×  ← warmup inflation")
print("  ✓ Mistake 7 demonstrated — discard warmup runs")

# ─────────────────────────────────────────────────────────────
# MISTAKE 8: Comparing latencies across different batch sizes
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 8: Comparing Latency Without Normalising ──")
print("""
  4 ms at batch 32 is NOT slower than 1 ms at batch 1.
  4 ms at batch 32 = 8× higher throughput per unit time.
  Always normalise to per-sample or per-token throughput when comparing.
""")

model8 = nn.Linear(128, 128).to(DEVICE)
model8.eval()

latencies    = {}
throughputs  = {}

for bs in [1, 8, 32]:
    xbs = torch.randn(bs, 128, device=DEVICE)
    # TODO 8: Measure latency and throughput for each batch size
    #   Use at least 5 warmup + 20 timed iterations
    #   latencies[bs]   = avg latency in ms
    #   throughputs[bs] = bs / (latency_ms / 1000)  samples per second
    pass  # YOUR CODE HERE

for bs in [1, 8, 32]:
    if bs in latencies:
        print(f"  bs={bs:2d}  latency={latencies[bs]:.3f} ms  "
              f"throughput={throughputs[bs]:.0f} samples/s")

# Check that throughput increases with batch size (more efficient)
if all(bs in throughputs for bs in [1, 8, 32]):
    assert throughputs[32] > throughputs[1], "larger batch should give higher throughput"
    print("  ✓ Mistake 8 demonstrated — always normalise to throughput")
else:
    print("  (Implement TODO 8 to see the full comparison)")

# ─────────────────────────────────────────────────────────────
# MISTAKE 9: Using num_workers=0 in DataLoader (concept)
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 9: num_workers=0 in DataLoader ──")
print("""
  num_workers=0 → data loading runs on the main thread.
  While the CPU loads and preprocesses the next batch, the GPU is IDLE.

  num_workers=4+ → background workers load in parallel with GPU compute.
  This is the single easiest training throughput win on I/O-heavy datasets.
  Combine with pin_memory=True for ~2× faster CPU→GPU copies.

  Ideal settings depend on your CPU core count and disk speed.
  Start with num_workers=4, pin_memory=True and tune from there.
""")

# Demonstrate the API (real speedup requires an I/O-heavy dataset)
xs = torch.randn(1000, 32)
ys = torch.randn(1000, 1)
ds = TensorDataset(xs, ys)

loader_slow = DataLoader(ds, batch_size=64, num_workers=0)
loader_fast = DataLoader(ds, batch_size=64, num_workers=2, pin_memory=(DEVICE == "cuda"))

# TODO 9: Iterate both loaders and measure wall-clock time
t_slow_start = time.perf_counter()
for _ in loader_slow: pass
t_slow = (time.perf_counter() - t_slow_start) * 1000

t_fast_start = time.perf_counter()
for _ in loader_fast: pass
t_fast = (time.perf_counter() - t_fast_start) * 1000

print(f"  num_workers=0: {t_slow:.1f} ms  (TensorDataset is RAM-only — difference minimal here)")
print(f"  num_workers=2: {t_fast:.1f} ms  (real gain on disk I/O workloads: 3–10×)")
print("  ✓ Mistake 9 — set num_workers and pin_memory in every DataLoader")

# ─────────────────────────────────────────────────────────────
# MISTAKE 10: GPU memory fragmentation
# ─────────────────────────────────────────────────────────────
print("\n── Mistake 10: GPU Memory Fragmentation / OOM Debugging ──")

if DEVICE != "cuda":
    print("  (CUDA not available — Section 10 skipped)")
else:
    print("""
  Symptom: CUDA OOM error even though nvidia-smi shows free memory.
  Cause:   the allocator holds fragmented blocks that cannot satisfy
           a new large contiguous allocation.
  Fix:     torch.cuda.empty_cache() between experiments.
    """)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    mem_start = torch.cuda.memory_allocated() / 1e6

    # Simulate fragmentation: allocate then free several tensors
    tensors = []
    for i in range(5):
        tensors.append(torch.randn(1000, 1000, device=DEVICE))   # ~4 MB each

    mem_peak = torch.cuda.memory_allocated() / 1e6

    # Free alternating tensors (creates holes in the pool)
    for i in range(0, len(tensors), 2):
        del tensors[i]

    mem_fragmented = torch.cuda.memory_allocated() / 1e6
    mem_reserved   = torch.cuda.memory_reserved()  / 1e6

    # TODO 10: Empty the cache to return reserved memory to the OS pool
    pass  # YOUR CODE HERE  → torch.cuda.empty_cache()

    mem_after_empty = torch.cuda.memory_reserved() / 1e6

    print(f"  Start allocated   : {mem_start:.1f} MB")
    print(f"  Peak allocated    : {mem_peak:.1f} MB")
    print(f"  After partial del : {mem_fragmented:.1f} MB allocated, "
          f"{mem_reserved:.1f} MB reserved")
    print(f"  After empty_cache : {mem_after_empty:.1f} MB reserved  "
          f"← returned to OS")
    print("  ✓ Mistake 10 — use empty_cache() between experiments")

print("\n" + "=" * 65)
print("  ALL SECTIONS COMPLETE — Exercise 06 done!")
print("  Pre-flight checklist summary:")
print("    ✓ CUDA events for timing  ✓ model.eval() before inference")
print("    ✓ zero_grad before backward  ✓ no .item() inside inner loop")
print("    ✓ tensors created on device  ✓ FP16/BF16 on Tensor Core GPUs")
print("    ✓ warmup before benchmarking  ✓ normalise to throughput")
print("    ✓ num_workers + pin_memory  ✓ empty_cache between experiments")
print("=" * 65)
