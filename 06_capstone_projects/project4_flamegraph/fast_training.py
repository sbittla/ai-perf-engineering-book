#!/usr/bin/env python3
"""
fast_training.py  —  Project 4, Step 3: All Bottlenecks Fixed
==============================================================
"""

import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import torchvision.transforms.v2 as T   # v2 supports GPU tensors
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--steps",      type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--log-file",   default="logs/fast_training_log.txt")
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nFAST Training Script — All bottlenecks fixed")
print(f"Device: {device}")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── FIX A: GPU augmentation pipeline ─────────────────────────────────────────
# torchvision.transforms.v2 operates on GPU tensors (unlike v1).
# By moving transforms to GPU, we:
#   1. Eliminate CPU computation in the hot loop
#   2. Allow vectorised (batched) operations instead of per-image loops
#   3. Free CPU DataLoader workers to focus on I/O, not compute
#
# The transform parameters (.to(device)) move any transform weights to GPU.
# RandomResizedCrop, RandomHorizontalFlip operate in parallel on all B images.
gpu_aug = T.Compose([
    T.RandomResizedCrop(224),                           # GPU: vectorised crop
    T.RandomHorizontalFlip(p=0.5),                      # GPU: vectorised flip
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),                 # GPU: vectorised normalise
])

def make_batch_fast(batch_size):
    """
    Create a batch as raw tensors (no per-image CPU augmentation).
    The augmentation happens on GPU after transfer.
    In a real pipeline: DataLoader workers load raw images → GPU augments.
    """
    # Raw tensors: no PIL conversion, no per-image transforms
    images = torch.rand(batch_size, 3, 256, 256, dtype=torch.float32)
    labels = torch.randint(0, 1000, (batch_size,))
    return images, labels

# ── FIX E: Pre-allocated arrays (no growing Python lists) ────────────────────
# We know the final size (args.steps), so allocate once.
# numpy arrays don't reallocate when assigned into by index.
all_losses_np     = np.zeros(args.steps, dtype=np.float32)
all_step_times_np = np.zeros(args.steps, dtype=np.float32)

# ── FIX D: In-memory log buffer ───────────────────────────────────────────────
# Accumulate log lines in a list and flush every N steps.
# N × disk writes → 1 disk write per N steps. Much less I/O pressure.
log_buffer = []
LOG_FLUSH_INTERVAL = 20

# ── FIX B: Loss accumulator on GPU ───────────────────────────────────────────
# Instead of loss.item() (GPU sync) every step, accumulate the loss tensor
# on GPU and call .item() only when logging (every LOG_FLUSH_INTERVAL steps).
# This trades per-step GPU barriers for periodic ones (N× fewer syncs).
loss_accumulator = torch.tensor(0.0, device=device, dtype=torch.float32)
acc_count = 0

print(f"Starting training for {args.steps} steps...")
total_start = time.perf_counter()

for step in range(args.steps):
    step_start = time.perf_counter()
    
    # Create raw batch (no CPU augmentation)
    images, labels = make_batch_fast(args.batch_size)
    
    # ── FIX C: non_blocking H2D transfer ─────────────────────────────────────
    # Returns immediately after enqueuing the DMA transfer.
    # CPU continues to zero_grad() while transfer proceeds.
    # GPU starts the model forward pass as soon as transfer completes.
    # The default CUDA stream handles the sequencing automatically.
    images = images.to(device, non_blocking=True)   # async transfer
    labels = labels.to(device, non_blocking=True)   # async transfer
    
    # ── FIX A: GPU augmentation ───────────────────────────────────────────────
    # Applied AFTER H2D transfer. Operates on the GPU tensor in-place.
    # Vectorised over the entire batch — single kernel call instead of B loops.
    with torch.no_grad():
        images = gpu_aug(images)
    
    # Forward + backward
    # set_to_none=True: faster than zero_grad() — releases gradient memory
    # instead of zeroing it. On next backward, fresh gradients allocated.
    optimiser.zero_grad(set_to_none=True)
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimiser.step()
    
    # ── FIX B: Accumulate loss on GPU; no .item() every step ─────────────────
    # .detach() removes the tensor from the computation graph
    # (prevents accidentally keeping references to past computations = memory leak)
    with torch.no_grad():
        loss_accumulator += loss.detach()
        acc_count += 1
    
    step_ms = (time.perf_counter() - step_start) * 1000
    
    # ── FIX E: Store in pre-allocated numpy array ─────────────────────────────
    # Direct index assignment — no list resizing, no Python object allocation
    all_step_times_np[step] = step_ms
    # Note: loss value not stored per-step (accumulated above)
    
    # ── Logging every N steps ──────────────────────────────────────────────────
    if (step + 1) % LOG_FLUSH_INTERVAL == 0 or step == args.steps - 1:
        
        # ── FIX B: GPU sync only at logging time ─────────────────────────────
        # .item() here is acceptable — we're already doing I/O (print + disk write)
        # so the GPU sync cost is dominated by the I/O anyway.
        avg_loss = (loss_accumulator / acc_count).item()
        loss_accumulator.zero_()   # reset for next window
        acc_count = 0
        
        # ── FIX D: Accumulate log lines in memory ─────────────────────────────
        window_start = max(0, step - LOG_FLUSH_INTERVAL)
        avg_time = all_step_times_np[window_start:step+1].mean()
        
        log_line = (f"steps={window_start}-{step} "
                    f"avg_loss={avg_loss:.6f} "
                    f"avg_step_ms={avg_time:.2f}\n")
        log_buffer.append(log_line)
        
        # ── FIX D: Single disk write for N steps ─────────────────────────────
        # Open once, write all buffered lines, close.
        # This reduces file syscalls from N per N steps to 1 per N steps.
        with open(args.log_file, "a") as f:
            f.writelines(log_buffer)
        log_buffer.clear()  # Free buffered lines
        
        print(f"  Step {step:4d}/{args.steps}  loss={avg_loss:.4f}  "
              f"step_time={avg_time:.0f}ms")

total_time = time.perf_counter() - total_start

print(f"\n{'='*50}")
print(f"  FAST TRAINING RESULTS")
print(f"{'='*50}")
print(f"  Total time     : {total_time:.1f}s")
print(f"  Avg step time  : {all_step_times_np.mean():.1f}ms")
print(f"  P50 step time  : {np.median(all_step_times_np):.1f}ms")
print(f"  P99 step time  : {np.percentile(all_step_times_np, 99):.1f}ms")
print(f"  Steps/sec      : {args.steps/total_time:.1f}")
print(f"\n  Now run: bash diff_flamegraph.sh")
print(f"  to generate before/after flamegraph comparison")
