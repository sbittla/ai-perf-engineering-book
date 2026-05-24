#!/usr/bin/env python3
"""
slow_training.py  —  Project 4, Step 1: Script With Common Python-Side Mistakes
==================================================================================
"""

import argparse
import time
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--steps",      type=int, default=100)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--log-file",   default="logs/slow_training_log.txt")
args = parser.parse_args()

os.makedirs("logs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nSLOW Training Script — All bottlenecks intentionally enabled")
print(f"Device: {device}")
print(f"Profile with: py-spy record -o slow_profile.svg -- python slow_training.py")
print()

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet18(weights=None).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

# ── BOTTLENECK A: CPU augmentation pipeline ───────────────────────────────────
# These transforms run on CPU inside the DataLoader / manually below.
# torchvision transforms v1 works on PIL Images (CPU only).
# Every image: convert to PIL → apply transforms → convert back to tensor.
# With 32 images per batch, this is 32 × (crop + flip + jitter + normalize).
cpu_aug = transforms.Compose([
    transforms.ToPILImage(),           # tensor → PIL (CPU memory copy)
    transforms.RandomResizedCrop(224), # CPU: random crop
    transforms.RandomHorizontalFlip(), # CPU: random flip
    transforms.ColorJitter(            # CPU: colour jitter
        brightness=0.3, contrast=0.3, saturation=0.3
    ),
    transforms.ToTensor(),             # PIL → tensor (CPU memory copy)
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def make_batch_with_cpu_augmentation(batch_size):
    """
    Create a batch by applying augmentation to each image individually on CPU.
    This is the WRONG way — augmentation should be vectorised on GPU.
    The loop here means Python function call overhead × batch_size.
    """
    images = []
    for _ in range(batch_size):
        # Each call to cpu_aug: PIL conversion, crop, flip, jitter, normalise
        img_tensor = torch.rand(3, 256, 256)     # simulates loaded image
        augmented  = cpu_aug(img_tensor)          # BOTTLENECK: CPU per-image
        images.append(augmented)
    
    # torch.stack allocates a new tensor and copies all images into it
    return torch.stack(images)  # (B, C, H, W)

# ── BOTTLENECK D: log file opened every step ─────────────────────────────────
# Open + write + close on every step = multiple syscalls per iteration
# Also: file.write acquires the GIL; other threads can't run during write
log_file_handle = open(args.log_file, "w")

# ── BOTTLENECK E: Python list accumulation ────────────────────────────────────
# These lists grow unboundedly — Python must resize the underlying array
# periodically, causing memory allocation and garbage collection pauses
all_losses    = []      # Will hold args.steps floats
all_step_times = []

print(f"Starting training for {args.steps} steps...")
print(f"Watch CPU load in: htop or top")
print(f"Watch GPU util in: watch -n 0.5 nvidia-smi")
print()

total_start = time.perf_counter()

for step in range(args.steps):
    step_start = time.perf_counter()
    
    # ── BOTTLENECK A: CPU augmentation per step ───────────────────────────────
    # This runs on the main thread, blocking everything else
    images = make_batch_with_cpu_augmentation(args.batch_size)
    labels = torch.randint(0, 1000, (args.batch_size,))
    
    # ── BOTTLENECK C: blocking transfer (no non_blocking) ────────────────────
    # .to(device) blocks: CPU waits for H2D DMA to complete before returning.
    # GPU is ready and waiting but can't start until CPU gets back control.
    images = images.to(device)    # BLOCKING — CPU stalls here
    labels = labels.to(device)    # BLOCKING — second stall
    
    # Forward + backward
    optimiser.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimiser.step()
    
    # ── BOTTLENECK B: synchronous .item() every step ──────────────────────────
    # .item() forces CPU-GPU synchronisation:
    #   1. CPU submits "copy loss scalar to CPU" to GPU command queue
    #   2. CPU BLOCKS until GPU finishes ALL pending work and executes the copy
    #   3. Only then does Python get the float value
    # This inserts a full GPU barrier on EVERY single training step.
    # On RTX 4060 this can add 0.5–2ms per step.
    loss_value = loss.item()    # GPU BARRIER — stalls every step
    
    step_ms = (time.perf_counter() - step_start) * 1000
    
    # ── BOTTLENECK D: file I/O every step ─────────────────────────────────────
    # Write to log on EVERY step = disk I/O in the hot loop
    # Better: batch log writes, use async I/O, or log every N steps
    log_file_handle.write(f"step={step} loss={loss_value:.6f} time={step_ms:.2f}ms\n")
    log_file_handle.flush()   # flush() forces OS to write to disk immediately
    
    # ── BOTTLENECK E: unbounded list growth ───────────────────────────────────
    # Appending to lists that grow to args.steps items — minor but measurable
    # at scale. Prefer: pre-allocated numpy arrays or deque with maxlen
    all_losses.append(loss_value)
    all_step_times.append(step_ms)
    
    if step % 20 == 0:
        avg_loss = sum(all_losses[-20:]) / min(20, len(all_losses))
        avg_time = sum(all_step_times[-20:]) / min(20, len(all_step_times))
        print(f"  Step {step:4d}/{args.steps}  loss={avg_loss:.4f}  "
              f"step_time={avg_time:.0f}ms")

log_file_handle.close()

total_time = time.perf_counter() - total_start
avg_step   = sum(all_step_times) / len(all_step_times)

print(f"\n{'='*50}")
print(f"  SLOW TRAINING RESULTS")
print(f"{'='*50}")
print(f"  Total time     : {total_time:.1f}s")
print(f"  Avg step time  : {avg_step:.1f}ms")
print(f"  Steps/sec      : {args.steps/total_time:.1f}")
print(f"\n  Record these numbers, then run fast_training.py")
print(f"  Then: bash diff_flamegraph.sh to see what changed")
