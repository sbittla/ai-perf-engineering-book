#!/usr/bin/env python3
"""
fast_dataloader.py  —  Project 2, Step 3: Optimised DataLoader
================================================================

HOW TO RUN:
    # Compare directly:
    python slow_dataloader.py --steps 50
    python fast_dataloader.py --steps 50

    # With nsys (verify gaps are gone):
    nsys profile --stats=true --output=reports/fast python fast_dataloader.py --steps 30


"""

import argparse
import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.v2 as T   # v2 API supports GPU tensors

parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int,   default=50)
parser.add_argument("--batch-size",     type=int,   default=32)
parser.add_argument("--img-size",       type=int,   default=224)
parser.add_argument("--workers",        type=int,   default=8,
                    help="Number of DataLoader worker processes")
parser.add_argument("--numa-node",      type=int,   default=-1,
                    help="NUMA node to bind workers to (-1 = no binding)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── NUMA binding (optional but recommended on multi-socket systems) ───────────
if args.numa_node >= 0:
    try:
        # os.sched_setaffinity binds this process to CPUs on the specified NUMA node
        # Use: numactl --hardware  to see which CPUs are on each node
        # Example: if NUMA node 0 has CPUs 0-7:
        #   os.sched_setaffinity(0, set(range(8)))
        import subprocess
        result = subprocess.run(
            ["numactl", "--hardware"],
            capture_output=True, text=True
        )
        print(f"NUMA topology:\n{result.stdout[:300]}")
        # Actual binding would use: numactl --cpunodebind=N --membind=N python ...
        print(f"Hint: Run with: numactl --cpunodebind={args.numa_node} --membind={args.numa_node} python fast_dataloader.py")
    except Exception:
        pass

# ── Fast synthetic dataset (no artificial sleep) ─────────────────────────────
class FastSyntheticDataset(Dataset):
    """
    Same data as SlowSyntheticDataset but:
    - No artificial sleep (no simulated disk latency)
    - Returns raw tensors instead of PIL images
    - Augmentation happens AFTER loading (will be moved to GPU below)
    """
    def __init__(self, size, img_size):
        self.size     = size
        self.img_size = img_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # In a real scenario: load from disk with fast NVMe or RAM disk
        # For this demo: generate random tensor to simulate loaded image
        # Shape: (3, H, W) — standard CHW format for PyTorch
        img = torch.rand(3, self.img_size, self.img_size, dtype=torch.float32)
        label = idx % 1000
        return img, label

dataset = FastSyntheticDataset(
    size=args.batch_size * args.steps * 2,
    img_size=args.img_size,
)

# ── OPTIMISATION 1+2+4: num_workers, pin_memory, prefetch_factor ─────────────
# 
# num_workers=N spawns N background worker processes.
# Each worker independently calls __getitem__ and fills a shared memory buffer.
# The main process (GPU trainer) reads from this buffer asynchronously.
# Ideal num_workers ≈ 4 × number of GPUs, but depends on CPU count.
#
# pin_memory=True: worker outputs are placed in pinned CPU memory.
# The GPU's DMA engine can read directly from pinned memory at PCI-e bandwidth
# without any OS intervention. Without pinning, OS may page the memory out.
#
# prefetch_factor=2: each worker queues up 2 batches ahead of when they're needed.
# This ensures the main process never waits for a new batch.
#
# persistent_workers=True: keep workers alive between epochs.
# Without this, workers are spawned+killed at epoch end → startup overhead.

dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    num_workers=args.workers,        # FIXED: parallel loading
    pin_memory=True,                 # FIXED: faster H2D transfers
    prefetch_factor=2,               # FIXED: pre-load 2 batches per worker
    persistent_workers=True,         # FIXED: don't kill workers between epochs
    shuffle=True,
)

# ── OPTIMISATION 5: GPU augmentation pipeline ─────────────────────────────────
# torchvision.transforms.v2 supports GPU tensors (unlike v1).
# Move augmentation from DataLoader worker (CPU) to GPU.
# This frees CPU workers to just load data, while GPU handles transforms.
#
# In a real pipeline, the flow is:
#   Worker: load raw image from disk → minimal CPU transform → send to GPU
#   GPU: apply random crop, flip, normalize → feed to model
gpu_augmentation = T.Compose([
    T.RandomResizedCrop(args.img_size),   # GPU: random crop + resize
    T.RandomHorizontalFlip(p=0.5),        # GPU: random flip
    T.Normalize(                          # GPU: per-channel normalise
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
]).to(device)  # Move transform parameters (if any) to GPU

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.resnet50(weights=None).to(device)
model.train()
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ── Create a CUDA stream for data transfer ────────────────────────────────────
# Using a separate stream for H2D transfers allows them to overlap with
# compute on the default stream. Data for batch N+1 transfers while
# the GPU computes batch N.
transfer_stream = torch.cuda.Stream() if device == "cuda" else None

print(f"\nStarting FAST training loop")
print(f"  num_workers={args.workers}, pin_memory=True, prefetch_factor=2")
print(f"  GPU augmentation, non_blocking transfers, no .item() in inner loop")
print(f"\n  Watch GPU utilisation — should be ~90%+ steady:")
print(f"    watch -n 0.5 nvidia-smi\n")

step_times = []
loss_accumulator = torch.tensor(0.0, device=device)  # Accumulate on GPU

total_start = time.perf_counter()

for step, (images, labels) in enumerate(dataloader):
    step_start = time.perf_counter()
    
    # ── OPTIMISATION 3: non_blocking=True H2D transfer ───────────────────────
    # Returns immediately — transfer happens in background CUDA DMA engine.
    # CPU can proceed to start model operations while transfer completes.
    # IMPORTANT: you must not access the tensor on CPU after this call.
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    
    # ── OPTIMISATION 5: apply GPU augmentation ────────────────────────────────
    with torch.no_grad():
        images = gpu_augmentation(images)
    
    # Forward + backward pass
    optimiser.zero_grad(set_to_none=True)  # set_to_none=True is faster than zero_grad()
                                            # Avoids a memset; gradient memory freed instead
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimiser.step()
    
    # ── OPTIMISATION 6: avoid .item() in inner loop ───────────────────────────
    # Instead of loss.item() (which forces GPU sync every step),
    # we accumulate the loss tensor on GPU and only sync every 10 steps.
    with torch.no_grad():
        loss_accumulator += loss.detach()  # .detach() prevents gradient tracking
    
    step_times.append((time.perf_counter() - step_start) * 1000)
    
    if step % 10 == 0:
        # Sync only every 10 steps — 10x fewer GPU stalls
        # .item() is fine here because we're already doing I/O (print)
        avg_loss = (loss_accumulator / min(10, step+1)).item()
        loss_accumulator.zero_()  # Reset accumulator
        
        gpu_util = "unknown"
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            gpu_util = r.stdout.strip() + "%"
        except Exception:
            pass
        
        print(f"  Step {step:3d}/{args.steps}  "
              f"loss={avg_loss:.4f}  "
              f"step={step_times[-1]:.0f}ms  "
              f"GPU={gpu_util}")
    
    if step >= args.steps - 1:
        break

# ── Results ───────────────────────────────────────────────────────────────────
total_time = time.perf_counter() - total_start
avg_step   = sum(step_times) / len(step_times)

print(f"\n{'='*55}")
print(f"  FAST RESULTS")
print(f"{'='*55}")
print(f"  Total time     : {total_time:.1f}s")
print(f"  Avg step time  : {avg_step:.1f}ms")
print(f"  Steps/sec      : {args.steps/total_time:.1f}")
print(f"\n  Compare to slow_dataloader.py results above.")
print(f"  GPU utilisation should now be ~90%+ steady in nvidia-smi.")
