#!/usr/bin/env python3
"""
slow_dataloader.py  —  Project 2, Step 1: Simulate a DataLoader Bottleneck
============================================================================

HOW TO RUN:
    # Basic run (watch GPU utilisation stay low)
    python slow_dataloader.py

    # With nsys profiling to capture the timeline
    nsys profile --stats=true --output=reports/slow_dataloader \
        python slow_dataloader.py --steps 20


"""

import argparse
import time
import random
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

parser = argparse.ArgumentParser()
parser.add_argument("--steps",        type=int, default=50,  help="Number of training steps")
parser.add_argument("--batch-size",   type=int, default=32,  help="Batch size")
parser.add_argument("--img-size",     type=int, default=224, help="Image size")
parser.add_argument("--sleep-ms",     type=float, default=20, help="Fake I/O delay per image (ms)")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Synthetic dataset with simulated slow I/O ─────────────────────────────────
class SlowSyntheticDataset(Dataset):
    """
    Simulates a real image dataset on slow storage.
    
    In production this would be ImageNet on HDD.
    We use in-memory random tensors but add artificial delays to simulate
    disk seek latency, allowing you to study the bottleneck pattern without
    needing a real HDD dataset.
    """
    
    def __init__(self, size, img_size, sleep_ms):
        self.size    = size
        self.img_size = img_size
        self.sleep_ms = sleep_ms
        
        # ── BOTTLENECK 1: Heavy CPU augmentation ─────────────────────────────
        # This pipeline runs on CPU inside each DataLoader worker.
        # Each image goes through: random crop, flip, colour jitter, normalize.
        # With num_workers=0 this blocks the main thread (and the GPU).
        self.cpu_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),    # CPU: resize + random crop
            transforms.RandomHorizontalFlip(),          # CPU: maybe flip
            transforms.ColorJitter(                     # CPU: random colour changes
                brightness=0.4, contrast=0.4,
                saturation=0.4, hue=0.1
            ),
            transforms.ToTensor(),                     # CPU: convert to float tensor
            transforms.Normalize(                      # CPU: per-channel normalise
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # ── BOTTLENECK 2: Simulated disk I/O latency ─────────────────────────
        # time.sleep() simulates reading from a slow HDD.
        # On a real system, this would be actual disk reads (open → read → decode).
        # With num_workers=0, this blocks the main thread.
        if self.sleep_ms > 0:
            time.sleep(self.sleep_ms / 1000.0)
        
        # Create a random PIL-like image tensor (normally would be PIL.Image from disk)
        img = torch.rand(3, self.img_size, self.img_size)
        
        # Apply transforms
        # In a real dataset you'd do: img = Image.open(path); img = self.cpu_transform(img)
        label = random.randint(0, 999)
        
        return img, label

# ── Create dataset and DataLoader with all bottlenecks enabled ───────────────
dataset = SlowSyntheticDataset(
    size=args.batch_size * args.steps,
    img_size=args.img_size,
    sleep_ms=args.sleep_ms,
)

# ── BOTTLENECK 3: num_workers=0 — single-threaded loading ────────────────────
# The main thread does everything:
#   Load batch → augment → transfer to GPU → train → repeat
# There is NO overlap between GPU training and CPU data loading.
# The GPU sits idle while CPU fetches the next batch.
#
# Compare to: num_workers=8 (8 parallel workers prefetch while GPU trains)
dataloader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    
    # THE MAIN BOTTLENECK — change to 4-8 to fix
    num_workers=0,
    
    # ── BOTTLENECK 4: pin_memory=False ───────────────────────────────────────
    # pin_memory=True "pins" CPU tensor memory so the GPU can DMA from it
    # directly without an intermediate OS copy. Much faster for H2D transfers.
    # Without it: CPU tensor → OS buffer → GPU (2 copies)
    # With it:    CPU pinned tensor → GPU directly (1 copy via DMA)
    pin_memory=False,
    
    # no prefetch_factor because num_workers=0 (workers must be > 0 for this)
    shuffle=True,
)

# ── Model: ResNet50 ──────────────────────────────────────────────────────────
model = models.resnet50(weights=None)
model = model.to(device)
model.train()

criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ── Training loop — with all bottlenecks active ──────────────────────────────
print(f"\nStarting SLOW training loop")
print(f"  batch_size={args.batch_size}, num_workers=0, pin_memory=False")
print(f"  Artificial I/O delay: {args.sleep_ms}ms per image")
print(f"  Steps: {args.steps}")
print(f"\n  Watch GPU utilisation in another terminal:")
print(f"    watch -n 0.5 nvidia-smi")
print(f"  Watch disk/CPU in another terminal:")
print(f"    vmstat 1\n")

step_times     = []
gpu_wait_times = []  # Time from step start to when GPU work begins

total_start = time.perf_counter()

for step, (images, labels) in enumerate(dataloader):
    step_start = time.perf_counter()
    
    # ── BOTTLENECK 5: blocking data transfer ─────────────────────────────────
    # non_blocking=True would let the CPU continue while the GPU DMA transfer
    # happens asynchronously. Without it, the CPU waits for the transfer to finish.
    images = images.to(device)   # SLOW: blocking + no pin_memory
    labels = labels.to(device)
    
    transfer_done = time.perf_counter()
    gpu_wait_ms   = (transfer_done - step_start) * 1000
    gpu_wait_times.append(gpu_wait_ms)
    
    # Forward pass
    optimiser.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass
    loss.backward()
    optimiser.step()
    
    # ── BOTTLENECK 6: synchronous .item() inside training loop ───────────────
    # loss.item() copies the loss scalar from GPU to CPU.
    # This forces a CPU-GPU synchronisation: the CPU BLOCKS until the GPU
    # finishes all queued work before the scalar value can be read.
    # Inside a tight loop this is called EVERY step → constant GPU stalls.
    #
    # Fix: only call .item() every N steps, or use a separate stream,
    #      or accumulate on GPU and sync only at the end.
    loss_value = loss.item()  # BLOCKING GPU SYNC — causes stall every step
    
    step_end  = time.perf_counter()
    step_ms   = (step_end - step_start) * 1000
    step_times.append(step_ms)
    
    if step % 10 == 0 or step < 3:
        gpu_util = "unknown"
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1
            )
            gpu_util = result.stdout.strip() + "%"
        except Exception:
            pass
        
        print(f"  Step {step:3d}/{args.steps}  "
              f"loss={loss_value:.4f}  "
              f"step={step_ms:.0f}ms  "
              f"transfer={gpu_wait_ms:.0f}ms  "
              f"GPU={gpu_util}")
    
    if step >= args.steps - 1:
        break

# ── Summary ───────────────────────────────────────────────────────────────────
total_time = time.perf_counter() - total_start
avg_step   = sum(step_times) / len(step_times)
avg_wait   = sum(gpu_wait_times) / len(gpu_wait_times)

print(f"\n{'='*55}")
print(f"  SLOW BASELINE RESULTS")
print(f"{'='*55}")
print(f"  Total time         : {total_time:.1f}s")
print(f"  Avg step time      : {avg_step:.1f}ms")
print(f"  Avg GPU wait       : {avg_wait:.1f}ms  ← transfer + queue overhead")
print(f"  GPU wait fraction  : {avg_wait/avg_step*100:.1f}% of each step")
print(f"  Steps/sec          : {args.steps/total_time:.1f}")
print(f"\n  Record these for comparison with fast_dataloader.py")
print(f"  Next: diagnose_io.sh to see the bottleneck with system tools")
