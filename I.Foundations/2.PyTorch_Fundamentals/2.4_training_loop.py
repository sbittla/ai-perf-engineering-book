#!/usr/bin/env python3
"""
I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py  ─  Chapter 2: The Complete Training Loop
=======================================================================
Covers book sections 2.4:
  • Custom Dataset
  • DataLoader (batch, shuffle, num_workers, pin_memory)
  • Building a model and moving to device
  • The 6-step training loop
  • Validation loop with torch.no_grad()
  • Gradient clipping
  • Automatic Mixed Precision (AMP)
  • Inference

Run:  python I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py
All sections must print ✓.
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("  Exercise 04 — The Complete Training Loop")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Custom Dataset
# ─────────────────────────────────────────────────────────────
print("── Section 1: Custom Dataset ──")
print("""
  Every Dataset must implement __len__ and __getitem__.
  PyTorch's DataLoader calls these to build batches automatically.
  y = sin(x) + noise is a simple regression task for testing.
""")

class SineDataset(Dataset):
    """y = sin(x) + noise — a simple 1-D regression dataset."""

    def __init__(self, n_samples: int, noise: float = 0.1):
        self.x = torch.linspace(-3.14, 3.14, n_samples).unsqueeze(1)   # (N, 1)
        self.y = torch.sin(self.x) + noise * torch.randn_like(self.x)  # (N, 1)

    def __len__(self):
        # TODO 1: Return the number of samples in the dataset
        pass  # YOUR CODE HERE

    def __getitem__(self, idx):
        # TODO 2: Return the (input, target) pair at index idx
        #   → (self.x[idx], self.y[idx])
        pass  # YOUR CODE HERE

ds = SineDataset(200)
sample_x, sample_y = ds[0]
assert len(ds) == 200,           "Dataset length wrong"
assert sample_x.shape == (1,),  f"x shape wrong: {sample_x.shape}"
assert sample_y.shape == (1,),  f"y shape wrong: {sample_y.shape}"
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 2: DataLoader
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: DataLoader ──")
print("""
  DataLoader handles batching, shuffling, and parallel data loading.
  Key arguments:
    num_workers=N  → N background worker processes (0 = synchronous, slow)
    pin_memory=True → allocate CPU tensors in pinned memory for faster GPU copy
    shuffle=True   → randomise sample order each epoch (train only)

  Always: shuffle=True for training, shuffle=False for validation.
""")

train_ds = SineDataset(800)
val_ds   = SineDataset(200)

# TODO 3: Create a DataLoader for train_ds
#   batch_size=32, shuffle=True, num_workers=0
train_loader = None  # YOUR CODE HERE

# TODO 4: Create a DataLoader for val_ds
#   batch_size=64, shuffle=False, num_workers=0
val_loader = None  # YOUR CODE HERE

batch_x, batch_y = next(iter(train_loader))
assert train_loader is not None,        "train_loader is None"
assert val_loader   is not None,        "val_loader is None"
assert batch_x.shape == (32, 1),       f"batch shape wrong: {batch_x.shape}"
print(f"  Batches in train_loader: {len(train_loader)}")
print(f"  Batches in val_loader  : {len(val_loader)}")

# Demonstrate num_workers impact (concept demonstration)
print("\n  DataLoader num_workers timing (SineDataset is tiny — difference is minimal here):")
for nw in [0, 2]:
    loader_timed = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=nw)
    t0 = time.perf_counter()
    for _ in loader_timed:
        pass
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"    num_workers={nw}: {elapsed:.1f} ms  (real datasets: 0 → GPU idle, 4 → overlap)")

print("  ✓ Section 2 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 3: Build the Model
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: Model ──")
print("""
  Build the model in __init__ with fixed architecture, move to device once.
  Never call model.to(device) inside the training loop.
""")

# TODO 5: Build a 3-layer MLP for regression:
#   Linear(1→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)
# TODO 6: Move the model to DEVICE by chaining .to(DEVICE)
model = None  # YOUR CODE HERE

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")
assert model is not None
assert next(model.parameters()).device.type == DEVICE, f"model should be on {DEVICE}"
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Training Loop (6 steps)
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Training Loop ──")
print("""
  Every training step — in this exact order:
    1. Move batch to device
    2. optimizer.zero_grad(set_to_none=True)   ← clears accumulated gradients
    3. pred = model(xb)                         ← forward pass
    4. loss = criterion(pred, yb)               ← compute scalar loss
    5. loss.backward()                          ← backpropagate
    6. optimizer.step()                         ← update weights
""")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 5
history  = {"train_loss": [], "val_loss": []}

for epoch in range(N_EPOCHS):
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        # TODO 7: Move xb and yb to DEVICE
        xb = None  # YOUR CODE HERE  → xb.to(DEVICE)
        yb = None  # YOUR CODE HERE  → yb.to(DEVICE)

        # TODO 8: Zero the gradients (set_to_none=True is slightly faster)
        pass  # YOUR CODE HERE

        # TODO 9: Forward pass — get predictions from model
        pred = None  # YOUR CODE HERE

        # TODO 10: Compute MSE loss
        loss = None  # YOUR CODE HERE

        # TODO 11: Backward pass
        pass  # YOUR CODE HERE

        # TODO 12: Optimizer step
        pass  # YOUR CODE HERE

        train_losses.append(loss.item())   # .item() syncs GPU — OK here (every 100 steps in prod)

    # Validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb   = xb.to(DEVICE)
            yb   = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())

    avg_train = sum(train_losses) / len(train_losses)
    avg_val   = sum(val_losses)   / len(val_losses)
    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    print(f"  Epoch {epoch+1}/{N_EPOCHS}  train={avg_train:.4f}  val={avg_val:.4f}")

assert history["train_loss"][-1] < history["train_loss"][0], \
    "training loss should decrease over epochs"
print("  ✓ Section 4 passed — loss decreased")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Gradient Clipping
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Gradient Clipping ──")
print("""
  Gradient clipping prevents 'exploding gradients' in deep networks and RNNs.
  It caps the global L2 norm of all gradients BEFORE the optimizer step.
  Transformers typically use max_norm=1.0.

  Order: backward() → clip_grad_norm_() → optimizer.step()
""")

model_clip = nn.Sequential(
    nn.Linear(1, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
).to(DEVICE)
opt_clip = torch.optim.Adam(model_clip.parameters(), lr=1e-3)

xb, yb = next(iter(train_loader))
xb, yb = xb.to(DEVICE), yb.to(DEVICE)

opt_clip.zero_grad(set_to_none=True)
pred = model_clip(xb)
loss = criterion(pred, yb)
loss.backward()

# Measure gradient norm before clipping
total_norm_before = sum(
    p.grad.data.norm(2).item() ** 2
    for p in model_clip.parameters() if p.grad is not None
) ** 0.5

# TODO 13: Clip gradients — torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
pass  # YOUR CODE HERE

total_norm_after = sum(
    p.grad.data.norm(2).item() ** 2
    for p in model_clip.parameters() if p.grad is not None
) ** 0.5

opt_clip.step()

print(f"  Gradient norm before clipping: {total_norm_before:.4f}")
print(f"  Gradient norm after  clipping: {total_norm_after:.4f}  (capped at 1.0)")
assert total_norm_after <= 1.01, f"clipped norm should be ≤ 1.0, got {total_norm_after:.4f}"
print("  ✓ Section 5 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Automatic Mixed Precision (AMP)
# ─────────────────────────────────────────────────────────────
print("\n── Section 6: AMP (Automatic Mixed Precision) ──")
print("""
  AMP runs forward/backward in FP16 (activating Tensor Cores),
  keeps master weights in FP32 (avoiding precision loss).
  GradScaler prevents FP16 gradient underflow.

  With AMP + gradient clipping:
    backward()                     → on scaled loss
    scaler.unscale_(optimizer)     → before clipping
    clip_grad_norm_()              → on unscaled gradients
    scaler.step() + scaler.update() → instead of optimizer.step()
""")

if DEVICE != "cuda":
    print("  (AMP is only meaningful on CUDA — skipped on CPU)")
else:
    model_amp = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    ).to(DEVICE)
    opt_amp = torch.optim.Adam(model_amp.parameters(), lr=1e-3)
    scaler  = torch.cuda.amp.GradScaler()

    model_amp.train()
    amp_losses = []
    t0 = time.perf_counter()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_amp.zero_grad(set_to_none=True)

        # TODO 14: Wrap forward in torch.autocast for FP16
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):  # YOUR CODE HERE
            pred = model_amp(xb)
            loss = criterion(pred, yb)

        # TODO 15: Scale and backward
        pass  # YOUR CODE HERE  → scaler.scale(loss).backward()

        # (optional: unscale + clip before step)
        scaler.unscale_(opt_amp)
        torch.nn.utils.clip_grad_norm_(model_amp.parameters(), max_norm=1.0)

        # TODO 16: scaler.step() then scaler.update()
        pass  # YOUR CODE HERE

        amp_losses.append(loss.item())

    amp_time = time.perf_counter() - t0
    print(f"  AMP 1-epoch training: {amp_time*1000:.1f} ms")
    print(f"  Final AMP loss: {amp_losses[-1]:.4f}")
    print("  ✓ Section 6 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Inference (eval mode + no_grad)
# ─────────────────────────────────────────────────────────────
print("\n── Section 7: Inference ──")
print("""
  Inference checklist:
    model.eval()          — disable dropout, freeze BatchNorm
    torch.no_grad()       — suppress computation graph (faster, less memory)
  Both are required. Forgetting either causes bugs or wasted compute.
""")

model.eval()
test_x = torch.tensor([[0.0], [1.5707], [3.1415]], device=DEVICE)   # 0, π/2, π

# TODO 17: Run inference — wrap with torch.no_grad()
with torch.no_grad():  # YOUR CODE HERE
    pred_y = model(test_x)

assert pred_y is not None,         "pred_y is None"
assert pred_y.shape == (3, 1),    f"pred_y shape wrong: {pred_y.shape}"
print(f"  sin(0)   pred: {pred_y[0].item():.3f}  (true:  0.000)")
print(f"  sin(π/2) pred: {pred_y[1].item():.3f}  (true:  1.000)")
print(f"  sin(π)   pred: {pred_y[2].item():.3f}  (true:  0.000)")
print("  ✓ Section 7 passed")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 04 complete!")
print("  You now know the full training loop including AMP and gradient clipping.")
print("  Next: I.Foundations/2.PyTorch_Fundamentals/2.5_gpu_timing.py")
print("=" * 60)
