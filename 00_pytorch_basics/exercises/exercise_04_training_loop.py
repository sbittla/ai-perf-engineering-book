#!/usr/bin/env python3
"""
exercise_04_training_loop.py  ─  PyTorch Basics: The Training Loop
====================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

print("=" * 55)
print("  Exercise 04 — The Training Loop")
print("=" * 55)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────
# SECTION 1: Custom Dataset
# ─────────────────────────────────────────────────────────
print("── Section 1: Custom Dataset ──")

class SineDataset(Dataset):
    """y = sin(x) + noise  — a simple regression task."""

    def __init__(self, n_samples: int, noise: float = 0.1):
        self.x = torch.linspace(-3.14, 3.14, n_samples).unsqueeze(1)   # (N, 1)
        self.y = torch.sin(self.x) + noise * torch.randn_like(self.x)  # (N, 1)

    def __len__(self):
        # TODO 1: Return the number of samples
        pass  # YOUR CODE HERE

    def __getitem__(self, idx):
        # TODO 2: Return the (input, target) pair at index idx
        pass  # YOUR CODE HERE

# Test it
ds = SineDataset(200)
sample_x, sample_y = ds[0]
assert len(ds) == 200,                    "Dataset length wrong"
assert sample_x.shape == (1,),           f"x shape wrong: {sample_x.shape}"
assert sample_y.shape == (1,),           f"y shape wrong: {sample_y.shape}"
print("  ✓ Dataset works")

# ─────────────────────────────────────────────────────────
# SECTION 2: DataLoader
# ─────────────────────────────────────────────────────────
print("\n── Section 2: DataLoader ──")

train_ds = SineDataset(800)
val_ds   = SineDataset(200)

# TODO 3: Create a DataLoader for train_ds
#   batch_size=32, shuffle=True, num_workers=0
train_loader = None  # YOUR CODE HERE

# TODO 4: Create a DataLoader for val_ds
#   batch_size=64, shuffle=False, num_workers=0
val_loader = None  # YOUR CODE HERE

# Check one batch
batch_x, batch_y = next(iter(train_loader))
assert train_loader is not None,                    "train_loader is None"
assert val_loader is not None,                      "val_loader is None"
assert batch_x.shape == (32, 1),                   f"batch shape wrong: {batch_x.shape}"
print(f"  Batches in train_loader: {len(train_loader)}")
print(f"  Batches in val_loader  : {len(val_loader)}")
print("  ✓ DataLoaders work")

# ─────────────────────────────────────────────────────────
# SECTION 3: Build the Model
# ─────────────────────────────────────────────────────────
print("\n── Section 3: Model ──")

# TODO 5: Build a 3-layer MLP for regression:
#   Linear(1→64) → ReLU → Linear(64→64) → ReLU → Linear(64→1)
model = None  # YOUR CODE HERE

# TODO 6: Move model to DEVICE
model = None  # YOUR CODE HERE

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params}")
assert model is not None,                           "model is None"
assert next(model.parameters()).device.type == DEVICE, "model not on DEVICE"
print("  ✓ Model built")

# ─────────────────────────────────────────────────────────
# SECTION 4: Training Loop
# ─────────────────────────────────────────────────────────
print("\n── Section 4: Training Loop ──")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 5
history  = {"train_loss": [], "val_loss": []}

for epoch in range(N_EPOCHS):
    # ── TRAINING PHASE ────────────────────────────────────
    model.train()
    train_losses = []

    for xb, yb in train_loader:
        # TODO 7: Move xb and yb to DEVICE
        xb = None  # YOUR CODE HERE
        yb = yb.to(DEVICE)

        # TODO 8: Zero the gradients
        pass  # YOUR CODE HERE

        # TODO 9: Forward pass — get predictions
        pred = None  # YOUR CODE HERE

        # TODO 10: Compute MSE loss
        loss = None  # YOUR CODE HERE

        # TODO 11: Backward pass
        pass  # YOUR CODE HERE

        # TODO 12: Optimizer step
        pass  # YOUR CODE HERE

        train_losses.append(loss.item())

    # ── VALIDATION PHASE ──────────────────────────────────
    model.eval()
    val_losses = []

    # TODO 13: Wrap the validation loop in torch.no_grad()
    pass  # YOUR CODE HERE
        for xb, yb in val_loader:
            xb  = xb.to(DEVICE)
            yb  = yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            val_losses.append(loss.item())

    avg_train = sum(train_losses) / len(train_losses)
    avg_val   = sum(val_losses)   / len(val_losses)
    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)

    print(f"  Epoch {epoch+1}/{N_EPOCHS}  "
          f"train={avg_train:.4f}  val={avg_val:.4f}")

# Verify loss decreased
assert history["train_loss"][-1] < history["train_loss"][0], \
    "Training loss should decrease over epochs"
print("  ✓ Training loop works — loss decreased")

# ─────────────────────────────────────────────────────────
# SECTION 5: Mixed Precision Training (AMP)
# ─────────────────────────────────────────────────────────
print("\n── Section 5: AMP (Automatic Mixed Precision) ──")
print("  AMP runs forward/backward in FP16, keeps master weights in FP32.")
print("  Typical speedup: 2–3× on CUDA with Tensor Cores.\n")

if DEVICE != "cuda":
    print("  (Skipped — AMP is only meaningful on CUDA)")
else:
    # Rebuild model for clean state
    model_amp = nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1)
    ).to(DEVICE)
    opt_amp    = torch.optim.Adam(model_amp.parameters(), lr=1e-3)

    # GradScaler prevents FP16 gradient underflow
    scaler = torch.cuda.amp.GradScaler()

    model_amp.train()
    t0 = time.perf_counter()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt_amp.zero_grad(set_to_none=True)

        # TODO 14: Use torch.autocast to run forward in FP16
        pass  # YOUR CODE HERE
            pred = model_amp(xb)
            loss = criterion(pred, yb)

        # TODO 15: Use scaler.scale(loss).backward()
        pass  # YOUR CODE HERE

        # TODO 16: Use scaler.step(opt_amp) and scaler.update()
        pass  # YOUR CODE HERE
        scaler.update()

    amp_time = time.perf_counter() - t0
    print(f"  AMP training time (1 epoch): {amp_time*1000:.1f}ms")
    print("  ✓ Section 5 passed")

# ─────────────────────────────────────────────────────────
# SECTION 6: Model inference (no grad, eval mode)
# ─────────────────────────────────────────────────────────
print("\n── Section 6: Inference ──")

model.eval()
test_x = torch.tensor([[0.0], [1.5707], [3.1415]], device=DEVICE)   # 0, π/2, π

# TODO 17: Run model inference — no_grad + eval mode
pass  # YOUR CODE HERE
    pred_y = model(test_x)

# sin(0)=0, sin(π/2)≈1, sin(π)≈0
assert pred_y is not None,             "pred_y is None"
assert pred_y.shape == (3, 1),         f"pred_y shape wrong: {pred_y.shape}"
print(f"  sin(0)    pred: {pred_y[0].item():.3f}  (true: 0.000)")
print(f"  sin(π/2)  pred: {pred_y[1].item():.3f}  (true: 1.000)")
print(f"  sin(π)    pred: {pred_y[2].item():.3f}  (true: 0.000)")
print("  ✓ Section 6 passed")

print("\n" + "=" * 55)
print("  ALL SECTIONS PASSED — Exercise 04 complete!")
print("  You now understand the full training loop.")
print("  Next: exercise_05_performance_basics.py")
print("=" * 55)