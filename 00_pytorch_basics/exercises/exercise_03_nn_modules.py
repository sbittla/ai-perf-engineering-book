#!/usr/bin/env python3
"""
exercise_03_nn_modules.py  ─  PyTorch Basics: Building Neural Networks
=======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 55)
print("  Exercise 03 — nn.Module & Neural Networks")
print("=" * 55)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────
# SECTION 1: Using built-in layers
# ─────────────────────────────────────────────────────────
print("\n── Section 1: Built-in Layers ──")

# TODO 1: Create a Linear layer: 8 inputs → 4 outputs, no bias
linear = None  # YOUR CODE HERE

# TODO 2: Create a layer stack with nn.Sequential:
#   Linear(16, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 10)
mlp = None  # YOUR CODE HERE

# TODO 3: Count total trainable parameters in mlp
n_params = None  # YOUR CODE HERE

x = torch.randn(4, 16)   # batch of 4, feature dim 16
out = mlp(x)

assert linear is not None and isinstance(linear, nn.Linear), "linear should be nn.Linear"
assert linear.bias is None,                                   "linear should have no bias"
assert mlp is not None and isinstance(mlp, nn.Sequential),   "mlp should be Sequential"
assert out.shape == (4, 10),                                  f"mlp output shape wrong: {out.shape}"
assert n_params is not None and n_params == 16*64+64 + 64*32+32 + 32*10+10, \
    f"n_params wrong: {n_params}"
print(f"  mlp params: {n_params}  output shape: {out.shape}")
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────
# SECTION 2: Custom nn.Module
# ─────────────────────────────────────────────────────────
print("\n── Section 2: Custom nn.Module ──")

class ResidualBlock(nn.Module):
    """
    A residual block: output = LayerNorm(x + Linear(x))
    Used in transformers: the residual connection preserves gradient flow.
    """
    def __init__(self, dim: int):
        super().__init__()
        # TODO 4: Define self.linear as a Linear layer: dim → dim
        self.linear = None  # YOUR CODE HERE
        # TODO 5: Define self.norm as a LayerNorm over dim
        self.norm = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 6: Return LayerNorm(x + linear(x))
        pass  # YOUR CODE HERE

dim = 32
block = ResidualBlock(dim).to(DEVICE)
inp   = torch.randn(2, 10, dim, device=DEVICE)   # (batch, seq, dim)
out_b = block(inp)

assert out_b.shape == inp.shape,                   f"ResidualBlock output shape wrong: {out_b.shape}"
assert hasattr(block, 'linear'),                   "block should have self.linear"
assert hasattr(block, 'norm'),                     "block should have self.norm"
# Gradient should flow: check that linear.weight has a grad after backward
loss_b = out_b.sum()
loss_b.backward()
assert block.linear.weight.grad is not None,       "linear.weight should have grad"
print("  ✓ Section 2 passed — ResidualBlock works and gradients flow")

# ─────────────────────────────────────────────────────────
# SECTION 3: train() vs eval() mode
# ─────────────────────────────────────────────────────────
print("\n── Section 3: train() vs eval() ──")
print("  Dropout and BatchNorm behave DIFFERENTLY in train vs eval mode.")
print("  Forgetting model.eval() during inference causes non-deterministic results!\n")

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.9)   # high dropout to make difference visible

    def forward(self, x):
        return self.dropout(self.fc(x))

mdl = ModelWithDropout()
x   = torch.ones(100, 10)

# TODO 7: Get output in TRAIN mode (dropout active — output will vary)
pass  # YOUR CODE HERE
with torch.no_grad():
    out_train = mdl(x)

# TODO 8: Get output in EVAL mode (dropout disabled — output is deterministic)
pass  # YOUR CODE HERE
with torch.no_grad():
    out_eval1 = mdl(x)
    out_eval2 = mdl(x)   # run again

# In train mode, dropout zeros ~90% of values randomly → high variance
train_zeros = (out_train == 0).float().mean().item()
eval_zeros  = (out_eval1 == 0).float().mean().item()
deterministic = torch.allclose(out_eval1, out_eval2)

print(f"  Train mode zero fraction: {train_zeros:.2f}  (expect ~0.9 with p=0.9)")
print(f"  Eval  mode zero fraction: {eval_zeros:.2f}  (expect 0.0)")
print(f"  Eval mode deterministic:  {deterministic}")

assert train_zeros > 0.5,   "dropout should zero many values in train mode"
assert eval_zeros  == 0.0,  "dropout should be disabled in eval mode"
assert deterministic,       "eval mode should give same output twice"
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────
# SECTION 4: Saving and loading a model
# ─────────────────────────────────────────────────────────
print("\n── Section 4: Saving & Loading ──")

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
net.eval()

# TODO 9: Save the state dict to "test_model.pt"
pass  # YOUR CODE HERE

# TODO 10: Create a NEW model with the same architecture
net2 = None  # YOUR CODE HERE

# TODO 11: Load the saved weights into net2
pass  # YOUR CODE HERE

# TODO 12: Verify outputs match
x_test = None  # YOUR CODE HERE
with torch.no_grad():
    out_orig   = net(x_test)
    out_loaded = net2(x_test) if net2 else None

assert net2 is not None,                              "net2 is None"
assert out_loaded is not None,                        "run net2 forward"
assert torch.allclose(out_orig, out_loaded),          "loaded weights should give same output"
print("  ✓ Section 4 passed")

import os; os.remove("test_model.pt")

# ─────────────────────────────────────────────────────────
# SECTION 5: Moving a model to GPU
# ─────────────────────────────────────────────────────────
print("\n── Section 5: Model on Device ──")

net3 = nn.Linear(8, 4)

# TODO 13: Move net3 to DEVICE
net3 = None  # YOUR CODE HERE
# TODO 14: Verify all parameters are on DEVICE
all_on_device = None  # YOUR CODE HERE

x_dev = torch.randn(2, 8, device=DEVICE)
out_dev = net3(x_dev)

assert all_on_device is not None and all_on_device, \
    "all parameters should be on DEVICE after .to(DEVICE)"
assert str(out_dev.device).startswith(DEVICE), \
    "output should be on DEVICE"
print(f"  ✓ Section 5 passed  (device={DEVICE})")

print("\n" + "=" * 55)
print("  ALL SECTIONS PASSED — Exercise 03 complete!")
print("  Next: exercise_04_training_loop.py")
print("=" * 55)