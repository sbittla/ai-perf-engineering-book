#!/usr/bin/env python3
"""
I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py  ─  Chapter 2: nn.Module & Neural Networks
=====================================================================
Covers book sections 2.3:
  • Built-in layers and nn.Sequential
  • Custom nn.Module (ResidualBlock)
  • train() vs eval() mode
  • Saving and loading state dicts
  • Moving models to device
  • Parameter and buffer registration
  • Module introspection

Run:  python I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py
All sections must print ✓.
"""

import os
import torch
import torch.nn as nn

print("=" * 60)
print("  Exercise 03 — nn.Module & Neural Networks")
print("=" * 60)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {DEVICE}\n")

# ─────────────────────────────────────────────────────────────
# SECTION 1: Built-in layers and nn.Sequential
# ─────────────────────────────────────────────────────────────
print("── Section 1: Built-in Layers ──")
print("""
  nn.Linear(in, out) is the fundamental building block.
  nn.Sequential chains layers: output of each feeds the next.
  This is the workhorse for feedforward networks.
""")

# TODO 1: Create a Linear layer — 8 inputs, 4 outputs, no bias
linear = None  # YOUR CODE HERE

# TODO 2: Create a 3-layer MLP with nn.Sequential:
#   Linear(16, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 10)
mlp = None  # YOUR CODE HERE

# TODO 3: Count total trainable parameters in mlp
#   Formula: each Linear(M, N) with bias has M*N + N parameters
n_params = None  # YOUR CODE HERE

x   = torch.randn(4, 16)   # batch of 4, feature dim 16
out = mlp(x)

assert linear is not None and isinstance(linear, nn.Linear), "linear should be nn.Linear"
assert linear.bias is None,                                   "linear should have no bias"
assert mlp    is not None and isinstance(mlp, nn.Sequential),"mlp should be Sequential"
assert out.shape == (4, 10),                                  f"mlp output shape: {out.shape}"
assert n_params == 16*64+64 + 64*32+32 + 32*10+10,          f"n_params wrong: {n_params}"
print(f"  mlp params: {n_params}  output shape: {out.shape}")
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Custom nn.Module — ResidualBlock
# ─────────────────────────────────────────────────────────────
print("\n── Section 2: Custom nn.Module ──")
print("""
  Subclass nn.Module when nn.Sequential is not enough:
  residual connections, attention, or any branching topology.
  Rules:
    • Define all sub-layers in __init__ (so they are registered)
    • Implement the forward pass in forward()
    • Never call forward() directly — use model(x) so hooks run
""")

class ResidualBlock(nn.Module):
    """output = LayerNorm(x + Linear(x))  — the core transformer block pattern."""

    def __init__(self, dim: int):
        super().__init__()
        # TODO 4: Define self.linear as nn.Linear(dim, dim)
        self.linear = None  # YOUR CODE HERE
        # TODO 5: Define self.norm as nn.LayerNorm(dim)
        self.norm = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 6: Return self.norm(x + self.linear(x))
        pass  # YOUR CODE HERE

dim   = 32
block = ResidualBlock(dim).to(DEVICE)
inp   = torch.randn(2, 10, dim, device=DEVICE)   # (batch, seq_len, dim)
out_b = block(inp)

assert out_b.shape == inp.shape,          f"ResidualBlock shape wrong: {out_b.shape}"
assert hasattr(block, "linear"),          "block must have self.linear"
assert hasattr(block, "norm"),            "block must have self.norm"

loss_b = out_b.sum()
loss_b.backward()
assert block.linear.weight.grad is not None, "gradients must flow through linear"
print("  ✓ Section 2 passed — ResidualBlock works, gradients flow")

# ─────────────────────────────────────────────────────────────
# SECTION 3: train() vs eval() mode
# ─────────────────────────────────────────────────────────────
print("\n── Section 3: train() vs eval() ──")
print("""
  Two layers behave DIFFERENTLY in train vs eval mode:
    Dropout:   zeros random activations during training (regularisation),
               disabled during eval (deterministic output).
    BatchNorm: uses batch statistics during training,
               uses frozen running stats during eval.
  ALWAYS call model.eval() before inference or benchmarking.
""")

class ModelWithDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc      = nn.Linear(10, 10)
        self.dropout = nn.Dropout(p=0.9)   # high p makes the difference obvious

    def forward(self, x):
        return self.dropout(self.fc(x))

mdl = ModelWithDropout()
x   = torch.ones(100, 10)

# TODO 7: Switch mdl to TRAIN mode (model.train())
pass  # YOUR CODE HERE
with torch.no_grad():
    out_train = mdl(x)

# TODO 8: Switch mdl to EVAL mode (model.eval())
pass  # YOUR CODE HERE
with torch.no_grad():
    out_eval1 = mdl(x)
    out_eval2 = mdl(x)

train_zeros   = (out_train == 0).float().mean().item()
eval_zeros    = (out_eval1 == 0).float().mean().item()
deterministic = torch.allclose(out_eval1, out_eval2)

print(f"  Train mode zero fraction : {train_zeros:.2f}  (expect ~0.9 with p=0.9)")
print(f"  Eval  mode zero fraction : {eval_zeros:.2f}  (expect 0.0)")
print(f"  Eval mode deterministic  : {deterministic}")
assert train_zeros > 0.5,  "dropout should zero many values in train mode"
assert eval_zeros  == 0.0, "dropout should be disabled in eval mode"
assert deterministic,      "eval mode should give same output twice"
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Saving and loading state dicts
# ─────────────────────────────────────────────────────────────
print("\n── Section 4: Saving & Loading ──")
print("""
  Save state_dict (weights only) — NOT the whole model object.
  The model object depends on class definition; state_dict is portable.
""")

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
net.eval()

# TODO 9: Save net's state_dict to "test_model.pt"
pass  # YOUR CODE HERE  → torch.save(...)

# TODO 10: Create net2 with the SAME architecture
net2 = None  # YOUR CODE HERE

# TODO 11: Load saved weights into net2
pass  # YOUR CODE HERE  → net2.load_state_dict(...)
net2.eval()

# TODO 12: Create a test input of shape (2, 4)
x_test = None  # YOUR CODE HERE

with torch.no_grad():
    out_orig   = net(x_test)
    out_loaded = net2(x_test)

assert torch.allclose(out_orig, out_loaded), "loaded weights should produce identical output"
print("  ✓ Section 4 passed")
os.remove("test_model.pt")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Moving a model to device
# ─────────────────────────────────────────────────────────────
print("\n── Section 5: Model on Device ──")
print("""
  model.to(device) moves ALL registered parameters and buffers.
  Plain Python attributes (floats, plain tensors) are NOT moved.
  Call .to(device) ONCE at startup — never move back mid-training.
""")

net3 = nn.Linear(8, 4)

# TODO 13: Move net3 to DEVICE in-place (net3 = net3.to(DEVICE))
net3 = None  # YOUR CODE HERE

# TODO 14: Check every parameter is on DEVICE
#   all(p.device.type == DEVICE for p in net3.parameters())
all_on_device = None  # YOUR CODE HERE

x_dev   = torch.randn(2, 8, device=DEVICE)
out_dev = net3(x_dev)

assert all_on_device, "all parameters should be on DEVICE"
assert out_dev.device.type == DEVICE, f"output should be on {DEVICE}"
print(f"  ✓ Section 5 passed  (device={DEVICE})")

# ─────────────────────────────────────────────────────────────
# SECTION 6: Parameter and buffer registration
# ─────────────────────────────────────────────────────────────
print("\n── Section 6: Parameters vs Buffers ──")
print("""
  Not every tensor-shaped thing inside a Module is automatically tracked.
    nn.Parameter   → learnable, shows in model.parameters(), moves with .to()
    register_buffer → non-learnable, in state_dict, moves with .to()
    plain tensor   → invisible to parameters(), does NOT move with .to()

  Common mistake: positional embeddings created as plain tensors stay on CPU
  even after model.to('cuda'). Use register_buffer instead.
""")

class BlockWithBuffer(nn.Module):
    def __init__(self, dim: int, max_seq: int = 128):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

        # Register a fixed positional embedding as a buffer
        # TODO 15: Use self.register_buffer("pos", torch.zeros(max_seq, dim))
        pass  # YOUR CODE HERE

        # This is a plain tensor — it will NOT move with model.to(device)
        self.scale = torch.tensor(1.0)  # intentionally wrong

    def forward(self, x):
        seq = x.size(1)
        return self.linear(x) + self.pos[:seq]   # pos should be on same device as x

blk   = BlockWithBuffer(16).to(DEVICE)
inp16 = torch.randn(2, 8, 16, device=DEVICE)
out16 = blk(inp16)    # should not raise even on CUDA

# Check that 'pos' is registered as a buffer (moves with model)
assert "pos" in dict(blk.named_buffers()), "pos should be a registered buffer"
# Check that 'scale' is NOT in parameters or buffers
param_names  = [n for n, _ in blk.named_parameters()]
buffer_names = [n for n, _ in blk.named_buffers()]
assert "scale" not in param_names,  "plain tensor 'scale' should not be a parameter"
assert "scale" not in buffer_names, "plain tensor 'scale' should not be a buffer"
print(f"  Buffer 'pos' device : {blk.pos.device}")
print(f"  'scale' device      : {blk.scale.device}  ← stays on CPU despite .to()")
print("  ✓ Section 6 passed")

# ─────────────────────────────────────────────────────────────
# SECTION 7: Module introspection
# ─────────────────────────────────────────────────────────────
print("\n── Section 7: Module Introspection ──")
print("""
  Before profiling, always understand the model you are profiling.
  Three tools: str(model), model.parameters(), model.named_parameters()
""")

big_model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

# TODO 16: Count total parameters (sum of p.numel() for all parameters)
total_params = None  # YOUR CODE HERE

# TODO 17: Count trainable parameters (same, but only where p.requires_grad)
trainable_params = None  # YOUR CODE HERE

# TODO 18: Compute model size in MB assuming float32 (4 bytes per element)
model_size_mb = None  # YOUR CODE HERE

assert total_params    == trainable_params, "all params should be trainable by default"
assert total_params    > 0,                 "total_params should be > 0"
assert model_size_mb   > 0,                 "model_size_mb should be > 0"

print(f"  Total params    : {total_params:,}")
print(f"  Trainable params: {trainable_params:,}")
print(f"  Model size (FP32): {model_size_mb:.3f} MB")

# Named parameters — see which layer is which
print("\n  Named parameters:")
for name, param in big_model.named_parameters():
    print(f"    {name:30s}  shape={str(param.shape):15s}  dtype={param.dtype}")

assert total_params == 784*256+256 + 256*128+128 + 128*10+10, \
    f"parameter count wrong: {total_params}"
print("  ✓ Section 7 passed")

print("\n" + "=" * 60)
print("  ALL SECTIONS PASSED — Exercise 03 complete!")
print("  You now know how to build, inspect, save, and deploy PyTorch models.")
print("  Next: I.Foundations/2.PyTorch_Fundamentals/2.4_training_loop.py")
print("=" * 60)
