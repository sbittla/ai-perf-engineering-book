#!/usr/bin/env python3
"""
exercise_01_tensors.py  ─  PyTorch Basics: Tensors & Operations
================================================================
"""

import torch

print("=" * 55)
print("  Exercise 01 — Tensors & Operations")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# SECTION 1: Creating tensors
# ─────────────────────────────────────────────────────────
print("\n── Section 1: Creating Tensors ──")

# TODO 1: Create a 1-D tensor with values [1.0, 2.0, 3.0, 4.0, 5.0]
t1 = None  # YOUR CODE HERE

# TODO 2: Create a 3×4 tensor filled with zeros, dtype=float32
t2 = None  # YOUR CODE HERE

# TODO 3: Create a 2×3 tensor of random floats between 0 and 1
t3 = None  # YOUR CODE HERE

# TODO 4: Create a 1-D tensor containing integers 0..9 (like Python range)
t4 = None  # YOUR CODE HERE

# --- checks ---
assert t1 is not None and t1.shape == (5,),          "t1 should have shape (5,)"
assert t1.sum().item() == 15.0,                       "t1 values should sum to 15"
assert t2 is not None and t2.shape == (3, 4),         "t2 should be 3×4"
assert t2.sum().item() == 0.0,                        "t2 should be all zeros"
assert t3 is not None and t3.shape == (2, 3),         "t3 should be 2×3"
assert 0.0 <= t3.min().item() and t3.max().item() <= 1.0, "t3 should be in [0,1]"
assert t4 is not None and list(t4.numpy()) == list(range(10)), "t4 should be 0..9"
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────
# SECTION 2: Shape operations
# ─────────────────────────────────────────────────────────
print("\n── Section 2: Shape Operations ──")

base = torch.arange(24, dtype=torch.float32)

# TODO 5: Reshape base into shape (2, 3, 4)
t5 = None  # YOUR CODE HERE

# TODO 6: Transpose t5 so axes become (2, 4, 3) — swap last two dims
t6 = None  # YOUR CODE HERE

# TODO 7: Flatten t5 back to 1-D
t7 = None  # YOUR CODE HERE

# TODO 8: Add a batch dimension to base: shape (1, 24)
t8 = None  # YOUR CODE HERE

# --- checks ---
assert t5 is not None and t5.shape == (2, 3, 4),  "t5 should be (2,3,4)"
assert t6 is not None and t6.shape == (2, 4, 3),  "t6 should be (2,4,3)"
assert t7 is not None and t7.shape == (24,),       "t7 should be flat (24,)"
assert t8 is not None and t8.shape == (1, 24),     "t8 should be (1,24)"
print("  ✓ Section 2 passed")

# ─────────────────────────────────────────────────────────
# SECTION 3: Math operations
# ─────────────────────────────────────────────────────────
print("\n── Section 3: Math Operations ──")

a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

# TODO 9: Element-wise multiplication of a and b
t9 = None  # YOUR CODE HERE

# TODO 10: Matrix multiplication of a and b  (use torch.mm or @)
t10 = None  # YOUR CODE HERE

# TODO 11: Mean of all elements in a
t11 = None  # YOUR CODE HERE

# TODO 12: Column-wise sum of a  →  shape (2,)
t12 = None  # YOUR CODE HERE

# --- checks ---
assert t9  is not None and t9.tolist()  == [[5.,12.],[21.,32.]], "t9 element-wise mul failed"
assert t10 is not None and t10.tolist() == [[19.,22.],[43.,50.]],"t10 matmul failed"
assert t11 is not None and t11.item()   == 2.5,                  "t11 mean failed"
assert t12 is not None and t12.tolist() == [4., 6.],             "t12 col-sum failed"
print("  ✓ Section 3 passed")

# ─────────────────────────────────────────────────────────
# SECTION 4: Indexing and slicing
# ─────────────────────────────────────────────────────────
print("\n── Section 4: Indexing & Slicing ──")

m = torch.arange(16, dtype=torch.float32).reshape(4, 4)
# m = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

# TODO 13: Get the element at row 2, col 3  (should be 11.0)
t13 = None  # YOUR CODE HERE

# TODO 14: Get the entire second row  →  tensor([4,5,6,7])
t14 = None  # YOUR CODE HERE

# TODO 15: Get rows 1 and 2, cols 1 and 2  →  shape (2,2)
t15 = None  # YOUR CODE HERE

# TODO 16: Boolean mask — select all elements > 8
t16 = None  # YOUR CODE HERE

# --- checks ---
assert t13 is not None and t13.item() == 11.0,         "t13 element wrong"
assert t14 is not None and t14.tolist() == [4,5,6,7],  "t14 row wrong"
assert t15 is not None and t15.shape == (2,2),          "t15 slice wrong"
assert t16 is not None and sorted(t16.tolist()) == [9,10,11,12,13,14,15], "t16 mask wrong"
print("  ✓ Section 4 passed")

# ─────────────────────────────────────────────────────────
# SECTION 5: Device movement (CPU ↔ GPU)
# ─────────────────────────────────────────────────────────
print("\n── Section 5: Device Movement ──")

cpu_tensor = torch.randn(3, 3)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# TODO 17: Move cpu_tensor to DEVICE (use .to())
t17 = None  # YOUR CODE HERE

# TODO 18: Move t17 back to CPU and convert to numpy
t18_numpy = None  # YOUR CODE HERE

# TODO 19: Create a new tensor directly on DEVICE
t19 = None  # YOUR CODE HERE

# --- checks ---
assert t17 is not None and str(t17.device).startswith(DEVICE), "t17 not on correct device"
assert t18_numpy is not None,                                    "t18_numpy is None"
assert t19 is not None and str(t19.device).startswith(DEVICE), "t19 not on DEVICE"
print(f"  ✓ Section 5 passed  (device={DEVICE})")

# ─────────────────────────────────────────────────────────
# SECTION 6: Dtype and memory
# ─────────────────────────────────────────────────────────
print("\n── Section 6: Dtypes & Memory ──")

x32 = torch.randn(1000, 1000, dtype=torch.float32)

# TODO 20: Convert x32 to float16
x16 = None  # YOUR CODE HERE

# TODO 21: How many bytes does x32 use? (use .element_size() * .numel())
bytes32 = None  # YOUR CODE HERE

# TODO 22: How many bytes does x16 use?
bytes16 = None  # YOUR CODE HERE

# --- checks ---
assert x16 is not None and x16.dtype == torch.float16,  "x16 should be float16"
assert bytes32 is not None and bytes32 == 4_000_000,     "bytes32 should be 4MB"
assert bytes16 is not None and bytes16 == 2_000_000,     "bytes16 should be 2MB"
assert bytes32 == bytes16 * 2,                            "FP32 should be 2× FP16"
print("  ✓ Section 6 passed")

print("\n" + "=" * 55)
print("  ALL SECTIONS PASSED — Exercise 01 complete!")
print("  Next: exercise_02_autograd.py")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# HINTS (uncomment if stuck)
# ─────────────────────────────────────────────────────────
# TODO 1:  torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
# TODO 2:  torch.zeros(3, 4)
# TODO 3:  torch.rand(2, 3)
# TODO 4:  torch.arange(10)
# TODO 5:  base.reshape(2, 3, 4)
# TODO 6:  t5.transpose(1, 2)  or  t5.permute(0, 2, 1)
# TODO 7:  t5.flatten()  or  t5.reshape(-1)
# TODO 8:  base.unsqueeze(0)  or  base.reshape(1, 24)
# TODO 9:  a * b
# TODO 10: a @ b  or  torch.mm(a, b)
# TODO 11: a.mean()
# TODO 12: a.sum(dim=0)
# TODO 13: m[2, 3]
# TODO 14: m[1]  or  m[1, :]
# TODO 15: m[1:3, 1:3]
# TODO 16: m[m > 8]
# TODO 17: cpu_tensor.to(DEVICE)
# TODO 18: t17.cpu().numpy()
# TODO 19: torch.zeros(2, device=DEVICE)
# TODO 20: x32.half()  or  x32.to(torch.float16)
# TODO 21: x32.element_size() * x32.numel()
# TODO 22: x16.element_size() * x16.numel()
