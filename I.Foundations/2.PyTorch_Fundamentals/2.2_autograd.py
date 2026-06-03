#!/usr/bin/env python3
"""
I.Foundations/2.PyTorch_Fundamentals/2.2_autograd.py  ─  PyTorch Basics: Autograd & Gradients
=================================================================
"""

import torch
import torch.nn as nn

print("=" * 55)
print("  Exercise 02 — Autograd & Gradients")
print("=" * 55)

# ─────────────────────────────────────────────────────────
# SECTION 1: requires_grad and grad_fn
# ─────────────────────────────────────────────────────────
print("\n── Section 1: requires_grad ──")

# TODO 1: Create a tensor x = [2.0, 3.0] that TRACKS gradients
x = torch.tensor([2.0, 3.0], requires_grad=True)

# TODO 2: Compute y = x[0]^2 + 3*x[1]   →   y = 4 + 9 = 13
y = x[0] ** 2 + 3 * x[1]

# TODO 3: Call backward on y
y.backward()

# TODO 4: What is the gradient of y w.r.t. x?
#   dy/dx[0] = 2*x[0] = 4
#   dy/dx[1] = 3
# Assign x.grad to grad_x
grad_x = x.grad

assert x   is not None and x.requires_grad,        "x must require grad"
assert y   is not None and y.item() == 13.0,        "y should be 13"
assert grad_x is not None,                          "call backward first, then access x.grad"
assert grad_x.tolist() == [4.0, 3.0],              f"grad_x wrong: {grad_x}"
print("  ✓ Section 1 passed")

# ─────────────────────────────────────────────────────────
# SECTION 2: torch.no_grad() — disabling gradient tracking
# ─────────────────────────────────────────────────────────
print("\n── Section 2: torch.no_grad() ──")

model_weight = torch.tensor([2.0], requires_grad=True)

# TODO 5: Inside torch.no_grad(), compute z = model_weight * 5.0
# z should NOT have a grad_fn (no graph built)
with torch.no_grad():
    z = model_weight * 5.0

# TODO 6: Does z have requires_grad?
z_requires_grad = z.requires_grad

assert z is not None and z.item() == 10.0,          "z should be 10"
assert z_requires_grad == False,                    "inside no_grad, output has no grad"
assert z.grad_fn is None,                           "z should have no grad_fn"
print("  ✓ Section 2 passed")
print("  → Always use torch.no_grad() during inference — saves memory + time")

# ─────────────────────────────────────────────────────────
# SECTION 3: Gradient accumulation pitfall
# ─────────────────────────────────────────────────────────
print("\n── Section 3: Gradient Accumulation ──")
print("  PyTorch ACCUMULATES gradients — you must zero them each step!")

w = torch.tensor([1.0], requires_grad=True)

# Step 1
loss1 = (w * 3).sum()
loss1.backward()
grad_after_step1 = w.grad.item()   # should be 3.0

# Step 2 WITHOUT zeroing grads
loss2 = (w * 3).sum()
loss2.backward()
grad_after_step2_no_zero = w.grad.item()   # WRONG: accumulated = 6.0

# TODO 7: Zero the gradient, then do step 2 again
w.grad.zero_()

# TODO 8: Compute loss3 = (w * 3).sum() and call backward
loss3 = (w * 3).sum()
loss3.backward()

# TODO 9: What is w.grad now?
grad_after_zero = w.grad.item()

assert grad_after_step1 == 3.0,   "first backward should give 3.0"
assert grad_after_step2_no_zero == 6.0,  "without zeroing, grad accumulates to 6.0"
assert grad_after_zero is not None and grad_after_zero == 3.0, \
    f"after zeroing, grad should be 3.0, got {grad_after_zero}"
print("  ✓ Section 3 passed")
print("  → optimizer.zero_grad(set_to_none=True) before every backward!")

# ─────────────────────────────────────────────────────────
# SECTION 4: A complete training step
# ─────────────────────────────────────────────────────────
print("\n── Section 4: Complete Training Step ──")

# Simple linear model: y = W*x + b
W = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([0.5, 0.5], requires_grad=True)

x_input = torch.tensor([1.0, 1.0])
y_target = torch.tensor([3.0, 7.0])

optimizer = torch.optim.SGD([W, b], lr=0.01)

# TODO 10: Zero gradients
optimizer.zero_grad()
# TODO 11: Compute prediction: pred = x_input @ W.T + b  (matrix-vector multiply)
pred = x_input @ W.T + b
# TODO 12: Compute MSE loss: loss = mean((pred - y_target)^2)
loss = ((pred - y_target) ** 2).mean()
# TODO 13: Backpropagate
loss.backward()
# TODO 14: Optimizer step
optimizer.step()

W_grad_exists    = W.grad is not None
b_grad_exists    = b.grad is not None
loss_decreased   = loss.item() >= 0   # just check it's a valid loss

assert pred is not None,     "pred is None"
assert loss is not None,     "loss is None"
assert W_grad_exists,        "W should have gradients after backward"
assert b_grad_exists,        "b should have gradients after backward"
print(f"  loss = {loss.item():.4f}  W.grad exists: {W_grad_exists}")
print("  ✓ Section 4 passed")

# ─────────────────────────────────────────────────────────
# SECTION 5: .detach() and stopping gradient flow
# ─────────────────────────────────────────────────────────
print("\n── Section 5: .detach() ──")
print("  .detach() creates a new tensor that shares data but has NO grad_fn.")
print("  Use it to: log loss values, compute metrics, stop gradient flow.\n")

w2 = torch.tensor([3.0], requires_grad=True)
result = w2 * w2 * 5   # result has grad_fn

# TODO 15: Detach result from the computation graph
result_detached = result.detach()

# TODO 16: Does result_detached require grad?
detached_req_grad = result_detached.requires_grad

assert result_detached is not None,                    "detach result"
assert result_detached.item() == 45.0,                "value should still be 45"
assert detached_req_grad == False,                    "detached tensor has no grad"
assert result_detached.grad_fn is None,               "detached has no grad_fn"
print("  ✓ Section 5 passed")
print("  → loss.item() implicitly detaches — safe for logging, wrong for backprop")

print("\n" + "=" * 55)
print("  ALL SECTIONS PASSED — Exercise 02 complete!")
print("  Next: I.Foundations/2.PyTorch_Fundamentals/2.3_nn_modules.py")
print("=" * 55)