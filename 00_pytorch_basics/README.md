# 00 — PyTorch Basics

**Complete these exercises before starting the main course.**

Estimated time: 2–3 days

---

## Why This Section Exists

The rest of this course assumes you can read and write PyTorch fluently.
These exercises give you a fast, practical on-ramp through the exact concepts
you'll encounter when profiling and optimising models.

## Exercise Order

| # | File | Topics | Time |
|---|------|--------|------|
| 01 | `exercise_01_tensors.py` | Creation, shapes, dtypes, devices, math, indexing | 30 min |
| 02 | `exercise_02_autograd.py` | `requires_grad`, backward, `no_grad`, gradient accumulation pitfall | 45 min |
| 03 | `exercise_03_nn_modules.py` | `nn.Module`, `Sequential`, custom modules, train/eval, save/load | 45 min |
| 04 | `exercise_04_training_loop.py` | Dataset, DataLoader, full train loop, AMP | 60 min |
| 05 | `exercise_05_performance_basics.py` | CUDA events, warmup, profiler, memory tracking, NVTX | 60 min |

## How to Use

Each file has `TODO` blocks. Fill them in, then run:

```bash
python exercise_01_tensors.py
```

All assertions must pass. Hints are at the bottom of each file.

Worked solutions are provided in the book.

## Key Concepts by Exercise

### Exercise 01 — Tensors
```python
# Create
t = torch.tensor([1.0, 2.0])          # from list
t = torch.zeros(3, 4)                  # zeros
t = torch.rand(2, 3)                   # uniform [0,1]
t = torch.arange(10)                   # 0..9

# Shape
t.reshape(2, 5)                        # reinterpret shape
t.transpose(0, 1)                      # swap dims
t.flatten()                            # → 1D
t.unsqueeze(0)                         # add dim

# Move to GPU
t_gpu = t.to("cuda")
t_cpu = t_gpu.cpu()

# Dtype
t.half()                               # float16
t.to(torch.bfloat16)                   # bfloat16
```

### Exercise 02 — Autograd
```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
print(x.grad)                          # dy/dx = 2x = 4

# Always zero before each step
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

# No gradient tracking for inference
with torch.no_grad():
    pred = model(x)
```

### Exercise 03 — Modules
```python
# Custom module
class MyLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, d)

    def forward(self, x):
        return self.w(x) + x   # residual

# ALWAYS set mode
model.train()   # dropout active, BN updates
model.eval()    # dropout disabled, BN frozen
```

### Exercise 04 — Training Loop
```python
for xb, yb in dataloader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)  # must zero each step
    pred = model(xb)
    loss = criterion(pred, yb)
    loss.backward()
    optimizer.step()
```

### Exercise 05 — Performance
```python
# CORRECT GPU timing
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
output = model(x)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end)           # milliseconds

# Warmup before benchmarking
for _ in range(5): model(x)            # discard first calls
```
