# Chapter 19 — Capstone 4: CPU-to-GPU Pipeline Flamegraph Challenge

A debugging challenge: find three hidden bottlenecks in a slow training loop using profiling techniques, then fix them and measure the improvement.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `19.1_slow_training_analysis.py` | Three bugs, isolation protocol, what flamegraphs show | `slow_training_step()`, `fixed_step_no_item()`, `measure_eval_overhead()` |
| `19.2_optimised_training.py` | Apply all fixes; final before/after comparison | `fast_step_no_item()`, `fast_step_nonblocking()`, `fast_step_all()`, `evaluate_fast()` |

## Workflow

```bash
python 19.1_slow_training_analysis.py   # saves /tmp/capstone19_diagnosis.json
python 19.2_optimised_training.py        # saves /tmp/capstone19_report.json
```

## The Three Bottlenecks

### Bug 1: `loss.item()` inside the training loop
```python
# WRONG: forces GPU-CPU sync every step
for x, y in loader:
    loss = criterion(model(x), y)
    running_loss += loss.item()   # ← sync here!

# CORRECT: accumulate on GPU; call .item() once per epoch
running_loss_tensor = torch.tensor(0.0, device=device)
for x, y in loader:
    loss = criterion(model(x), y)
    running_loss_tensor += loss.detach()   # stays on GPU
epoch_loss = running_loss_tensor.item() / len(loader)   # one sync
```

### Bug 2: Blocking H2D transfer
```python
# WRONG: blocks CPU until copy completes
x = x_cpu.to(device)

# CORRECT: non-blocking; overlaps copy with CPU work
x = x_cpu.to(device, non_blocking=True)   # requires pin_memory=True in DataLoader
```

### Bug 3: No `torch.no_grad()` in evaluation
```python
# WRONG: builds computation graph; wastes memory and time
def evaluate(model, x):
    return model(x).mean()   # autograd is active!

# CORRECT: disable graph building for inference
def evaluate(model, x):
    with torch.no_grad():
        return model(x).mean()
```

**Note:** `model.eval()` changes BatchNorm/Dropout behaviour but does NOT disable gradients. You need both.

## Flamegraph Evidence

| Bug | What the flamegraph shows |
|-----|--------------------------|
| `.item()` sync | Wide `cudaDeviceSynchronize` bands at regular intervals |
| Blocking H2D | Wide `cudaMemcpyAsync` stacks blocking CPU thread |
| No no_grad | Extra autograd engine nodes in evaluation call stacks |

## Generating Real Flamegraphs

```bash
pip install py-spy
py-spy record -o before.svg -- python 19.1_slow_training_analysis.py
py-spy record -o after.svg  -- python 19.2_optimised_training.py
# Open SVGs in browser; compare wide stacks
```
