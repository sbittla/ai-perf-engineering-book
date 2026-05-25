# Chapter 17 — Capstone 2: DataLoader Bottleneck Hunt

A detective exercise: find three hidden DataLoader bottlenecks, fix them one at a time, and measure the improvement from each fix.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `17.1_slow_dataloader.py` | Diagnose the bottleneck; measure idle_pct and per-bug costs | `diagnose_dataloader()` |
| `17.2_fast_dataloader.py` | Apply each fix independently; build before/after table | `fix1_workers()`, `fix1_fix2()`, `all_fixes()` |

## Workflow

```bash
python 17.1_slow_dataloader.py   # diagnoses bottleneck; saves /tmp/capstone17_diagnosis.json
python 17.2_fast_dataloader.py   # applies fixes; saves /tmp/capstone17_report.json
```

## The Three Bugs

| Bug | Description | Fix | Typical gain |
|-----|-------------|-----|-------------|
| 1 | `num_workers=0` — single-threaded loading | `num_workers=N` | 3–10× for I/O-heavy datasets |
| 2 | No `pin_memory` — staging copy on every H2D | `pin_memory=True` | 1.5–2× H2D bandwidth |
| 3 | Slow per-sample augmentation (simulated) | Pre-cache or GPU-side augment | Depends on augmentation cost |

## Key Metrics

**GPU idle percentage:**
```python
idle_pct = (total_wall_ms - total_gpu_ms) / total_wall_ms * 100
# > 10%: DataLoader is the bottleneck
```

**Isolation protocol:** fix ONE bug per experiment; report speedup of each fix before combining.

## When Workers Don't Help

For purely in-memory datasets (no disk I/O), `num_workers > 0` can be slower due to IPC overhead. Workers are always beneficial when:
- Data is loaded from disk (SSD or HDD)
- Preprocessing is CPU-heavy (JPEG decode, augmentation)
- `sleep_ms` per sample > GPU compute time per batch / batch_size
