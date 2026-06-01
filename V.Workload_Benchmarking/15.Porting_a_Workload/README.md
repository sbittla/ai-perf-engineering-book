# Chapter 15 — Porting a Workload

Three exercises covering the full process of moving a workload from CPU to GPU: the porting checklist, understanding how the bottleneck shifts at each optimization rung, and tuning the DataLoader to prevent GPU starvation.

## Exercises

| File | Topic | Key Functions |
|------|-------|---------------|
| `15.1_porting_checklist.py` | 6-step porting checklist, dtype traps, H2D transfer cost, output validation | `safe_from_numpy()`, `measure_h2d_bandwidth_gbs()`, `validate_port()` |
| `15.2_bottleneck_shift.py` | Optimization ladder, bottleneck diagnosis, Python overhead floor | `benchmark_config()`, `throughput_scaling_ratio()`, `measure_overhead_ms()` |
| `15.3_dataloader_at_scale.py` | DataLoader throughput, num_workers sweep, GPU idle time, CPU affinity | `measure_loader_throughput()`, `sweep_num_workers()`, `measure_idle_pct()` |

## Quick Start

```bash
python 15.1_porting_checklist.py
python 15.2_bottleneck_shift.py
python 15.3_dataloader_at_scale.py
```

## Key Concepts

### The 6-Step Porting Checklist (15.1)

1. **Profile first** — identify the hotspot before porting
2. **Validate dtype** — NumPy defaults to float64; GPU prefers float32
3. **Validate shape** — `.T` is non-contiguous; call `.contiguous()` before CUDA kernels
4. **Measure H2D** — PCIe is 60× slower than HBM; minimise copies in the hot path
5. **Port hotspot** — only port the bottleneck; bookkeeping stays on CPU
6. **Validate output** — `max_abs_diff < 1e-4` for float32 operations

**Common dtype trap:**
```python
arr = np.array([1.0, 2.0])        # float64 (NumPy default)
t = torch.from_numpy(arr).float()  # float32 ← correct
t = t.to("cuda")
```

### The Bottleneck Shift (15.2)

| Rung | Configuration | Bottleneck | Speedup potential |
|------|--------------|------------|-------------------|
| 1 | CPU FP32 | Compute (100 GFLOPS) | baseline |
| 2 | GPU FP32 | Memory bandwidth | 10–50× |
| 3 | GPU FP16 | Tensor Core utilization | 1.5–2× |
| 4 | + torch.compile | Python overhead / kernel fusion | 1.2–1.8× |
| 5 | + INT8 quant | Cache I/O pattern | 1.5–3× |

**Python overhead floor:**
```
overhead_ms = wall_clock_ms - gpu_kernel_ms_from_cuda_events
```
If overhead > 30% of wall time → use `torch.compile` or CUDA Graphs.

**Throughput scaling ratio:**
- > 30×: strong compute scaling (well utilized GPU)
- 5–30×: moderate (approaching bandwidth ceiling)
- < 5×: memory-bound or overhead-dominated

### DataLoader Tuning (15.3)

**GPU idle pct (starvation):**
```
idle_pct = max(0, total_wall_ms - total_gpu_ms) / total_wall_ms × 100
```
- `idle_pct > 10%`: DataLoader is the bottleneck
- Fix: increase `num_workers`, add `pin_memory=True`, raise `prefetch_factor`

**Optimal num_workers sweep recipe:**
```python
for nw in [0, 1, 2, 4, 8]:
    loader = DataLoader(ds, batch_size=64, num_workers=nw,
                        pin_memory=True, persistent_workers=(nw > 0))
    # measure batches/sec and pick the knee
```

**Full tuning checklist:**
- `num_workers`: sweep to find the knee (start with `N_CPUs // 2`)
- `pin_memory=True`: always on CUDA systems
- `persistent_workers=True`: avoids per-epoch worker respawn cost
- `prefetch_factor=4`: if `idle_pct > 10%`
- `drop_last=True`: avoid variable last-batch shape overhead
- `worker_init_fn` with `os.sched_setaffinity`: NUMA systems only
